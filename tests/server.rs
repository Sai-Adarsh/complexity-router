#![cfg(unix)]

use axum::{
    Json, Router,
    http::{HeaderMap, StatusCode},
    routing::post,
};
use serde_json::{Value, json};
use std::{
    io::{ErrorKind, Read},
    net::TcpListener as Reservation,
    process::{Child, Command, ExitStatus, Stdio},
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::net::TcpListener;

const BIN: &str = env!("CARGO_BIN_EXE_complexity-router");

struct RouterProcess(Child);

impl RouterProcess {
    async fn start(upstream_url: &str, allow_fallback: bool) -> (Self, reqwest::Client, String) {
        let reservation = Reservation::bind("127.0.0.1:0").unwrap();
        let addr = reservation.local_addr().unwrap().to_string();
        drop(reservation);
        let mut command = Command::new(BIN);
        command
            .arg("serve")
            .env("ROUTE_THRESHOLD", "3.2")
            .env("SMALL_BASE_URL", upstream_url)
            .env_remove("SMALL_MODEL")
            .env("SMALL_API_KEY", "small-fixture-credential")
            .env("LARGE_BASE_URL", upstream_url)
            .env("LARGE_MODEL", "larger-fixture")
            .env("LARGE_API_KEY", "large-fixture-credential")
            .env("ROUTER_BIND", &addr)
            .env("TOKIO_WORKER_THREADS", "2")
            .env_remove("ROUTER_ALLOW_LARGE_FALLBACK");
        if allow_fallback {
            command.env("ROUTER_ALLOW_LARGE_FALLBACK", "true");
        }
        let mut router = Self(command.spawn().unwrap());
        let base = format!("http://{addr}");
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            assert!(
                router.0.try_wait().unwrap().is_none(),
                "router exited before listening"
            );
            if let Ok(response) = client.get(format!("{base}/healthz")).send().await {
                assert_eq!(response.text().await.unwrap(), "ok\n");
                return (router, client, base);
            }
            assert!(
                std::time::Instant::now() < deadline,
                "router failed to start"
            );
            tokio::time::sleep(Duration::from_millis(15)).await;
        }
    }

    fn interrupt(&mut self) {
        let status = Command::new("kill")
            .args(["-INT", &self.0.id().to_string()])
            .status()
            .unwrap();
        assert!(status.success());
        let status = self.wait_for_exit();
        assert!(status.success(), "router exited ungracefully: {status}");
    }

    fn wait_for_exit(&mut self) -> ExitStatus {
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                return status;
            }
            assert!(std::time::Instant::now() < deadline, "router did not exit");
            std::thread::sleep(Duration::from_millis(15));
        }
    }
}

impl Drop for RouterProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
async fn serves_chat_and_shuts_down_cleanly_on_interrupt() {
    let upstream = Router::new().route(
        "/v1/chat/completions",
        post(|Json(value): Json<Value>| async move {
            Json(json!({"model": value["model"], "max_tokens": value["max_tokens"]}))
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream_url = format!("http://{}/v1/", listener.local_addr().unwrap());
    let upstream = tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });
    let (mut router, client, base) = RouterProcess::start(&upstream_url, false).await;
    let reply = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "stream": false,
            "messages":[{"role":"user","content":"Hi"}],
            "max_tokens": 43
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(reply.headers()["x-inference-tier"], "small");
    assert_eq!(
        reply.json::<Value>().await.unwrap(),
        json!({"model":"quantized-8b", "max_tokens":43})
    );
    router.interrupt();
    upstream.abort();
}

#[tokio::test]
async fn falls_back_only_for_explicit_capacity_rejections_when_enabled() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let captured = calls.clone();
    let upstream = Router::new().route(
        "/v1/chat/completions",
        post(move |headers: HeaderMap, Json(body): Json<Value>| {
            let captured = captured.clone();
            async move {
                let model = body["model"].as_str().unwrap();
                let prompt = body["messages"][0]["content"].as_str().unwrap();
                let status = match (model, prompt) {
                    ("quantized-8b", "limited") => StatusCode::TOO_MANY_REQUESTS,
                    ("quantized-8b", "busy" | "full") => StatusCode::SERVICE_UNAVAILABLE,
                    ("quantized-8b", "invalid") => StatusCode::BAD_REQUEST,
                    ("quantized-8b", "error") => StatusCode::INTERNAL_SERVER_ERROR,
                    ("larger-fixture", "full") => StatusCode::SERVICE_UNAVAILABLE,
                    _ => StatusCode::OK,
                };
                let auth = headers["authorization"].to_str().unwrap().to_owned();
                captured
                    .lock()
                    .unwrap()
                    .push((model.to_owned(), auth, body["max_tokens"].clone()));
                (status, [("Retry-After", "7")], Json(json!({"model":model})))
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let upstream = tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });
    for (option, prompt, forced, expected, tier, reason) in [
        (
            false,
            "limited",
            false,
            StatusCode::TOO_MANY_REQUESTS,
            "small",
            None,
        ),
        (
            true,
            "limited",
            false,
            StatusCode::OK,
            "large",
            Some("small_http_429"),
        ),
        (
            true,
            "busy",
            false,
            StatusCode::OK,
            "large",
            Some("small_http_503"),
        ),
        (
            true,
            "full",
            false,
            StatusCode::SERVICE_UNAVAILABLE,
            "large",
            Some("small_http_503"),
        ),
        (
            true,
            "invalid",
            false,
            StatusCode::BAD_REQUEST,
            "small",
            None,
        ),
        (
            true,
            "error",
            false,
            StatusCode::INTERNAL_SERVER_ERROR,
            "small",
            None,
        ),
        (
            true,
            "full",
            true,
            StatusCode::SERVICE_UNAVAILABLE,
            "large",
            None,
        ),
    ] {
        let (mut router, client, base) = RouterProcess::start(&url, option).await;
        let mut body = json!({"messages":[{"role":"user","content":prompt}],"max_tokens":43});
        if forced {
            body["tools"] = json!([]);
        }
        let response = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), expected, "{prompt}");
        assert_eq!(response.headers()["x-inference-tier"], tier);
        assert_eq!(
            response
                .headers()
                .get("x-inference-fallback")
                .map(|v| v.to_str().unwrap()),
            reason
        );
        if expected == StatusCode::TOO_MANY_REQUESTS || expected == StatusCode::SERVICE_UNAVAILABLE
        {
            assert_eq!(response.headers()["retry-after"], "7");
        }
        let observed = std::mem::take(&mut *calls.lock().unwrap());
        assert_eq!(observed.len(), if reason.is_some() { 2 } else { 1 });
        for (model, auth, max_tokens) in observed {
            let expected_auth = if model == "quantized-8b" {
                "small"
            } else {
                "large"
            };
            assert_eq!(auth, format!("Bearer {expected_auth}-fixture-credential"));
            assert_eq!(max_tokens, 43);
        }
        router.interrupt();
    }
    upstream.abort();
}

#[test]
fn defaults_to_the_documented_loopback_port_without_an_override() {
    let _port = match Reservation::bind("127.0.0.1:8010") {
        Ok(listener) => Some(listener),
        Err(error) if error.kind() == ErrorKind::AddrInUse => None,
        Err(error) => panic!("reserve the default port: {error}"),
    };
    let mut router = RouterProcess(
        Command::new(BIN)
            .arg("serve")
            .env_remove("ROUTER_BIND")
            .env("ROUTE_THRESHOLD", "3.2")
            .env("SMALL_BASE_URL", "http://127.0.0.1:1/")
            .env("LARGE_BASE_URL", "http://127.0.0.1:1/")
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let status = router.wait_for_exit();
    let mut stderr = String::new();
    router
        .0
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert!(!status.success());
    assert!(stderr.contains("AddrInUse"), "{stderr}");
}

#[test]
fn rejects_unsupported_backend_urls_at_startup() {
    let output = Command::new(BIN)
        .arg("serve")
        .env("SMALL_BASE_URL", "ftp://127.0.0.1")
        .env("ROUTE_THRESHOLD", "3.2")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("SMALL_BASE_URL must use HTTP or HTTPS")
    );
}
