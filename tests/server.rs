#![cfg(unix)]

use axum::{Json, Router, routing::post};
use serde_json::{Value, json};
use std::{
    net::TcpListener as Reservation,
    process::{Child, Command},
    time::Duration,
};
use tokio::net::TcpListener;

const BIN: &str = env!("CARGO_BIN_EXE_complexity-router");

struct RouterProcess(Child);

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
    let reservation = Reservation::bind("127.0.0.1:0").unwrap();
    let addr = reservation.local_addr().unwrap().to_string();
    drop(reservation);
    let mut router = RouterProcess(
        Command::new(BIN)
            .arg("serve")
            .env("ROUTE_THRESHOLD", "3.2")
            .env("SMALL_BASE_URL", &upstream_url)
            .env_remove("SMALL_MODEL")
            .env("LARGE_BASE_URL", &upstream_url)
            .env("LARGE_MODEL", "larger-fixture")
            .env("ROUTER_BIND", &addr)
            .env("TOKIO_WORKER_THREADS", "2")
            .spawn()
            .unwrap(),
    );
    let base = format!("http://{addr}");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(300))
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
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "router failed to start"
        );
        std::thread::sleep(Duration::from_millis(15));
    }
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
    let interrupt = Command::new("kill")
        .args(["-INT", &router.0.id().to_string()])
        .status()
        .unwrap();
    assert!(interrupt.success());
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    let status = loop {
        if let Some(status) = router.0.try_wait().unwrap() {
            break status;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "router did not exit on SIGINT"
        );
        std::thread::sleep(Duration::from_millis(15));
    };
    assert!(status.success(), "router exited ungracefully: {status}");
    upstream.abort();
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
