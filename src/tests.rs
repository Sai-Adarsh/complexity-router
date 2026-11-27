use super::*;
use axum::{body::to_bytes, extract::Request};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

async fn state() -> (AppState, tokio::task::JoinHandle<()>) {
    let upstream = Router::new().route(
        "/chat/completions",
        post(|headers: HeaderMap, Json(body): Json<Value>| async move {
            let status = if body["model"] == "large-backend" {
                StatusCode::TOO_MANY_REQUESTS
            } else {
                StatusCode::OK
            };
            (
                status,
                Json(json!({
                    "model": body["model"],
                    "auth": headers.get(header::AUTHORIZATION).map(|value| value.to_str().unwrap()),
                    "max_tokens": body["max_tokens"],
                })),
            )
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/chat/completions", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });
    let state = AppState {
        client: reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap(),
        small: Backend {
            url: url.clone(),
            model: "small-backend".into(),
            api_key: None,
        },
        large: Backend {
            url,
            model: "large-backend".into(),
            api_key: Some("fixture-key".into()),
        },
        threshold: 3.2,
        allow_large_fallback: false,
    };
    (state, task)
}

async fn read(response: Response) -> (StatusCode, HeaderMap, Value) {
    let (parts, body) = response.into_parts();
    let bytes = to_bytes(body, 4096).await.unwrap();
    (
        parts.status,
        parts.headers,
        serde_json::from_slice(&bytes).unwrap(),
    )
}

#[tokio::test]
async fn forwards_chat_and_forces_large_for_tools_and_multimodal_parts() {
    let (state, upstream) = state().await;
    let mut chat = json!({
        "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": null}],
        "model": "untrusted", "stream": false, "max_tokens": 10
    });
    let (code, headers, body) =
        read(complete(State(state.clone()), Json(chat.clone())).await).await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(headers["x-inference-tier"], "small");
    assert_eq!(
        body,
        json!({"model":"small-backend", "auth":null, "max_tokens":10})
    );
    chat["tools"] = json!([]);
    let (code, headers, body) = read(complete(State(state.clone()), Json(chat)).await).await;
    assert_eq!(code, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(headers["x-inference-tier"], "large");
    assert_eq!(headers[header::CONTENT_TYPE], "application/json");
    assert_eq!(body["auth"], "Bearer fixture-key");
    let image = json!({"messages":[{"role":"user", "content":[{"type":"text","text":"Analyze"}]}]});
    let (_, headers, _) = read(complete(State(state), Json(image)).await).await;
    assert_eq!(headers["x-inference-tier"], "large");
    upstream.abort();
}

#[tokio::test]
async fn malformed_chat_fails_before_contacting_the_upstream() {
    let (state, upstream) = state().await;
    upstream.abort();
    for (body, expected) in [
        (json!({"stream":true}), "Use stream=false"),
        (json!({}), "messages must be an array"),
        (json!({"messages":[{"role":"unknown"}]}), "supported role"),
        (
            json!({"messages":[{"role":"system"}]}),
            "Include a user message",
        ),
    ] {
        let (status, _, body) = read(complete(State(state.clone()), Json(body)).await).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains(expected)
        );
    }
}

#[tokio::test]
async fn reports_unreachable_timed_out_and_incomplete_upstream_responses() {
    let request = json!({"messages":[{"role":"user","content":"Hi"}]});
    let (mut state, upstream) = state().await;
    upstream.abort();
    upstream.await.unwrap_err();
    let reply = complete(State(state.clone()), Json(request.clone())).await;
    assert_eq!(reply.status(), StatusCode::BAD_GATEWAY);

    let silent = Router::new().route(
        "/chat",
        post(|_: Request| async { std::future::pending::<StatusCode>().await }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    state.small.url = format!("http://{}/chat", listener.local_addr().unwrap());
    state.client = reqwest::Client::builder()
        .timeout(Duration::from_millis(150))
        .build()
        .unwrap();
    let upstream = tokio::spawn(async move { axum::serve(listener, silent).await.unwrap() });
    let reply = complete(State(state.clone()), Json(request.clone())).await;
    assert_eq!(reply.status(), StatusCode::GATEWAY_TIMEOUT);
    upstream.abort();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    state.small.url = format!("http://{}", listener.local_addr().unwrap());
    state.client = reqwest::Client::new();
    let upstream = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = [0; 4096];
        assert!(stream.read(&mut request).await.unwrap() > 0);
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\nshort")
            .await
            .unwrap();
        stream.shutdown().await.unwrap();
    });
    let (status, _, body) = read(complete(State(state), Json(request)).await).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(body["error"]["message"], "Model response was incomplete.");
    upstream.await.unwrap();
}

#[test]
fn entropy_and_score_respect_words_code_fences_and_cutoffs() {
    let repeat = classify("Echo eChO echo", 3.2, false);
    assert_eq!(repeat.words, 3);
    assert_eq!(repeat.token_entropy_bits, 0.0);
    assert!(classify("derive compare ```code```", 3.2, false).score > 3.2);
    assert!(matches!(classify("", 3.2, false).route, Tier::Small));
    assert!(matches!(route_score(3.2, 3.2, false), Tier::Large));
}
