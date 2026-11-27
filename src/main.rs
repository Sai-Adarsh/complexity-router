use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::HashMap, env, fs::File, io::{BufRead, BufReader}, time::Duration};

mod paired;
#[cfg(test)]
mod tests;

const MAX_UPSTREAM_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Tier {
    Small,
    Large,
}

impl Tier {
    fn name(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Large => "large",
        }
    }
}

#[derive(Serialize)]
struct Decision {
    route: Tier,
    words: usize,
    token_entropy_bits: f64,
    score: f64,
}

fn classify(prompt: &str, threshold: f64, force_large: bool) -> Decision {
    let mut counts = HashMap::new();
    let mut words = 0;
    for token in prompt.split(|c: char| !c.is_alphanumeric()).filter(|word| !word.is_empty()) {
        *counts.entry(token.to_lowercase()).or_insert(0usize) += 1;
        words += 1;
    }
    let mut frequencies: Vec<_> = counts.into_values().collect();
    frequencies.sort_unstable();
    let token_entropy_bits = frequencies.iter().map(|&count| {
        let probability = count as f64 / words as f64;
        -probability * probability.log2()
    }).sum::<f64>();
    let lower = prompt.to_lowercase();
    let reasoning_markers = ["step by step", "trade-off", "derive", "prove", "debug", "compare", "design", "why"];
    let marker_count = reasoning_markers.iter().filter(|marker| {
        if marker.contains([' ', '-']) {
            lower.contains(**marker)
        } else {
            lower.split(|c: char| !c.is_alphanumeric()).any(|word| word == **marker)
        }
    }).count();
    let score = 0.65 * token_entropy_bits + 0.8 * marker_count as f64
        + 0.5 * (1.0 + words as f64 / 32.0).log2()
        + if prompt.contains("```") { 1.5 } else { 0.0 };
    let route = route_score(score, threshold, force_large);
    Decision { route, words, token_entropy_bits, score }
}

fn route_score(score: f64, threshold: f64, force_large: bool) -> Tier {
    if force_large || score >= threshold { Tier::Large } else { Tier::Small }
}

#[derive(Clone)]
struct Backend {
    url: String,
    model: String,
    api_key: Option<String>,
}

impl Backend {
    fn from_env(prefix: &str, default_model: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let base = env::var(format!("{prefix}_BASE_URL"))?;
        let url = reqwest::Url::parse(&format!("{}/chat/completions", base.trim_end_matches('/')))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(format!("{prefix}_BASE_URL must use HTTP or HTTPS").into());
        }
        Ok(Self {
            url: url.to_string(),
            model: env::var(format!("{prefix}_MODEL")).unwrap_or_else(|_| default_model.into()),
            api_key: env::var(format!("{prefix}_API_KEY")).ok(),
        })
    }
}

#[derive(Clone)]
struct AppState {
    client: reqwest::Client,
    small: Backend,
    large: Backend,
    threshold: f64,
    allow_large_fallback: bool,
}

impl AppState {
    async fn send(&self, tier: Tier, body: &mut Value) -> reqwest::Result<reqwest::Response> {
        let backend = match tier {
            Tier::Small => &self.small,
            Tier::Large => &self.large,
        };
        body["model"] = json!(backend.model);
        let mut request = self.client.post(&backend.url).json(body);
        if let Some(key) = &backend.api_key {
            request = request.bearer_auth(key);
        }
        request.send().await
    }
}

fn failure(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({ "error": { "message": message } }))).into_response()
}

fn route_input(body: &Value) -> Result<(String, bool), &'static str> {
    if body.get("stream").is_some_and(|stream| stream != false) {
        return Err("Use stream=false with this router.");
    }
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return Err("messages must be an array.");
    };
    let mut context = Vec::new();
    let mut force_large = body.get("tools").is_some_and(|tools| !tools.is_null());
    let mut has_user = false;
    for message in messages {
        match message.get("role").and_then(Value::as_str) {
            Some("user") => has_user = true,
            Some("system" | "developer" | "assistant" | "tool") => {}
            _ => return Err("Each message needs a supported role."),
        }
        if let Some(content) = message.get("content") {
            match content {
                Value::String(text) => context.push(text.as_str()),
                Value::Null => {}
                _ => force_large = true,
            }
        }
    }
    if !has_user {
        return Err("Include a user message.");
    }
    Ok((context.join("\n"), force_large))
}

async fn complete(State(state): State<AppState>, Json(mut body): Json<Value>) -> Response {
    let (context, force_large) = match route_input(&body) {
        Ok(input) => input,
        Err(error) => return failure(StatusCode::BAD_REQUEST, error),
    };
    let decision = classify(&context, state.threshold, force_large);
    let mut tier = decision.route;
    let mut reply = state.send(tier, &mut body).await;
    let fallback = if state.allow_large_fallback && tier == Tier::Small {
        match reply.as_ref().map(reqwest::Response::status) {
            Ok(StatusCode::TOO_MANY_REQUESTS) => Some("small_http_429"),
            Ok(StatusCode::SERVICE_UNAVAILABLE) => Some("small_http_503"),
            _ => None,
        }
    } else {
        None
    };
    if fallback.is_some() {
        drop(reply);
        tier = Tier::Large;
        reply = state.send(tier, &mut body).await;
    }
    let mut response = match reply {
        Ok(upstream) => forward(upstream).await,
        Err(error) if error.is_timeout() => failure(StatusCode::GATEWAY_TIMEOUT, "Model request timed out."),
        Err(_) => failure(StatusCode::BAD_GATEWAY, "Model endpoint is unavailable."),
    };
    let headers = response.headers_mut();
    headers.insert("x-inference-tier", header::HeaderValue::from_static(tier.name()));
    if let Some(reason) = fallback {
        headers.insert("x-inference-fallback", header::HeaderValue::from_static(reason));
    }
    response
}

async fn forward(mut upstream: reqwest::Response) -> Response {
    let too_large = "Model response exceeds 4 MiB.";
    if upstream
        .content_length()
        .is_some_and(|length| length > MAX_UPSTREAM_RESPONSE_BYTES as u64)
    {
        return failure(StatusCode::BAD_GATEWAY, too_large);
    }
    let status = upstream.status();
    let mut headers = HeaderMap::new();
    for name in [header::CONTENT_TYPE, header::RETRY_AFTER] {
        if let Some(value) = upstream.headers().get(&name) {
            headers.insert(name, value.clone());
        }
    }
    let mut body = Vec::new();
    loop {
        match upstream.chunk().await {
            Ok(Some(chunk)) => {
                if chunk.len() > MAX_UPSTREAM_RESPONSE_BYTES - body.len() {
                    return failure(StatusCode::BAD_GATEWAY, too_large);
                }
                body.extend_from_slice(&chunk);
            }
            Ok(None) => return (status, headers, body).into_response(),
            Err(_) => return failure(StatusCode::BAD_GATEWAY, "Model response was incomplete."),
        }
    }
}

#[derive(Deserialize)]
struct LabeledPrompt {
    prompt: String,
    route: Tier,
}

fn evaluate(file: &str, threshold: f64) -> Result<(), Box<dyn std::error::Error>> {
    let mut correct = 0;
    let mut total = 0;
    let mut routed_small = 0;
    let mut complex_routed_small = 0;
    for (index, line) in BufReader::new(File::open(file)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let labeled: LabeledPrompt = serde_json::from_str(&line)
            .map_err(|error| format!("{}:{}: {error}", file, index + 1))?;
        let route = classify(&labeled.prompt, threshold, false).route;
        correct += usize::from(route == labeled.route);
        routed_small += usize::from(route == Tier::Small);
        complex_routed_small += usize::from(route == Tier::Small && labeled.route == Tier::Large);
        total += 1;
    }
    if total == 0 {
        return Err("No labeled prompts".into());
    }
    println!("{}", serde_json::to_string_pretty(&json!({
        "samples": total,
        "threshold": threshold,
        "label_accuracy": correct as f64 / total as f64,
        "small_model_share": routed_small as f64 / total as f64,
        "complex_routed_small": complex_routed_small
    }))?);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let threshold: f64 = env::var("ROUTE_THRESHOLD").unwrap_or_else(|_| "3.2".into()).parse()?;
    if !threshold.is_finite() || threshold <= 0.0 {
        return Err("ROUTE_THRESHOLD must be finite and positive".into());
    }
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["demo"] => {
            for prompt in ["Translate hello to French.", "Design a cache for a GPU inference fleet. Compare the trade-offs and derive its cost."] {
                println!("{}", json!({"prompt": prompt, "decision": classify(prompt, threshold, false)}));
            }
            Ok(())
        }
        ["evaluate", file] => evaluate(file, threshold),
        ["evaluate-paired", records, rates, thresholds @ ..] => {
            paired::evaluate(records, rates, thresholds, threshold)
        }
        ["serve"] | [] => {
            let allow_large_fallback: bool = env::var("ROUTER_ALLOW_LARGE_FALLBACK")
                .unwrap_or_else(|_| "false".into()).parse()
                .map_err(|_| "ROUTER_ALLOW_LARGE_FALLBACK must be true or false")?;
            let state = AppState {
                client: reqwest::Client::builder().timeout(Duration::from_secs(120))
                    .redirect(reqwest::redirect::Policy::none()).build()?,
                small: Backend::from_env("SMALL", "quantized-8b")?,
                large: Backend::from_env("LARGE", "reasoning-405b")?,
                threshold,
                allow_large_fallback,
            };
            let app = Router::new()
                .route("/healthz", get(|| async { Bytes::from_static(b"ok\n") }))
                .route("/v1/chat/completions", post(complete))
                .layer(DefaultBodyLimit::max(1024 * 1024))
                .with_state(state);
            let bind = env::var("ROUTER_BIND").unwrap_or_else(|_| "127.0.0.1:8010".into());
            let listener = tokio::net::TcpListener::bind(bind.as_str()).await.map_err(|error| {
                std::io::Error::new(error.kind(), format!("bind complexity router to {bind}: {error}"))
            })?;
            eprintln!("Complexity router listening on {}", listener.local_addr()?);
            axum::serve(listener, app).with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            }).await?;
            Ok(())
        }
        _ => Err("usage: complexity-router [demo | evaluate PROMPTS.jsonl | evaluate-paired RECORDS.jsonl RATES.json [THRESHOLD ...] | serve]".into()),
    }
}
