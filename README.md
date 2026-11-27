# Multi-model complexity router

This Rust HTTP gateway routes chat requests between a quantized 8B model and a larger reasoning model. The routing score combines Shannon entropy over case-folded words, prompt length, reasoning terms, and code blocks. It scores plain strings and text-only content arrays, joining text parts with spaces for scoring. Tool and multimodal requests go directly to the large model. The gateway replaces the request's model ID and adds `x-inference-tier` to the upstream response.

## Run

```sh
cargo run --release -- demo
cargo run --release -- evaluate examples.jsonl

export SMALL_BASE_URL=http://127.0.0.1:8000/v1 SMALL_MODEL=quantized-8b
export LARGE_BASE_URL=https://your-provider.example/v1 LARGE_MODEL=reasoning-405b
# Configure SMALL_API_KEY or LARGE_API_KEY in your environment when required.
cargo run --release -- serve
```

```sh
curl -i http://127.0.0.1:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Translate hello to French."}],"max_tokens":30}'
```

The gateway listens on loopback by default. Set `ROUTER_BIND` to change the address and `ROUTE_THRESHOLD` to change the score cutoff. It supports nonstreaming chat completions. [vLLM serving API](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/).

Set `ROUTER_ALLOW_LARGE_FALLBACK=true` to try the large backend once when a small-backend request receives HTTP 429 or 503. Use this option when the large backend is approved to receive the same payload and its cost is acceptable. The response gives the final attempted tier in `x-inference-tier` and the small-backend status in `x-inference-fallback` when fallback occurs. The router forwards the final backend's `Retry-After` header. Each upstream attempt has a 120-second timeout. Connection errors and timeouts end after the first attempt because the upstream may still process them. The paired evaluator measures routing under available backends.

The router returns HTTP 502 if the final upstream response body exceeds 4 MiB.

## Evaluate

`evaluate examples.jsonl` checks route labels. Run `evaluate-paired` to compare spend and quality on the same requests across thresholds and an always-large baseline.

```sh
cargo run --locked --release -- evaluate-paired fixtures/paired.jsonl fixtures/prices.json 1.8 3.2 5.0 8.0
```

The bundled corpus has six synthetic pairs. Its token counts, grades, and prices are invented to exercise the evaluator. At 1.8 the fixture saves 1.43% with zero quality loss. At 8.0 it saves 60.43% and regresses on two requests. The `data_kind` field travels with the report.

For measured data, run each held-out chat request against both model versions with the same parameters. Copy each response's input and output token counts from provider usage into `small` and `large`. Blind-grade each output with the same rubric on a 0–1 scale. Give each pair a unique `id`, preserve the request body in `request`, and record the matching model IDs and per-million token prices in the rate card. Use `data_kind: "measured"` when the usage and grades come from those runs. Keep provider credentials out of the files. Evaluate thresholds on a separate calibration set before reporting the selected threshold on held-out traffic.

The report prices each selected response's input and output tokens separately. It applies the live router's scoring and forced-large rules to the recorded request. The signed mean quality delta can hide regressions when other responses improve, so the report also averages `max(large_quality - selected_quality, 0)` across all requests. It lists regression counts and the ten worst request IDs with scores and dollar savings. Open those requests in your grading set to see which cases a lower threshold would recover. Savings use the two flat rates supplied per model. Account separately for pricing features such as cached-input discounts.
