# Multi-model complexity router

[Quick start](../README.md). Run commands from the repository root.

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

The gateway serves nonstreaming chat completions and binds to loopback by default. Set `ROUTER_BIND` to change the listen address. [vLLM serving API](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)

## Routing and fallback

The score combines Shannon entropy over case-folded words, prompt length, reasoning terms, and code blocks. Set `ROUTE_THRESHOLD` to change the cutoff. Tool and multimodal requests go directly to the large model. The gateway replaces the request's model ID and reports the final attempted tier in `x-inference-tier`.

Set `ROUTER_ALLOW_LARGE_FALLBACK=true` to try the large backend once after HTTP 429 or 503 from the small backend. Enable it only if the large backend is approved to receive the same payload and its cost is acceptable.

- When fallback occurs, `x-inference-fallback` gives the small backend's status.
- The router forwards the final backend's `Retry-After` header.
- Each upstream attempt has a 120-second timeout. Connection errors and timeouts end after the first attempt because the upstream may still process the request.

The router returns HTTP 502 if the final upstream response body exceeds 4 MiB.

## Evaluate

`evaluate examples.jsonl` checks route labels. Run `evaluate-paired` to compare spend and quality on the same requests across thresholds and an always-large baseline.

```sh
cargo run --locked --release -- evaluate-paired fixtures/paired.jsonl fixtures/prices.json 1.8 3.2 5.0 8.0
```

### Synthetic fixture

The bundled corpus has six synthetic pairs with invented token counts, grades, and prices. The report carries the `data_kind` field.

| Threshold | Savings | Quality outcome |
| --- | ---: | --- |
| 1.8 | 1.43% | Zero quality loss |
| 8.0 | 60.43% | Two requests regressed |

### Measure on held-out requests

1. Run each held-out chat request against both model versions with the same parameters.
2. Copy each response's input and output token counts from provider usage into `small` and `large`. Blind-grade each output with the same rubric on a 0–1 scale.
3. Give each pair a unique `id` and preserve the request body in `request`. Record the matching model IDs and per-million token prices in the rate card. Use `data_kind: "measured"` when the usage and grades come from those runs. Keep provider credentials out of the files.
4. Evaluate thresholds on a separate calibration set before reporting the selected threshold on held-out traffic.

### Read the report

The paired evaluator uses the live router's scoring and forced-large rules on each recorded request and measures routing under available backends. It prices each selected response's input and output tokens separately. Savings use the two flat rates per model. Account separately for pricing features such as cached-input discounts.

The signed mean quality delta can hide regressions when other responses improve. The report also averages `max(large_quality - selected_quality, 0)` across all requests and lists the regression count. It includes the ten worst request IDs with scores and dollar savings. Look up those IDs in the grading set to see which requests a lower threshold would recover.
