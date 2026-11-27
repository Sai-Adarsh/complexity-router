# Multi-model complexity router

This Rust HTTP gateway routes chat requests between a quantized 8B model and a larger reasoning model. The routing score combines Shannon entropy over case-folded words, prompt length, reasoning terms, and code blocks. Tool and multimodal requests go directly to the large model. The gateway replaces the request's model ID and adds `x-inference-tier` to the upstream response.

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

## Evaluate

The bundled labels exercise routing decisions. To assess quality and test the 60% savings target, run a representative held-out set through the router and the large model alone. Grade outputs and price actual input and output tokens at provider rates.
