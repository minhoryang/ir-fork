# Design: Ollama Remote Model Support (v0.14.1 PoC)

**Branch:** `v0.14.1-ollama-poc`
**Date:** 2026-05-04
**Source:** Port of `minho/poc-ollama` (based on v0.12.0) to v0.14.1

---

## Problem

`ir` requires local GGUF model files for all LLM work. Users running Ollama (local or remote) cannot
reuse those models without downloading large GGUFs. This PoC makes `IR_EMBEDDING_MODEL` and
`IR_COMBINED_MODEL` accept an `ollama://` scheme URL to delegate inference to an Ollama server.

---

## Goals

1. `IR_EMBEDDING_MODEL=ollama://host:port/model` routes embedding calls to Ollama `/api/embed`.
2. `IR_COMBINED_MODEL=ollama://host:port/model` routes expansion and reranking to Ollama `/api/generate`.
3. Existing local/HuggingFace env values are unaffected.
4. Local file validation in `prepare_model_envs` is skipped for `ollama://` values.
5. All implementation changes stay inside `src/llm/*` and `Cargo.toml`.

## Non-goals

- No CLI surface changes.
- No daemon/pipeline changes.
- No README updates (PoC).
- No generic multi-provider abstraction.
- No `IR_EXPANDER_MODEL` or `IR_RERANKER_MODEL` Ollama support (parser accepts it; behavior untested).

---

## Env format

```bash
IR_EMBEDDING_MODEL=ollama://127.0.0.1:11434/embeddinggemma:300m
IR_COMBINED_MODEL=ollama://127.0.0.1:11434/qwen3.5:2b
```

Parsing rules:
- `ollama://` prefix marks value as remote
- authority → `http://host[:port]`
- path after first `/` → model name verbatim (may contain `/` and `:`)
- Empty authority or empty model name → parse failure → treated as local value (will fail local lookup)

---

## Files changed

| File | Change |
|------|--------|
| `Cargo.toml` | Add `ureq = { version = "3", default-features = false, features = ["rustls", "json"] }` |
| `src/llm/remote.rs` | New — `OllamaRemote` struct + `parse_ollama_env_value` + unit tests |
| `src/llm/mod.rs` | Add `pub mod remote` |
| `src/llm/embedding.rs` | `EmbedderBackend` enum (Local/Ollama), `OllamaEmbedder`, dimension probe, response parsing |
| `src/llm/combined.rs` | `CombinedBackend` enum (Local/Ollama), `OllamaCombined`, logprob scoring, rerank prompt |
| `src/llm/download.rs` | `validate_one` helper — skips local validation for `ollama://` values |

---

## Architecture

### `src/llm/remote.rs` (new)

```rust
pub struct OllamaRemote { pub base_url: String, pub model_name: String }
pub fn parse_ollama_env_value(raw: &str) -> Option<OllamaRemote>
```

Single responsibility: parse and hold remote coordinates.

### `src/llm/embedding.rs`

```
Embedder.backend: EmbedderBackend
  ├── Local(LocalEmbedder)  — existing llama_cpp_2 path, unchanged
  └── Ollama(OllamaEmbedder { remote: OllamaRemote, dimensions: usize })
```

`load_default()` checks env keys for `ollama://` **before** the existing local/HF lookup. Env key
iteration unified: reads **first matched key** (same pattern as combined), not all keys.

`OllamaEmbedder`:
- `probe_dimensions` — sends `"dimension probe"` at startup to measure vector length
- `embed_request` — `POST /api/embed`, parses `embeddings[0]`, L2-normalizes
- `profile_for_remote_model_name` — infers `EmbeddingProfile` from model name substring

### `src/llm/combined.rs`

```
Combined.backend: CombinedBackend
  ├── Local(LocalCombined)  — existing llama_cpp_2 path, unchanged
  └── Ollama(OllamaCombined { remote: OllamaRemote })
```

`load_default()` reads `IR_COMBINED_MODEL` / `IR_QWEN_MODEL`; checks `ollama://` before local/HF.

`OllamaCombined`:
- `expand` — `POST /api/generate` with `stream: false, think: false`; parses `response` text via existing `parse_output` + `fallback`
- `score_batch` — one `POST /api/generate` per doc with `logprobs: true, top_logprobs: 2, num_predict: 1`; `score_from_ollama_logprobs` softmax(P(Yes), P(No))
- `build_rerank_prompt` — DSPy-aligned Qwen3 prompt (mirrors local path), truncates doc at `MAX_RERANK_DOC_CHARS = 6000`

### `src/llm/download.rs`

`prepare_model_envs` gains a `validate_one` inner function that checks each env key; if the value
parses as `ollama://`, returns `Ok(())` immediately without touching the filesystem or HF.

---

## Data flow

```
IR_EMBEDDING_MODEL=ollama://...
        │
        ▼
Embedder::load_default()
  parse_ollama_env_value → Some(remote)
        │
        ▼
OllamaEmbedder::probe_dimensions   (POST /api/embed "dimension probe")
        │
  dimensions stored
        │
  on each embed call:
        ▼
OllamaEmbedder::embed_request      (POST /api/embed input)
  parse_embed_response → l2_normalize → Vec<f32>
```

```
IR_COMBINED_MODEL=ollama://...
        │
        ▼
Combined::load_default()
  parse_ollama_env_value → Some(remote)
        │
  expansion:  OllamaCombined::expand    (POST /api/generate)
  reranking:  OllamaCombined::score_batch (POST /api/generate × N docs)
```

---

## Error handling

All HTTP and JSON errors map to `Error::Other(String)` — consistent with existing llama path.
Startup probe failure (bad URL, model not found) surfaces immediately as a load error.

---

## Testing

New unit tests use `ENV_LOCK` mutex (not `unsafe set_var`) — per CLAUDE.md requirement.

| Test | Location |
|------|----------|
| `parse_ollama_env_value_*` (4 cases) | `remote.rs` |
| `remote_profile_for_model_name_*` | `embedding.rs` |
| `parse_embed_response_normalizes_vector` | `embedding.rs` |
| `score_yes_probability_uses_top_logprobs` | `combined.rs` |
| `remote_combined_name_comes_from_model_name` | `combined.rs` |
| `prepare_model_envs_accepts_remote_embedding_value` | `download.rs` |
| `resolve_env_hf_or_path_stays_local_only_for_remote_like_value` | `download.rs` |

All new tests are pure-unit (no network, no model files). LLM tests that require a live Ollama
instance are `#[ignore]` (same convention as existing LLM tests).

---

## Known gaps (PoC — not addressed)

1. No streaming — `generate` blocks until full response
2. Sequential reranking — one HTTP call per doc
3. No timeout / connection pool — slow Ollama hangs `ir search`
4. No logprobs fallback — hard error if `top_logprobs` absent in response
5. No benchmark coverage — nDCG parity unverified
