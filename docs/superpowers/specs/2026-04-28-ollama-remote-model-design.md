# Spec: Ollama-backed remote model resolution in `src/llm/*`

## Problem

`ir` currently assumes `IR_*_MODEL` values resolve to local GGUF files, directories, or known HuggingFace repo IDs. That forces in-process `llama_cpp_2` model loading even in environments where the user already runs models behind Ollama and wants `ir` to call HTTP APIs directly.

For this change, the goal is to make that remote path possible with a surgical implementation limited to `src/llm/*`, while preserving current behavior for existing non-remote env values.

## Goals

1. Allow a remote Ollama model to be selected through existing `IR_*_MODEL` env vars.
2. Keep implementation changes limited to `src/llm/*`.
3. Preserve current behavior for file, directory, and HuggingFace repo ID env values.
4. Make remote mode explicit and fail fast on invalid configuration or bad remote responses.
5. Ensure remote mode bypasses HuggingFace download and in-process llama loading for the selected role.

## Non-goals

1. Do not redesign the search pipeline, daemon orchestration, or CLI surface outside `src/llm/*`.
2. Do not add generic provider abstractions for non-Ollama backends.
3. Do not introduce auth semantics for URL userinfo.
4. Do not require or promise full remote validation for dedicated expander/reranker mode in this change.

## Scope

### Required and tested path

- `IR_EMBEDDING_MODEL`
- `IR_COMBINED_MODEL`

These two env vars must support remote execution end to end.

### Parser-compatible but not required/tested in this change

- `IR_RERANKER_MODEL`
- `IR_EXPANDER_MODEL`

The remote env format may be accepted for these vars by shared parsing logic, but the acceptance criteria for this change only require validated remote behavior for embedding and combined mode.

## User-facing env format

Remote mode uses a URL-like convention:

```bash
IR_EMBEDDING_MODEL=http://embeddinggemma:300m@127.0.0.1:11112
IR_COMBINED_MODEL=http://qwen3.5:2b@127.0.0.1:11111
```

Interpretation rules:

- scheme + host + optional port form the Ollama base URL
- raw userinfo is the model name
- `username[:password]` is reconstructed back into the model name verbatim
- userinfo is **not** credentials in this mode
- percent-encoding is not required

Examples:

| Env value | Base URL | Model name |
|---|---|---|
| `http://embeddinggemma:300m@127.0.0.1:11112` | `http://127.0.0.1:11112` | `embeddinggemma:300m` |
| `http://qwen3.5:2b@127.0.0.1:11111` | `http://127.0.0.1:11111` | `qwen3.5:2b` |

This is a purpose-built URL-like convention for model selection. It must be documented as such so it is not confused with auth support.

## Env resolution rules

Existing accepted forms remain unchanged:

1. local file path
2. local directory
3. known HuggingFace repo ID

The new remote form adds one more explicit branch:

4. URL-like Ollama remote value

Resolution order for a set env var becomes:

1. if it matches the Ollama remote syntax, resolve to remote mode
2. else if it is a file, use local file
3. else if it is a directory, resolve known model filename within that directory
4. else if it is a known HF repo ID, use current HF path/download flow
5. else error

The remote syntax check must happen before file/path/HF error reporting so a valid remote value is not rejected as an invalid path.

## Internal design

All implementation changes stay inside `src/llm/*`.

### New shared source representation

Introduce a small internal source representation in `src/llm/*`:

- local path source
- remote Ollama source `{ base_url, model_name }`

This source type is internal to the llm layer. Callers outside `src/llm/*` keep using the existing loader entry points.

### File boundaries

- `src/llm/download.rs`
  - parse and validate the remote env format
  - return an internal source variant instead of assuming every explicit env value resolves to a local path
  - keep existing local path and HF behavior unchanged

- `src/llm/embedding.rs`
  - keep current formatting rules for query/doc text
  - add a remote embedder path using Ollama HTTP
  - keep public loader shape unchanged

- `src/llm/combined.rs`
  - keep current public combined loader shape unchanged
  - add a remote combined path using Ollama HTTP
  - preserve the current expansion parser contract and yes/no reranker contract

- a shared helper module under `src/llm/*` is allowed for:
  - HTTP request/response types
  - remote parsing helpers
  - common remote error formatting

No search, daemon, or CLI call site is required to know whether a loaded model is local or remote.

## Remote runtime behavior

### Embedding

If `IR_EMBEDDING_MODEL` resolves to remote mode:

- do not initialize `LlamaBackend`
- do not resolve a local GGUF path
- do not call HuggingFace download logic
- call `POST {base_url}/api/embed`
- send the resolved model name in the JSON body

The existing embedding-side text shaping should remain unchanged in intent and should be selected by model-name profile inference:

- model names containing `embeddinggemma` use the current EmbeddingGemma query/doc formatting
- model names containing `bge-m3` use the current BGE-M3 query/doc formatting
- all other remote embedding model names use the current Generic profile formatting

This keeps remote embedding behavior aligned with the existing formatting profiles without depending on local GGUF metadata.

### Combined mode

If `IR_COMBINED_MODEL` resolves to remote mode:

- do not initialize `LlamaBackend`
- do not resolve a local GGUF path
- do not call HuggingFace download logic
- do not activate dedicated expander/reranker loading for tier-2 execution

`IR_COMBINED_MODEL` remains the single source of truth for combined mode. When it is set, `IR_RERANKER_MODEL` and `IR_EXPANDER_MODEL` logic is bypassed for actual tier-2 activation, consistent with current combined-mode precedence.

#### Expansion path

Expansion continues to use the current typed output contract:

```text
lex: ...
vec: ...
hyde: ...
```

This means the remote combined path changes transport only:

- keep the existing expansion prompt intent
- call `POST {base_url}/api/generate`
- parse `lex:/vec:/hyde:` lines the same way as current logic
- preserve current fallback behavior when output is empty, malformed, or insufficiently grounded in the original query

#### Reranking path

Reranking keeps the current yes/no judgment concept, but obtains its numeric signal from Ollama response data instead of in-process logits:

- call `POST {base_url}/api/generate`
- request deterministic output settings
- request `logprobs`
- constrain the output to a yes/no answer contract
- derive a numeric score from the returned token log probabilities for the generated answer

The score extraction rule is:

1. request exactly one answer token with deterministic settings (`stream: false`, `temperature: 0`, `max_tokens: 1`, `logprobs: true`, `top_logprobs >= 2`)
2. force the prompt to return only `Yes` or `No`
3. read the first generated token's `top_logprobs`
4. find the `Yes` and `No` alternatives in that list
5. compute the score as the normalized probability of `Yes` over `Yes` + `No`
6. error if the first token does not expose both `Yes` and `No` alternatives clearly enough to compute that value

## HTTP contract

### Embedding endpoint

Remote embedding uses:

```text
POST {base_url}/api/embed
```

Expected request body:

- `model`
- `input`

Expected success response:

- embedding array payload from Ollama

### Generation endpoint

Remote combined expansion and reranking use:

```text
POST {base_url}/api/generate
```

Expected request body includes:

- `model`
- `prompt`
- `stream: false`
- `logprobs: true` for reranking
- `temperature: 0` and `max_tokens: 1` for reranking

The implementation should construct endpoint paths internally from the base URL. The env var itself stores the base URL only, not `/api/embed` or `/api/generate`.

## Error handling

Remote mode is explicit, so failures must also be explicit.

The llm layer must fail fast with clear `Error::Other(...)` messages for:

- malformed remote env value
- missing scheme, host, or model name
- unsupported URL shape for remote mode
- network failure
- non-success HTTP status
- invalid JSON response
- missing embeddings payload
- missing or unusable `logprobs` for reranking
- generation output that cannot be parsed into the required contract

Error messages should name:

- the env var being resolved
- the operation being attempted
- the base URL
- the model name where useful

Remote mode must not silently:

- fall back to local GGUF loading
- fall back to HuggingFace download
- skip a configured tier and continue as though configuration succeeded

## `HF_HUB_OFFLINE=1` behavior

When an env var resolves to remote mode, the HuggingFace path is not entered at all. For those branches:

- no HF lookup
- no download
- no interaction with `HF_HUB_OFFLINE`

This must be true even when `HF_HUB_OFFLINE=1` is set. Remote mode is an alternate execution path, not a variation of model download behavior.

## Compatibility and preservation requirements

This change must preserve existing behavior for non-remote env values:

- existing path values still resolve as before
- existing directory values still resolve as before
- existing known HF repo ID values still resolve as before
- unset env vars still use current defaults

The design is additive. Existing users who do not opt into remote syntax should observe no behavior change.

## Testing requirements

Testing should stay inside the existing Rust test suite and focus on `src/llm/*`.

### Required tests

1. env parsing tests for the URL-like remote syntax
2. tests that reconstruct model names containing `:`
3. tests that reject malformed remote values with clear errors
4. tests that confirm non-remote path/dir/HF values still resolve as before
5. tests for Ollama embedding response parsing
6. tests for Ollama rerank response parsing and yes/no score extraction
7. tests that remote resolution bypasses HF download/local-path fallback behavior
8. tests that combined mode precedence still bypasses dedicated tier-2 activation when `IR_COMBINED_MODEL` is set

### Acceptance expectations

For this change, the acceptance path is:

- remote embedding via `IR_EMBEDDING_MODEL`
- remote combined expansion + reranking via `IR_COMBINED_MODEL`

Dedicated remote expander/reranker mode may share the parser but is not part of the must-validate surface for this spec.

## Documentation requirements

Because this is a user-facing feature:

- update `README.md`
- update `README.ko.md`
- update `CHANGELOG.md` Unreleased section

Documentation must cover:

- the new URL-like env format
- examples for embedding and combined mode
- the fact that userinfo is interpreted as model name, not auth
- the fact that remote mode bypasses in-process llama loading for the selected role
- the fact that remote mode constructs `/api/embed` and `/api/generate` from the base URL internally

## Acceptance summary

This spec is complete when:

1. existing `IR_*_MODEL` values still behave as before
2. `IR_EMBEDDING_MODEL` can point to an Ollama-served embedding model using the URL-like syntax
3. `IR_COMBINED_MODEL` can point to an Ollama-served combined model using the URL-like syntax
4. remote combined mode bypasses dedicated tier-2 activation logic
5. remote mode never downloads models or initializes llama.cpp for the selected role
6. failures are explicit and user-readable
