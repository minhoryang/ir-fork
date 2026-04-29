# Ollama Remote Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add URL-like Ollama-backed remote model support for `IR_EMBEDDING_MODEL` and `IR_COMBINED_MODEL` while keeping `Embedder` and `Combined` as public concrete types and preserving the existing local file / directory / HuggingFace flow.

**Architecture:** Parse the remote URL-like form in a small shared `src/llm/remote.rs` helper, let `prepare_model_envs()` treat remote values as already valid, and have `Embedder::load_default()` / `Combined::try_load_default()` branch early to `load_with_ollama_url(...)`. Keep `resolve_env_hf_or_path()` local-only, and hide the new remote behavior behind internal backend enums so call sites in search, indexing, and daemon code do not change.

**Tech Stack:** Rust 2024, llama-cpp-2, serde/serde_json, ureq (blocking HTTP client), cargo test

---

## File map

- **Create:** `src/llm/remote.rs`
  - Own the URL-like parser, remote endpoint struct, shared request/response structs, and small pure helpers for remote request/response validation.
- **Modify:** `src/llm/mod.rs`
  - Export `remote` and keep module wiring minimal.
- **Modify:** `src/llm/download.rs`
  - Keep `resolve_env_hf_or_path()` local-only.
  - Update `prepare_model_envs()` to accept remote values for the supported env vars without entering local/HF validation.
- **Modify:** `src/llm/embedding.rs`
  - Keep `Embedder` public.
  - Add an internal backend enum plus `load_with_ollama_url(...)`.
  - Route `embedding_dim`, `embed_query`, `embed_query_batch`, and `embed_doc_batch` through the backend enum.
- **Modify:** `src/llm/combined.rs`
  - Keep `Combined` public.
  - Add an internal backend enum plus `load_with_ollama_url(...)`.
  - Route `name()`, query expansion, and reranking through the backend enum.
- **Modify:** `Cargo.toml`
  - Add a small blocking HTTP dependency for Ollama requests.
- **Modify:** `README.md`
  - Document the new env format and examples.
- **Modify:** `README.ko.md`
  - Mirror the README change in Korean.
- **Modify:** `CHANGELOG.md`
  - Add the user-facing env format and remote-model behavior under Unreleased.

## Test map

- `src/llm/remote.rs`
  - parser tests for `http://model:tag@host:port`
  - parser rejection tests for malformed values
- `src/llm/download.rs`
  - preflight tests proving remote values are accepted without local/HF resolution
- `src/llm/embedding.rs`
  - backend selection tests
  - request/response parsing tests for `/api/embed`
- `src/llm/combined.rs`
  - backend selection tests
  - yes/no score extraction tests for Ollama logprobs
  - expansion request/response tests for `/api/generate`

### Task 1: Add shared remote parser and preflight acceptance

**Files:**
- Create: `src/llm/remote.rs`
- Modify: `src/llm/mod.rs`
- Modify: `src/llm/download.rs`
- Test: `src/llm/remote.rs`
- Test: `src/llm/download.rs`

- [ ] **Step 1: Write the failing parser tests in `src/llm/remote.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ollama_env_value_extracts_model_and_base_url() {
        let parsed = parse_ollama_env_value("http://embeddinggemma:300m@127.0.0.1:11112")
            .expect("remote value should parse");
        assert_eq!(parsed.base_url, "http://127.0.0.1:11112");
        assert_eq!(parsed.model_name, "embeddinggemma:300m");
    }

    #[test]
    fn parse_ollama_env_value_rejects_missing_userinfo() {
        assert!(parse_ollama_env_value("http://127.0.0.1:11112").is_none());
    }

    #[test]
    fn parse_ollama_env_value_rejects_non_http_scheme() {
        assert!(parse_ollama_env_value("file://embeddinggemma@127.0.0.1:11112").is_none());
    }
}
```

- [ ] **Step 2: Run the parser test to verify it fails**

Run: `cargo test parse_ollama_env_value_extracts_model_and_base_url -- --nocapture`  
Expected: FAIL with `cannot find function 'parse_ollama_env_value'` or `cannot find type 'OllamaRemote'`

- [ ] **Step 3: Implement the shared parser in `src/llm/remote.rs` and export it from `src/llm/mod.rs`**

```rust
// src/llm/remote.rs
use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OllamaRemote {
    pub base_url: String,
    pub model_name: String,
}

pub fn parse_ollama_env_value(raw: &str) -> Option<OllamaRemote> {
    let (scheme, rest) = raw.split_once("://")?;
    if !matches!(scheme, "http" | "https") {
        return None;
    }
    let (userinfo, authority) = rest.rsplit_once('@')?;
    if userinfo.is_empty() || authority.is_empty() {
        return None;
    }
    Some(OllamaRemote {
        base_url: format!("{scheme}://{authority}"),
        model_name: userinfo.to_string(),
    })
}

pub fn require_ollama_env_value(key: &str, raw: &str) -> Result<OllamaRemote> {
    parse_ollama_env_value(raw).ok_or_else(|| {
        Error::Other(format!(
            "{key}={raw:?} is not a valid Ollama remote value.\n  \
             Expected form: http://model-name@host:port"
        ))
    })
}
```

```rust
// src/llm/mod.rs
pub mod remote;
```

- [ ] **Step 4: Add preflight tests in `src/llm/download.rs` for remote acceptance**

```rust
#[test]
fn prepare_model_envs_accepts_remote_embedding_value() {
    unsafe {
        std::env::set_var(
            "IR_EMBEDDING_MODEL",
            "http://embeddinggemma:300m@127.0.0.1:11112",
        );
    }

    let result = prepare_model_envs();

    unsafe {
        std::env::remove_var("IR_EMBEDDING_MODEL");
    }

    assert!(result.is_ok(), "remote value should bypass local/HF validation");
}

#[test]
fn resolve_env_hf_or_path_stays_local_only_for_remote_like_value() {
    unsafe {
        std::env::set_var(
            "IR_TEST_REMOTE_LOCAL_ONLY",
            "http://embeddinggemma:300m@127.0.0.1:11112",
        );
    }

    let result = resolve_env_hf_or_path(&["IR_TEST_REMOTE_LOCAL_ONLY"], &[models::EMBEDDING]);

    unsafe {
        std::env::remove_var("IR_TEST_REMOTE_LOCAL_ONLY");
    }

    assert!(result.is_err(), "local-only resolver must not claim remote values");
}
```

- [ ] **Step 5: Run the download preflight test to verify it fails**

Run: `cargo test prepare_model_envs_accepts_remote_embedding_value -- --nocapture`  
Expected: FAIL because `prepare_model_envs()` still sends the value into local/HF validation

- [ ] **Step 6: Update `prepare_model_envs()` to skip local/HF validation when the remote helper matches**

```rust
pub fn prepare_model_envs() -> Result<()> {
    use crate::llm::env;

    fn validate_one(env_vars: &[&str], dir_candidates: &[&str]) -> Result<()> {
        for key in env_vars {
            let Some(raw) = std::env::var_os(key) else {
                continue;
            };
            let raw = raw.to_string_lossy().into_owned();
            if crate::llm::remote::parse_ollama_env_value(&raw).is_some() {
                return Ok(());
            }
            let _ = resolve_env_hf_or_path(&[*key], dir_candidates)?;
            return Ok(());
        }
        Ok(())
    }

    validate_one(env::EMBEDDING_MODEL, &[models::EMBEDDING, models::BGE_M3])?;
    validate_one(env::RERANKER_MODEL, &[models::RERANKER])?;
    validate_one(env::EXPANDER_MODEL, &[models::EXPANDER])?;
    validate_one(&[env::COMBINED_MODEL, env::QWEN_MODEL], &[models::QWEN35_2B, models::QWEN35_0_8B])?;
    Ok(())
}
```

- [ ] **Step 7: Run the focused tests to verify they pass**

Run: `cargo test parse_ollama_env_value -- --nocapture && cargo test prepare_model_envs_accepts_remote_embedding_value -- --nocapture`  
Expected: PASS for parser and preflight acceptance tests

- [ ] **Step 8: Commit**

```bash
git add src/llm/remote.rs src/llm/mod.rs src/llm/download.rs
git commit -m "feat: accept Ollama remote model env values"
```

### Task 2: Add remote embedder backend behind `Embedder`

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/llm/embedding.rs`
- Modify: `src/llm/remote.rs`
- Test: `src/llm/embedding.rs`

- [ ] **Step 1: Add the failing embedding backend-selection and response tests**

```rust
#[test]
fn remote_profile_for_model_name_uses_embeddinggemma_rules() {
    let profile = profile_for_remote_model_name("embeddinggemma:300m");
    assert_eq!(profile, EmbeddingProfile::EmbeddingGemma);
}

#[test]
fn parse_embed_response_normalizes_vector() {
    let json = serde_json::json!({
        "model": "embeddinggemma:300m",
        "embeddings": [[3.0, 4.0]]
    });

    let mut emb = parse_embed_response(&json).expect("valid embed response");
    crate::llm::l2_normalize(&mut emb);
    let mag: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((mag - 1.0).abs() < 1e-6);
}
```

- [ ] **Step 2: Run the embedding test to verify it fails**

Run: `cargo test remote_profile_for_model_name_uses_embeddinggemma_rules -- --nocapture`  
Expected: FAIL with missing helper functions / backend enum

- [ ] **Step 3: Add a blocking HTTP client dependency**

```toml
[dependencies]
ureq = { version = "3", features = ["json"] }
```

- [ ] **Step 4: Refactor `Embedder` to keep the public type while hiding local/remote behavior behind an internal enum**

```rust
enum EmbedderBackend {
    Local {
        backend: &'static LlamaBackend,
        model: LlamaModel,
    },
    Ollama(OllamaEmbedder),
}

struct OllamaEmbedder {
    remote: crate::llm::remote::OllamaRemote,
    profile: EmbeddingProfile,
    dimensions: usize,
}

pub struct Embedder {
    backend: EmbedderBackend,
    profile: EmbeddingProfile,
    pooling_override: Option<EmbeddingPooling>,
}
```

- [ ] **Step 5: Add the remote constructor and early branch in `load_default()`**

```rust
pub fn load_default() -> Result<Self> {
    for key in crate::llm::env::EMBEDDING_MODEL {
        if let Some(raw) = std::env::var_os(key) {
            let raw = raw.to_string_lossy().into_owned();
            if let Some(remote) = crate::llm::remote::parse_ollama_env_value(&raw) {
                return Self::load_with_ollama_url(remote);
            }
        }
    }

    let path = match crate::llm::download::resolve_env_hf_or_path(
        crate::llm::env::EMBEDDING_MODEL,
        &[models::EMBEDDING, models::BGE_M3],
    )? {
        Some(p) => p,
        None => crate::llm::download::ensure_model(models::EMBEDDING)?,
    };

    Self::load_with_gpu_layers(&path, crate::llm::gpu_layers())
}
```

- [ ] **Step 6: Add the remote embed request path and backend dispatch**

```rust
fn load_with_ollama_url(remote: crate::llm::remote::OllamaRemote) -> Result<Self> {
    let profile = profile_for_remote_model_name(&remote.model_name);
    Ok(Self {
        backend: EmbedderBackend::Ollama(OllamaEmbedder {
            remote,
            profile,
            dimensions: 0,
        }),
        profile,
        pooling_override: None,
    })
}

pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
    match &self.backend {
        EmbedderBackend::Local { .. } => self.embed_single(&self.format_query(query)),
        EmbedderBackend::Ollama(remote) => remote.embed(&self.format_query(query)),
    }
}
```

```rust
impl OllamaEmbedder {
    fn embed(&self, input: &str) -> Result<Vec<f32>> {
        let body = serde_json::json!({
            "model": self.remote.model_name,
            "input": input,
        });
        let response: serde_json::Value = ureq::post(&format!("{}/api/embed", self.remote.base_url))
            .send_json(body)
            .map_err(|e| Error::Other(format!("ollama embed request: {e}")))?
            .into_json()
            .map_err(|e| Error::Other(format!("ollama embed decode: {e}")))?;
        let mut emb = parse_embed_response(&response)?;
        crate::llm::l2_normalize(&mut emb);
        Ok(emb)
    }
}
```

- [ ] **Step 7: Run the focused embedding tests to verify they pass**

Run: `cargo test remote_profile_for_model_name_uses_embeddinggemma_rules -- --nocapture && cargo test parse_embed_response_normalizes_vector -- --nocapture`  
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml src/llm/remote.rs src/llm/embedding.rs
git commit -m "feat: add Ollama-backed embedding backend"
```

### Task 3: Add remote combined backend behind `Combined`

**Files:**
- Modify: `src/llm/combined.rs`
- Modify: `src/llm/remote.rs`
- Test: `src/llm/combined.rs`

- [ ] **Step 1: Add the failing combined helper tests**

```rust
#[test]
fn score_yes_probability_uses_top_logprobs() {
    let json = serde_json::json!([{
        "token": "Yes",
        "top_logprobs": [
            {"token": "Yes", "logprob": -0.1},
            {"token": "No", "logprob": -2.0}
        ]
    }]);

    let score = score_from_ollama_logprobs(&json).expect("score should parse");
    assert!(score > 0.8, "expected a strong yes score, got {score}");
}

#[test]
fn remote_combined_name_comes_from_model_name() {
    let combined = Combined::load_with_ollama_url(crate::llm::remote::OllamaRemote {
        base_url: "http://127.0.0.1:11111".into(),
        model_name: "qwen3.5:2b".into(),
    }).expect("remote combined");

    assert_eq!(combined.name(), "qwen3.5:2b");
}
```

- [ ] **Step 2: Run the combined test to verify it fails**

Run: `cargo test score_yes_probability_uses_top_logprobs -- --nocapture`  
Expected: FAIL with missing remote combined helpers

- [ ] **Step 3: Refactor `Combined` to keep the public type while hiding local/remote behavior behind an internal enum**

```rust
enum CombinedBackend {
    Local {
        backend: &'static LlamaBackend,
        model: LlamaModel,
        yes_token_id: i32,
        no_token_id: i32,
        cached_rerank_ctx: Mutex<Option<LlamaContext<'static>>>,
    },
    Ollama(OllamaCombined),
}

struct OllamaCombined {
    remote: crate::llm::remote::OllamaRemote,
}

pub struct Combined {
    backend: CombinedBackend,
    model_name: String,
}
```

- [ ] **Step 4: Add the early remote branch in `try_load_default()` and the remote constructor**

```rust
pub fn try_load_default() -> Result<Option<Self>> {
    use crate::llm::env;

    let env_key: &[&str] = if std::env::var_os(env::COMBINED_MODEL).is_some() {
        &[env::COMBINED_MODEL]
    } else if std::env::var_os(env::QWEN_MODEL).is_some() {
        eprintln!("ir: IR_QWEN_MODEL is deprecated — use IR_COMBINED_MODEL instead");
        &[env::QWEN_MODEL]
    } else {
        &[]
    };

    if env_key.is_empty() {
        return Ok(None);
    }

    let raw = std::env::var(env_key[0]).map_err(|e| Error::Other(format!("read combined env: {e}")))?;
    if let Some(remote) = crate::llm::remote::parse_ollama_env_value(&raw) {
        return Ok(Some(Self::load_with_ollama_url(remote)?));
    }

    match crate::llm::download::resolve_env_hf_or_path(
        env_key,
        &[models::QWEN35_2B, models::QWEN35_0_8B],
    )? {
        Some(p) => Ok(Some(Self::load(&p)?)),
        None => Ok(None),
    }
}
```

- [ ] **Step 5: Implement remote expansion and reranking over `/api/generate`**

```rust
impl Combined {
    fn load_with_ollama_url(remote: crate::llm::remote::OllamaRemote) -> Result<Self> {
        Ok(Self {
            model_name: remote.model_name.clone(),
            backend: CombinedBackend::Ollama(OllamaCombined { remote }),
        })
    }

    pub fn expand(&self, query: &str) -> Result<Vec<SubQuery>> {
        match &self.backend {
            CombinedBackend::Local { .. } => self.expand_local(query),
            CombinedBackend::Ollama(remote) => remote.expand(query),
        }
    }
}

impl OllamaCombined {
    fn expand(&self, query: &str) -> Result<Vec<SubQuery>> {
        let body = serde_json::json!({
            "model": self.remote.model_name,
            "prompt": build_expand_prompt(query),
            "stream": false,
        });
        let json: serde_json::Value = ureq::post(&format!("{}/api/generate", self.remote.base_url))
            .send_json(body)
            .map_err(|e| Error::Other(format!("ollama expand request: {e}")))?
            .into_json()
            .map_err(|e| Error::Other(format!("ollama expand decode: {e}")))?;
        let raw = json["response"]
            .as_str()
            .ok_or_else(|| Error::Other("ollama expand response missing text".into()))?;
        let parsed = parse_output(raw);
        if parsed.is_empty() { Ok(fallback(query)) } else { Ok(parsed) }
    }
}
```

- [ ] **Step 6: Add `Scorer` dispatch for remote yes/no logprob scoring**

```rust
fn score_from_ollama_logprobs(value: &serde_json::Value) -> Result<f64> {
    let entries = value
        .as_array()
        .ok_or_else(|| Error::Other("ollama rerank logprobs missing array".into()))?;
    let top = &entries[0]["top_logprobs"];
    let variants = top
        .as_array()
        .ok_or_else(|| Error::Other("ollama rerank top_logprobs missing array".into()))?;

    let yes = variants.iter().find(|v| v["token"].as_str() == Some("Yes"));
    let no = variants.iter().find(|v| v["token"].as_str() == Some("No"));

    let yes = yes.and_then(|v| v["logprob"].as_f64())
        .ok_or_else(|| Error::Other("ollama rerank missing Yes logprob".into()))?;
    let no = no.and_then(|v| v["logprob"].as_f64())
        .ok_or_else(|| Error::Other("ollama rerank missing No logprob".into()))?;

    let max = yes.max(no);
    let yes_exp = (yes - max).exp();
    let no_exp = (no - max).exp();
    Ok(yes_exp / (yes_exp + no_exp))
}
```

- [ ] **Step 7: Run the focused combined tests to verify they pass**

Run: `cargo test score_yes_probability_uses_top_logprobs -- --nocapture && cargo test remote_combined_name_comes_from_model_name -- --nocapture`  
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/llm/combined.rs src/llm/remote.rs
git commit -m "feat: add Ollama-backed combined backend"
```

### Task 4: Update docs and changelog, then run project verification

**Files:**
- Modify: `README.md`
- Modify: `README.ko.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the failing doc expectations by writing the exact examples to include**

```md
export IR_EMBEDDING_MODEL="http://embeddinggemma:300m@127.0.0.1:11112"
export IR_COMBINED_MODEL="http://qwen3.5:2b@127.0.0.1:11111"
```

```md
`IR_*_MODEL` also accepts an Ollama remote value in the form `http://model-name@host:port`.
In this form, the URL userinfo is interpreted as the model name, not credentials.
Remote values bypass HuggingFace download and in-process llama.cpp loading for that role.
```

- [ ] **Step 2: Update `README.md`, `README.ko.md`, and `CHANGELOG.md`**

```md
## [Unreleased]

### Features

- `IR_EMBEDDING_MODEL` and `IR_COMBINED_MODEL` now accept Ollama remote values in the form
  `http://model-name@host:port`. These values skip HuggingFace download and use Ollama HTTP
  endpoints directly for embedding or combined expand+rerank work.
```

- [ ] **Step 3: Run the targeted and full test commands**

Run: `cargo test llm::download -- --nocapture && cargo test llm::embedding -- --nocapture && cargo test llm::combined -- --nocapture && cargo test`  
Expected: PASS for the new unit tests and the full existing unit-test suite

- [ ] **Step 4: Commit**

```bash
git add README.md README.ko.md CHANGELOG.md
git commit -m "docs: document Ollama remote model support"
```
