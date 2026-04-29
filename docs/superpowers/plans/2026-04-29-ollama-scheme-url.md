# Ollama Scheme URL Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the remote model env syntax with `ollama://host:port/model[:tag]` while keeping the existing Ollama-backed embedder/combined runtime behavior unchanged.

**Architecture:** Keep the change parser-first. `src/llm/remote.rs` becomes the only place that understands Ollama remote syntax, `prepare_model_envs()` continues to treat recognized remote values as valid without local/HF resolution, and the existing embedder/combined loaders keep consuming the same `{ base_url, model_name }` output. No transport redesign, no compatibility shim, no change to the local/HF resolution flow.

**Tech Stack:** Rust 2024, serde/serde_json, ureq, cargo test

---

## File map

- **Modify:** `src/llm/remote.rs`
  - Replace the old `http://model@host:port` parser with `ollama://host:port/model`.
  - Update the parser unit tests to cover slash-containing model names and malformed `ollama://` values.
- **Modify:** `src/llm/download.rs`
  - Keep the current remote preflight branch, but update tests so accepted remote values use the new `ollama://` syntax.
- **Modify:** `README.md`
  - Replace remote examples and syntax description with `ollama://host:port/model[:tag]`.
- **Modify:** `README.ko.md`
  - Mirror the README syntax update in Korean.
- **Modify:** `CHANGELOG.md`
  - Update the Unreleased feature note to describe the new remote syntax.

## Test map

- `src/llm/remote.rs`
  - success parse for `ollama://localhost:11111/batiai/qwen3.6-27b:iq4`
  - reject missing model path
  - reject missing authority
  - reject non-`ollama` schemes
- `src/llm/download.rs`
  - preflight still accepts remote embedding values using `ollama://...`
  - local-only resolver still does not claim remote-like values

### Task 1: Swap the parser and preflight tests to `ollama://`

**Files:**
- Modify: `src/llm/remote.rs`
- Modify: `src/llm/download.rs`
- Test: `src/llm/remote.rs`
- Test: `src/llm/download.rs`

- [ ] **Step 1: Replace the parser tests in `src/llm/remote.rs` with failing `ollama://` cases**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ollama_env_value_extracts_model_and_base_url() {
        let parsed = parse_ollama_env_value("ollama://localhost:11111/batiai/qwen3.6-27b:iq4")
            .expect("remote value should parse");
        assert_eq!(parsed.base_url, "http://localhost:11111");
        assert_eq!(parsed.model_name, "batiai/qwen3.6-27b:iq4");
    }

    #[test]
    fn parse_ollama_env_value_rejects_missing_model_path() {
        assert!(parse_ollama_env_value("ollama://localhost:11111").is_none());
    }

    #[test]
    fn parse_ollama_env_value_rejects_missing_host() {
        assert!(parse_ollama_env_value("ollama:///embeddinggemma:300m").is_none());
    }

    #[test]
    fn parse_ollama_env_value_rejects_non_ollama_scheme() {
        assert!(parse_ollama_env_value("http://localhost:11111/embeddinggemma:300m").is_none());
    }
}
```

- [ ] **Step 2: Run the parser test to verify it fails**

Run: `cargo test parse_ollama_env_value_extracts_model_and_base_url -- --nocapture`  
Expected: FAIL because the current parser only accepts `http://model@host:port`

- [ ] **Step 3: Implement the new parser shape in `src/llm/remote.rs`**

```rust
pub fn parse_ollama_env_value(raw: &str) -> Option<OllamaRemote> {
    let rest = raw.strip_prefix("ollama://")?;
    let (authority, path) = rest.split_once('/')?;
    if authority.is_empty() || path.is_empty() {
        return None;
    }

    Some(OllamaRemote {
        base_url: format!("http://{authority}"),
        model_name: path.to_string(),
    })
}
```

- [ ] **Step 4: Update the preflight tests in `src/llm/download.rs` to use `ollama://` values**

```rust
#[test]
fn prepare_model_envs_accepts_remote_embedding_value() {
    unsafe {
        std::env::set_var(
            "IR_EMBEDDING_MODEL",
            "ollama://127.0.0.1:11112/embeddinggemma:300m",
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
            "ollama://127.0.0.1:11112/embeddinggemma:300m",
        );
    }

    let result = resolve_env_hf_or_path(&["IR_TEST_REMOTE_LOCAL_ONLY"], &[models::EMBEDDING]);

    unsafe {
        std::env::remove_var("IR_TEST_REMOTE_LOCAL_ONLY");
    }

    assert!(result.is_err(), "local-only resolver must not claim remote values");
}
```

- [ ] **Step 5: Run the focused parser/preflight tests to verify they pass**

Run: `cargo test parse_ollama_env_value -- --nocapture && cargo test prepare_model_envs_accepts_remote_embedding_value -- --nocapture && cargo test resolve_env_hf_or_path_stays_local_only_for_remote_like_value -- --nocapture`  
Expected: PASS for all three tests

- [ ] **Step 6: Commit**

```bash
git add src/llm/remote.rs src/llm/download.rs
git commit -m "feat: switch Ollama env syntax to scheme URLs"
```

### Task 2: Update docs and verify the narrowed behavior

**Files:**
- Modify: `README.md`
- Modify: `README.ko.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update `README.md` remote examples and syntax text**

````md
**Remote Ollama models:**

```bash
export IR_EMBEDDING_MODEL="ollama://127.0.0.1:11112/embeddinggemma:300m"
export IR_COMBINED_MODEL="ollama://127.0.0.1:11111/batiai/qwen3.6-27b:iq4"
```

`IR_*_MODEL` env vars accept a path to a `.gguf` file, a directory containing a known model file, a HuggingFace repo ID (`owner/name`), or an Ollama remote value in the form `ollama://host:port/model[:tag]`. In the remote form, the path component is interpreted as the model name. Remote values bypass HuggingFace download and in-process llama.cpp loading for that role.
````

- [ ] **Step 2: Update `README.ko.md` with the same syntax change**

````md
**원격 Ollama 모델:**

```bash
export IR_EMBEDDING_MODEL="ollama://127.0.0.1:11112/embeddinggemma:300m"
export IR_COMBINED_MODEL="ollama://127.0.0.1:11111/batiai/qwen3.6-27b:iq4"
```

`IR_*_MODEL` 환경변수는 `.gguf` 파일 경로, 모델이 포함된 디렉터리 경로, HuggingFace 레포 ID(`owner/name`), 또는 `ollama://host:port/model[:tag]` 형식의 Ollama 원격 값을 허용합니다. 원격 형식에서는 path 컴포넌트가 모델 이름으로 해석됩니다. 원격 값은 해당 역할에서 HuggingFace 다운로드와 in-process llama.cpp 로딩을 우회합니다.
````

- [ ] **Step 3: Update the Unreleased changelog entry**

```md
### Features

- `IR_EMBEDDING_MODEL` and `IR_COMBINED_MODEL` now accept Ollama remote values in the form
  `ollama://host:port/model[:tag]`. These values skip HuggingFace download and use Ollama HTTP
  endpoints directly for embedding or combined expand+rerank work.
```

- [ ] **Step 4: Run the full test suite and inspect the docs diff**

Run: `cargo test && git --no-pager diff -- README.md README.ko.md CHANGELOG.md src/llm/remote.rs src/llm/download.rs`  
Expected: `cargo test` exits 0, and the diff shows only the syntax swap plus test updates

- [ ] **Step 5: Commit**

```bash
git add README.md README.ko.md CHANGELOG.md
git commit -m "docs: update Ollama remote syntax examples"
```

## Self-review

- **Spec coverage:** Task 1 covers the new `ollama://host:port/model` parser, slash-containing model names, and preflight acceptance. Task 2 covers the required README/README.ko/CHANGELOG updates and full verification. No spec requirement is left without a task.
- **Placeholder scan:** No `TODO`, `TBD`, or hand-wavy “appropriate handling” steps remain. Each code-editing step includes the exact code shape to add or replace, and each validation step includes an exact command.
- **Type consistency:** The plan uses the existing `OllamaRemote { base_url, model_name }` output everywhere. The parser helper name, preflight helper name, and env var names match the current codebase.
