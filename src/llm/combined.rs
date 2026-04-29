// Combined model: one GGUF load serving both query expansion and reranking.
// Activated via IR_COMBINED_MODEL (preferred) or IR_QWEN_MODEL (deprecated).
// Currently tuned for instruction-following models with ChatML format (Qwen3.5).
// docs: https://docs.rs/llama-cpp-2/latest/llama_cpp_2/
//
// Reranker role:  Yes/No logit scoring (same protocol as reranker.rs)
// Expander role:  autoregressive generation → lex:/vec:/hyde: lines
//
// Prompts are DSPy-MIPROv2 optimized (see research/dspy_optimize.py).
// Run research/dspy_optimize.py to re-optimize; hardcode results here.

use crate::error::{Error, Result};
use crate::llm::expander::{QueryExpander, SubQuery, fallback, parse_output};
use crate::llm::remote::OllamaRemote;
use crate::llm::generate::{self, GenerateParams};
use crate::llm::scoring::{self, Scorer};
use crate::llm::{LlamaBackend, model_load_params, models};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::{AddBos, LlamaModel};
use std::path::Path;
use std::sync::Mutex;

const RERANK_CONTEXT_SIZE: u32 = 2048;
const EXPAND_CONTEXT_SIZE: u32 = 2048;
const MAX_EXPAND_TOKENS: usize = 300;
const MAX_RERANK_DOC_CHARS: usize = 6000;

pub struct Combined {
    backend: CombinedBackend,
    model_name: String,
}

enum CombinedBackend {
    Local(LocalCombined),
    Ollama(OllamaCombined),
}

struct LocalCombined {
    backend: &'static LlamaBackend,
    model: LlamaModel,
    yes_token_id: i32,
    no_token_id: i32,
    cached_rerank_ctx: Mutex<Option<LlamaContext<'static>>>,
}

struct OllamaCombined {
    remote: OllamaRemote,
}

// ! Safety: LlamaModel is Send+Sync, LlamaContext access is serialized by Mutex
unsafe impl Send for Combined {}
unsafe impl Sync for Combined {}

impl Combined {
    pub fn load(path: &Path) -> Result<Self> {
        let backend = crate::llm::init_backend()?;
        let model = LlamaModel::load_from_file(backend, path, &model_load_params())
            .map_err(|e| Error::Other(format!("load combined model: {e}")))?;
        let model_filename = path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        let (yes_token_id, no_token_id) = scoring::resolve_yes_no_tokens(&model)?;
        Ok(Self {
            backend: CombinedBackend::Local(LocalCombined {
                backend,
                model,
                yes_token_id,
                no_token_id,
                cached_rerank_ctx: Mutex::new(None),
            }),
            model_name: model_filename,
        })
    }

    pub fn name(&self) -> &str {
        &self.model_name
    }

    fn load_with_ollama_url(remote: OllamaRemote) -> Result<Self> {
        let model_name = remote.model_name.clone();
        Ok(Self {
            backend: CombinedBackend::Ollama(OllamaCombined { remote }),
            model_name,
        })
    }

    /// Resolve the combined model path from explicit env vars only.
    /// Returns `Ok(None)` when neither IR_COMBINED_MODEL nor IR_QWEN_MODEL is set.
    pub fn try_load_default() -> Result<Option<Self>> {
        use crate::llm::env;

        // Priority: IR_COMBINED_MODEL > IR_QWEN_MODEL (deprecated).
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

    /// Expand a query into typed sub-queries (lex/vec/hyde). Falls back on parse failure.
    ///
    /// // ! DSPy-optimized prompt — do not edit manually; re-run research/dspy_optimize.py
    pub fn expand(&self, query: &str) -> Result<Vec<SubQuery>> {
        match &self.backend {
            CombinedBackend::Local(_) => self.expand_local(query),
            CombinedBackend::Ollama(remote) => remote.expand(query),
        }
    }

    fn expand_local(&self, query: &str) -> Result<Vec<SubQuery>> {
        let local = match &self.backend {
            CombinedBackend::Local(local) => local,
            CombinedBackend::Ollama(_) => {
                return Err(Error::Other(
                    "remote Ollama combined backend does not use llama generation".into(),
                ));
            }
        };
        let prompt = build_expand_prompt(query);
        let raw = generate::generate(
            &local.model,
            local.backend,
            &prompt,
            &GenerateParams {
                ctx_size: EXPAND_CONTEXT_SIZE,
                max_tokens: MAX_EXPAND_TOKENS,
                add_bos: AddBos::Never, // ! ChatML prompt; no extra BOS
                temp: 0.7,
                top_k: 20,
                top_p: 0.8,
                seed: 42,
            },
        )?;
        let parsed = parse_output(&raw);

        let query_lower = query.to_lowercase();
        let valid = parsed.iter().any(|s| {
            s.text
                .split_whitespace()
                .any(|w| query_lower.contains(&w.to_lowercase()))
        });

        if parsed.is_empty() || !valid {
            Ok(fallback(query))
        } else {
            Ok(parsed)
        }
    }

    fn get_or_create_rerank_ctx(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Option<LlamaContext<'static>>>> {
        let local = match &self.backend {
            CombinedBackend::Local(local) => local,
            CombinedBackend::Ollama(_) => {
                return Err(Error::Other(
                    "remote Ollama combined backend does not create llama contexts".into(),
                ));
            }
        };
        let mut guard = local.cached_rerank_ctx.lock().unwrap();
        if guard.is_none() {
            let ctx =
                scoring::create_scoring_context(&local.model, local.backend, RERANK_CONTEXT_SIZE)?;
            // ! Safety: model lives in same struct; context is dropped first via Drop impl
            let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
            *guard = Some(ctx);
        }
        Ok(guard)
    }
}

impl Drop for Combined {
    fn drop(&mut self) {
        // ! Drop context before model
        if let CombinedBackend::Local(local) = &self.backend {
            let _ = local.cached_rerank_ctx.lock().map(|mut g| g.take());
        }
    }
}

impl Scorer for Combined {
    fn model_id(&self) -> &str {
        &self.model_name
    }

    /// // ! DSPy-optimized prompt — do not edit manually; re-run research/dspy_optimize.py
    fn score_batch(&self, query: &str, docs: &[&str]) -> Result<Vec<f64>> {
        match &self.backend {
            CombinedBackend::Local(local) => {
                let mut guard = self.get_or_create_rerank_ctx()?;
                let ctx = guard.as_mut().unwrap();
                scoring::score_batch_with_ctx(
                    ctx,
                    &local.model,
                    local.yes_token_id,
                    local.no_token_id,
                    query,
                    docs,
                    RERANK_CONTEXT_SIZE,
                )
            }
            CombinedBackend::Ollama(remote) => remote.score_batch(query, docs),
        }
    }
}

impl QueryExpander for Combined {
    fn expand_query(&self, query: &str) -> Result<Vec<SubQuery>> {
        self.expand(query)
    }
    fn model_id(&self) -> &str {
        &self.model_name
    }
}

/// // ! DSPy-optimized prompt — do not edit manually; re-run research/dspy_optimize.py
fn build_expand_prompt(query: &str) -> String {
    format!(
        "<|im_start|>system\n\
         Generate search sub-queries for document retrieval. \
         Output exactly three lines: lex (2-5 keywords for BM25), \
         vec (natural language reformulation), hyde (1-2 sentence hypothetical answer passage).<|im_end|>\n\
         <|im_start|>user\n\
         Query: {query}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

fn build_rerank_prompt(query: &str, doc: &str) -> String {
    let doc_truncated = if doc.len() > MAX_RERANK_DOC_CHARS {
        &doc[..doc.floor_char_boundary(MAX_RERANK_DOC_CHARS)]
    } else {
        doc
    };
    format!(
        "<|im_start|>system\n\
         Judge whether the Document meets the requirements based on the Query and the Instruct provided. \
         Note that the answer can only be \"Yes\" or \"No\".<|im_end|>\n\
         <|im_start|>user\n\
         <Instruct>: Given a web search query, retrieve relevant passages that answer the query\n\
         <Query>: {query}\n\
         <Document>: {doc_truncated}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

fn score_from_ollama_logprobs(value: &serde_json::Value) -> Result<f64> {
    let entries = value
        .as_array()
        .ok_or_else(|| Error::Other("ollama rerank logprobs missing array".into()))?;
    let top = &entries[0]["top_logprobs"];
    let variants = top
        .as_array()
        .ok_or_else(|| Error::Other("ollama rerank top_logprobs missing array".into()))?;

    let yes = variants
        .iter()
        .find(|v| v["token"].as_str() == Some("Yes"))
        .and_then(|v| v["logprob"].as_f64())
        .ok_or_else(|| Error::Other("ollama rerank missing Yes logprob".into()))?;
    let no = variants
        .iter()
        .find(|v| v["token"].as_str() == Some("No"))
        .and_then(|v| v["logprob"].as_f64())
        .ok_or_else(|| Error::Other("ollama rerank missing No logprob".into()))?;

    let max = yes.max(no);
    let yes_exp = (yes - max).exp();
    let no_exp = (no - max).exp();
    Ok(yes_exp / (yes_exp + no_exp))
}

impl OllamaCombined {
    fn generate(&self, body: serde_json::Value) -> Result<serde_json::Value> {
        ureq::post(&format!("{}/api/generate", self.remote.base_url))
            .send_json(body)
            .map_err(|e| Error::Other(format!("ollama generate request: {e}")))?
            .into_body()
            .read_json::<serde_json::Value>()
            .map_err(|e| Error::Other(format!("ollama generate decode: {e}")))
    }

    fn expand(&self, query: &str) -> Result<Vec<SubQuery>> {
        let json = self.generate(serde_json::json!({
            "model": self.remote.model_name,
            "prompt": build_expand_prompt(query),
            "stream": false,
            "think": false,
        }))?;
        let raw = json["response"]
            .as_str()
            .ok_or_else(|| Error::Other("ollama expand response missing text".into()))?;
        let parsed = parse_output(raw);

        let query_lower = query.to_lowercase();
        let valid = parsed.iter().any(|s| {
            s.text
                .split_whitespace()
                .any(|w| query_lower.contains(&w.to_lowercase()))
        });

        if parsed.is_empty() || !valid {
            Ok(fallback(query))
        } else {
            Ok(parsed)
        }
    }

    fn score_batch(&self, query: &str, docs: &[&str]) -> Result<Vec<f64>> {
        docs.iter()
            .map(|doc| {
                let json = self.generate(serde_json::json!({
                    "model": self.remote.model_name,
                    "prompt": build_rerank_prompt(query, doc),
                    "stream": false,
                    "logprobs": true,
                    "top_logprobs": 2,
                    "options": {
                        "temperature": 0,
                        "num_predict": 1
                    }
                }))?;
                score_from_ollama_logprobs(&json["logprobs"])
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn load_0_8b_and_tokenize() {
        let path = dirs::home_dir()
            .unwrap()
            .join("local-models")
            .join(models::QWEN35_0_8B);
        assert!(path.exists(), "model not found: {}", path.display());

        let q = Combined::load(&path).expect("load 0.8B");

        let local = match &q.backend {
            CombinedBackend::Local(local) => local,
            CombinedBackend::Ollama(_) => panic!("expected local combined backend"),
        };
        let tokens = local
            .model
            .str_to_token("hello world", AddBos::Never)
            .expect("tokenize");
        assert!(!tokens.is_empty(), "tokenization returned empty");

        let result = q.expand("hello");
        assert!(result.is_ok(), "expand failed: {:?}", result.err());
        println!("0.8B output: {:?}", result.unwrap());
    }

    #[test]
    #[ignore]
    fn load_2b_and_tokenize() {
        let path = dirs::home_dir()
            .unwrap()
            .join("local-models")
            .join(models::QWEN35_2B);
        assert!(path.exists(), "model not found: {}", path.display());

        let q = Combined::load(&path).expect("load 2B");
        let local = match &q.backend {
            CombinedBackend::Local(local) => local,
            CombinedBackend::Ollama(_) => panic!("expected local combined backend"),
        };
        let tokens = local
            .model
            .str_to_token("hello world", AddBos::Never)
            .expect("tokenize");
        assert!(!tokens.is_empty());
        println!("2B vocab size: {}", local.model.n_vocab());
    }

    #[test]
    #[ignore]
    fn expand_returns_valid_subqueries() {
        use crate::llm::expander::SubQueryKind;
        let q = Combined::try_load_default().expect("load model").unwrap();
        let subs = q.expand("rust memory management").expect("expand");
        assert!(!subs.is_empty());
        let any_relevant = subs
            .iter()
            .any(|s| s.text.contains("rust") || s.text.contains("memory"));
        assert!(any_relevant, "no relevant sub-query in: {subs:?}");

        let kinds: Vec<SubQueryKind> = subs.iter().map(|s| s.kind).collect();
        assert!(kinds.contains(&SubQueryKind::Lex));
        assert!(kinds.contains(&SubQueryKind::Vec));
    }

    #[test]
    #[ignore]
    fn score_batch_orders_correctly() {
        use crate::llm::scoring::Scorer;
        let q = Combined::try_load_default().expect("load model").unwrap();
        let scores = q
            .score_batch(
                "rust memory management",
                &[
                    "Rust uses ownership and borrowing to manage memory without a garbage collector",
                    "Python uses a garbage collector. JavaScript also has automatic memory management.",
                ],
            )
            .expect("score_batch");
        assert_eq!(scores.len(), 2);
        assert!(
            scores[0] > scores[1],
            "relevant={:.3} should > irrelevant={:.3}",
            scores[0],
            scores[1]
        );
    }

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
        })
        .expect("remote combined");

        assert_eq!(combined.name(), "qwen3.5:2b");
    }
}
