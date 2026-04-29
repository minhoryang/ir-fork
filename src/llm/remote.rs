#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OllamaRemote {
    pub base_url: String,
    pub model_name: String,
}

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
