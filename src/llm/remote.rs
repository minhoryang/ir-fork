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
