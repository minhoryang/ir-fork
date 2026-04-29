1. ir 을 gpu가 없는 환경에서도 실행하고싶어.
2. gpu가 없는 환경에서 mcp모드로 띄우면 daemon처럼 동작한다는건 알겠어.
3. 문제는 model을 띄우는게 문제인데, ir 내에서 llama_cpp_2::llama_backend::LlamaBackend 로 모델을 띄우고 싶지 않아.
4. 모델은 내가 ollama로 띄워뒀어.
5. 준비된 모델과 주소는 다음과 같아.
- IR_EMBEDDING_MODEL
    - `http://127.0.0.1:11112/api/embeds`
        - model name: `embeddinggemma:300m` (preloaded in CPU)
        - model name: `bge-m3:567m` (preloaded in CPU)
- IR_RERANKER_MODEL / IR_EXPANDER_MODEL => IR_COMBINED_MODEL 쓰기로 함.
- IR_COMBINED_MODEL
    - `http://127.0.0.1:11111/api/generate`
        - model name: `qwen3.5:2b` (preloaded in GPU)
- `HF_HUB_OFFLINE=1` (모델 다운로드 금지)

6. 내가 환경변수에 hostname:port와 model_name을 구겨넣어서 실행하면, (eg: `IR_EMBEDDING_MODEL=http://127.0.0.1:11112,embeddinggemma:300m`)
7. 이를 파싱하고 분기해서, ir이 llama_cpp_2의 도움 없이, 그냥 바로 api 요청을 날렸으면 좋겠어.
8. 나는 코드의 최소부분만 surgical knife 방식으로 수정해서, 위의 목표가 가능한지 확인하고 싶어.
