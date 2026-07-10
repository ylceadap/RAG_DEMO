# RAG Demo Comparison

This repository contains two implementations of the same company knowledge assistant:

```text
RAG_DEMO/
├── custom_version/     # Hand-built baseline with custom indexing and retrieval logic
└── langchain_version/  # LangChain implementation of the same core pipeline
```

Both versions use the same English sample documents and the same DeepSeek-compatible API. Their vector indexes are kept separate so the implementations can be tested independently.

## Run the custom baseline

```bash
cd custom_version
../.venv/bin/python ingest.py --reset
../.venv/bin/streamlit run app.py
```

## Run the LangChain version

```bash
cd langchain_version
../.venv/bin/python ingest.py --reset
../.venv/bin/streamlit run app.py
```

Copy the environment template into the version you want to run, or place a shared `.env` file in the repository root:

```bash
cp custom_version/.env.example .env
```

## Compare the implementations

Use the same questions and compare:

- Retrieval source accuracy
- Answer quality
- Citation quality
- Response latency
- Token usage
- Amount of custom code

The custom implementation is the baseline for understanding the mechanics. The LangChain implementation demonstrates how loaders, splitters, embeddings, vector stores, retrievers, prompts, and chat models can be composed through a framework.
