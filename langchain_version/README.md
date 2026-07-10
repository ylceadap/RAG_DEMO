# LangChain Version

This folder reimplements the core RAG pipeline with LangChain components.

## Run

From the repository root:

```bash
cp .env.example .env
../.venv/bin/python ingest.py --reset
../.venv/bin/streamlit run app.py
```

The implementation uses LangChain document loading, `RecursiveCharacterTextSplitter`, `HuggingFaceEmbeddings`, `Chroma`, `ChatPromptTemplate`, and the OpenAI-compatible DeepSeek chat model.

This version now supports incremental indexing, OCR fallback, duplicate-file detection, index metadata, relevance thresholds, simple math handling, recent-history plus summary memory, source snippets, and optional password protection.

## Tests

From the repository root:

```bash
PYTHONPATH=langchain_version .venv/bin/python -m unittest discover -s langchain_version/tests -p 'test_*.py'
```
