# LangChain Version

This folder reimplements the core RAG pipeline with LangChain components.

## Run

From the repository root:

```bash
cp .env.example .env
../.venv/bin/python ingest.py --reset
../.venv/bin/streamlit run app.py
```

The implementation uses LangChain document loaders, `RecursiveCharacterTextSplitter`, `HuggingFaceEmbeddings`, `Chroma`, `ChatPromptTemplate`, and the OpenAI-compatible DeepSeek chat model.

This version intentionally keeps the first implementation simple so it can be compared with `custom_version`. The custom version currently has more advanced incremental indexing and conversation summarization.
