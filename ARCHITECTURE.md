# RAG Demo Comparison Architecture

The detailed architecture and step-by-step notes are in [custom_version/ARCHITECTURE.md](custom_version/ARCHITECTURE.md). The repository now contains two parallel implementations:

```mermaid
flowchart LR
    A[Shared English documents] --> B[custom_version]
    A --> C[langchain_version]
    B --> D[Custom ingestion, ChromaDB, DeepSeek, Streamlit]
    C --> E[LangChain loaders, splitters, Chroma, DeepSeek, Streamlit]
    D --> F[Compare retrieval and answer quality]
    E --> F
```
