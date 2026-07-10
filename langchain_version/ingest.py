import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR, EMBEDDING_MODEL

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def load_documents() -> list:
    documents = []
    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        if path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        else:
            loaded = TextLoader(str(path), encoding="utf-8").load()
        for document in loaded:
            document.metadata["source"] = path.name
        documents.extend(loaded)
    return documents


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})


def build_index(reset: bool = False) -> tuple[int, int]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    if chunks:
        ids = [f"{chunk.metadata['source']}-{number}" for number, chunk in enumerate(chunks)]
        vectorstore.add_documents(chunks, ids=ids)
    return len({document.metadata["source"] for document in documents}), len(chunks)


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    files, chunks = build_index(reset=reset)
    print(f"Indexed {files} files and created {chunks} chunks.")
