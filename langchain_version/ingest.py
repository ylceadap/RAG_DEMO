import hashlib
import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR, EMBEDDING_MODEL

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
INDEX_VERSION = "2"


def read_document(path: Path) -> list[Document]:
    if path.suffix.lower() != ".pdf":
        return [Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})]

    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        try:
            from pdf2image import convert_from_path
            import pytesseract
        except ImportError as exc:
            raise RuntimeError("Scanned PDF detected. Install pytesseract and pdf2image, plus Tesseract and Poppler.") from exc
        images = convert_from_path(str(path), dpi=200)
        text = "\n".join(pytesseract.image_to_string(image) for image in images)
    return [Document(page_content=text, metadata={"source": path.name})]


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})


def build_index(reset: bool = False) -> tuple[int, int]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    paths = [
        path
        for path in sorted(DATA_DIR.iterdir())
        if path.suffix.lower() in {".txt", ".md", ".pdf"}
    ]
    current_sources = {path.name for path in paths}
    existing = vectorstore.get(include=["metadatas"])
    existing_metadata = [meta for meta in existing.get("metadatas", []) if meta]
    existing_sources = {meta.get("source") for meta in existing_metadata}
    for stale_source in existing_sources - current_sources:
        vectorstore.delete(where={"source": stale_source})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    changed_files = 0
    added_chunks = 0
    for path in paths:
        documents = read_document(path)
        content = "\n".join(document.page_content for document in documents)
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        current_records = vectorstore.get(where={"source": path.name}, include=["metadatas"])
        current_metadata = current_records.get("metadatas", [])
        expected = {
            "file_hash": file_hash,
            "index_version": INDEX_VERSION,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }
        if current_metadata and all(
            all(meta.get(key) == value for key, value in expected.items())
            for meta in current_metadata
        ):
            continue

        duplicate_records = vectorstore.get(where={"file_hash": file_hash}, include=["metadatas"])
        duplicate_sources = {
            meta.get("source") for meta in duplicate_records.get("metadatas", []) if meta
        }
        if duplicate_sources - {path.name}:
            if current_metadata:
                vectorstore.delete(where={"source": path.name})
            continue
        if current_metadata:
            vectorstore.delete(where={"source": path.name})

        chunks = splitter.split_documents(documents)
        if not chunks:
            continue
        for chunk in chunks:
            chunk.metadata.update(expected)
        ids = [f"{path.name}-{number}" for number in range(len(chunks))]
        vectorstore.add_documents(chunks, ids=ids)
        changed_files += 1
        added_chunks += len(chunks)
    return changed_files, added_chunks


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    files, chunks = build_index(reset=reset)
    mode = "Full rebuild" if reset else "Incremental update"
    print(f"{mode}: updated {files} files and added {chunks} text chunks.")
