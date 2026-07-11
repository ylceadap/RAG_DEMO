import hashlib  # Detect whether document content has changed.
import shutil  # Remove the local vector database during a full rebuild.
import sys  # Read the optional --reset command-line flag.
from pathlib import Path  # Work with document paths.

from langchain_chroma import Chroma  # Connect LangChain to ChromaDB.
from langchain_core.documents import Document  # Represent loaded source documents.
from langchain_huggingface import HuggingFaceEmbeddings  # Create embeddings through LangChain.
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Split documents recursively.
from pypdf import PdfReader  # Extract selectable text from PDFs.

from config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR, EMBEDDING_MODEL  # Import project configuration.

CHUNK_SIZE = 700  # Set the maximum chunk size.
CHUNK_OVERLAP = 100  # Preserve context between neighboring chunks.
INDEX_VERSION = "2"  # Change this when the index metadata schema changes.
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}  # Define the file types that can be indexed.


def read_document(path: Path) -> list[Document]:  # Load one file into LangChain Documents.
    if path.suffix.lower() != ".pdf":  # Handle text and Markdown files directly.
        return [  # Return one LangChain document for the text file.
            Document(  # Wrap raw text in LangChain's standard document object.
                page_content=path.read_text(encoding="utf-8"),  # Read UTF-8 file content.
                metadata={"source": path.name},  # Keep the source filename for citations.
            )  # Finish the document object.
        ]  # Finish the document list.

    reader = PdfReader(str(path))  # Open the PDF file.
    text = "\n".join(page.extract_text() or "" for page in reader.pages)  # Extract selectable PDF text.
    if not text.strip():  # Detect a scanned PDF with no selectable text.
        try:  # Load optional OCR dependencies only when needed.
            from pdf2image import convert_from_path  # Render PDF pages as images.
            import pytesseract  # Convert page images into text.
        except ImportError as exc:  # Explain missing OCR dependencies.
            raise RuntimeError("Scanned PDF detected. Install pytesseract and pdf2image, plus Tesseract and Poppler.") from exc  # Stop with an actionable error.
        images = convert_from_path(str(path), dpi=200)  # Render pages at a readable resolution.
        text = "\n".join(pytesseract.image_to_string(image) for image in images)  # OCR all rendered pages.
    return [Document(page_content=text, metadata={"source": path.name})]  # Return the extracted document.


def get_embeddings() -> HuggingFaceEmbeddings:  # Create the LangChain embedding adapter.
    return HuggingFaceEmbeddings(  # Construct the embedding model wrapper.
        model_name=EMBEDDING_MODEL,  # Select the configured English model.
        encode_kwargs={"normalize_embeddings": True},  # Normalize vectors for cosine-style comparison.
    )  # Return the embedding adapter.


def open_vectorstore(reset: bool) -> Chroma:  # Open a persistent store, optionally replacing its on-disk data.
    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    return Chroma(
        collection_name=COLLECTION_NAME,  # Use the LangChain-specific collection name.
        persist_directory=str(CHROMA_DIR),  # Persist vectors under this version's directory.
        embedding_function=get_embeddings(),  # Tell Chroma how to embed new documents.
    )


def get_document_paths() -> list[Path]:  # List source files in a stable order.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return [
        path
        for path in sorted(DATA_DIR.iterdir())
        if path.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def get_metadata(vectorstore: Chroma, where: dict | None = None) -> list[dict]:  # Read non-empty metadata records.
    records = vectorstore.get(where=where, include=["metadatas"]) if where else vectorstore.get(include=["metadatas"])
    return [metadata for metadata in records.get("metadatas", []) if metadata]


def delete_stale_sources(vectorstore: Chroma, paths: list[Path]) -> None:  # Remove vectors for files deleted from DATA_DIR.
    current_sources = {path.name for path in paths}
    indexed_sources = {metadata.get("source") for metadata in get_metadata(vectorstore)}
    for source in indexed_sources - current_sources:
        vectorstore.delete(where={"source": source})


def create_index_metadata(content: str) -> dict:  # Describe the content and settings used to create an index.
    return {
        "file_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "index_version": INDEX_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


def is_current(metadata_records: list[dict], expected: dict) -> bool:  # Check whether every stored chunk matches current settings.
    return bool(metadata_records) and all(
        all(metadata.get(key) == value for key, value in expected.items())
        for metadata in metadata_records
    )


def has_duplicate_source(vectorstore: Chroma, file_hash: str, source: str) -> bool:  # Check whether another file already has identical content.
    duplicate_sources = {
        metadata.get("source")
        for metadata in get_metadata(vectorstore, where={"file_hash": file_hash})
    }
    return bool(duplicate_sources - {source})


def get_text_splitter() -> RecursiveCharacterTextSplitter:  # Create the configured LangChain text splitter.
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )


def write_file_chunks(vectorstore: Chroma, splitter: RecursiveCharacterTextSplitter, path: Path, documents: list[Document], metadata: dict) -> int:  # Split and save one file.
    chunks = splitter.split_documents(documents)
    if not chunks:
        return 0
    for chunk in chunks:
        chunk.metadata.update(metadata)
    ids = [f"{path.name}-{number}" for number in range(len(chunks))]
    vectorstore.add_documents(chunks, ids=ids)
    return len(chunks)


def build_index(reset: bool = False) -> tuple[int, int]:  # Build or incrementally update the vector index.
    vectorstore = open_vectorstore(reset)
    paths = get_document_paths()
    delete_stale_sources(vectorstore, paths)
    splitter = get_text_splitter()

    changed_files = 0  # Count new or changed files.
    added_chunks = 0  # Count chunks written to Chroma.
    for path in paths:
        documents = read_document(path)
        content = "\n".join(document.page_content for document in documents)
        metadata = create_index_metadata(content)
        current_metadata = get_metadata(vectorstore, where={"source": path.name})

        if is_current(current_metadata, metadata):
            continue
        if has_duplicate_source(vectorstore, metadata["file_hash"], path.name):
            if current_metadata:
                vectorstore.delete(where={"source": path.name})
            continue

        if current_metadata:
            vectorstore.delete(where={"source": path.name})
        chunk_count = write_file_chunks(vectorstore, splitter, path, documents, metadata)
        if not chunk_count:
            continue
        changed_files += 1  # Count the updated file.
        added_chunks += chunk_count
    return changed_files, added_chunks  # Return update statistics.


if __name__ == "__main__":  # Run this block only when called as a script.
    reset = "--reset" in sys.argv  # Detect the optional full-rebuild flag.
    files, chunks = build_index(reset=reset)  # Build or update the index.
    mode = "Full rebuild" if reset else "Incremental update"  # Describe the selected mode.
    print(f"{mode}: updated {files} files and added {chunks} text chunks.")  # Print the result.
