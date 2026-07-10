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


def build_index(reset: bool = False) -> tuple[int, int]:  # Build or incrementally update the vector index.
    DATA_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the document directory exists.
    if reset and CHROMA_DIR.exists():  # Check whether a full rebuild was requested.
        shutil.rmtree(CHROMA_DIR)  # Delete the old LangChain vector database.

    embeddings = get_embeddings()  # Load the embedding model used by the vector store.
    vectorstore = Chroma(  # Create a persistent LangChain Chroma store.
        collection_name=COLLECTION_NAME,  # Use the LangChain-specific collection name.
        persist_directory=str(CHROMA_DIR),  # Persist vectors under this version's directory.
        embedding_function=embeddings,  # Tell Chroma how to embed new documents.
    )  # Finish creating the vector store.
    paths = [  # Find all supported source files.
        path  # Keep the current path.
        for path in sorted(DATA_DIR.iterdir())  # Iterate through the document directory.
        if path.suffix.lower() in {".txt", ".md", ".pdf"}  # Keep supported file formats only.
    ]  # Finish collecting paths.
    current_sources = {path.name for path in paths}  # Record files that still exist on disk.
    existing = vectorstore.get(include=["metadatas"])  # Read metadata from indexed chunks.
    existing_metadata = [meta for meta in existing.get("metadatas", []) if meta]  # Remove empty metadata records.
    existing_sources = {meta.get("source") for meta in existing_metadata}  # Find indexed source filenames.
    for stale_source in existing_sources - current_sources:  # Find files deleted from the source directory.
        vectorstore.delete(where={"source": stale_source})  # Remove their old chunks.

    splitter = RecursiveCharacterTextSplitter(  # Create LangChain's recursive text splitter.
        chunk_size=CHUNK_SIZE,  # Set the target chunk size.
        chunk_overlap=CHUNK_OVERLAP,  # Preserve overlap between chunks.
        separators=["\n\n", "\n", " ", ""],  # Prefer paragraph, line, word, then character boundaries.
    )  # Finish the splitter configuration.
    changed_files = 0  # Count new or changed files.
    added_chunks = 0  # Count chunks written to Chroma.
    for path in paths:  # Process each source file.
        documents = read_document(path)  # Load the current file.
        content = "\n".join(document.page_content for document in documents)  # Combine its text for hashing.
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()  # Create a stable content hash.
        current_records = vectorstore.get(where={"source": path.name}, include=["metadatas"])  # Read this file's old records.
        current_metadata = current_records.get("metadatas", [])  # Extract the old metadata.
        expected = {  # Define metadata required for a compatible index.
            "file_hash": file_hash,  # Match the current file content.
            "index_version": INDEX_VERSION,  # Match the current index schema.
            "embedding_model": EMBEDDING_MODEL,  # Match the current embedding model.
            "chunk_size": CHUNK_SIZE,  # Match the current chunk size.
            "chunk_overlap": CHUNK_OVERLAP,  # Match the current overlap.
        }  # Finish the compatibility metadata.
        if current_metadata and all(  # Check whether every existing chunk is current.
            all(meta.get(key) == value for key, value in expected.items())  # Compare every expected field.
            for meta in current_metadata  # Check every chunk metadata record.
        ):  # Finish the compatibility check.
            continue  # Skip unchanged and compatible files.

        duplicate_records = vectorstore.get(where={"file_hash": file_hash}, include=["metadatas"])  # Search for identical content.
        duplicate_sources = {  # Collect source names for matching content.
            meta.get("source") for meta in duplicate_records.get("metadatas", []) if meta  # Ignore empty records.
        }  # Finish duplicate source collection.
        if duplicate_sources - {path.name}:  # Avoid indexing the same content twice.
            if current_metadata:  # Remove an outdated copy of this source if necessary.
                vectorstore.delete(where={"source": path.name})  # Delete its old vectors.
            continue  # Skip this duplicate file.
        if current_metadata:  # Check whether an older version must be replaced.
            vectorstore.delete(where={"source": path.name})  # Remove its old chunks.

        chunks = splitter.split_documents(documents)  # Split documents into retrievable chunks.
        if not chunks:  # Check whether the file contains usable text.
            continue  # Skip empty files.
        for chunk in chunks:  # Add index metadata to every chunk.
            chunk.metadata.update(expected)  # Store version and content information with the chunk.
        ids = [f"{path.name}-{number}" for number in range(len(chunks))]  # Create stable chunk IDs.
        vectorstore.add_documents(chunks, ids=ids)  # Embed and persist only this file's chunks.
        changed_files += 1  # Count the updated file.
        added_chunks += len(chunks)  # Count its new chunks.
    return changed_files, added_chunks  # Return update statistics.


if __name__ == "__main__":  # Run this block only when called as a script.
    reset = "--reset" in sys.argv  # Detect the optional full-rebuild flag.
    files, chunks = build_index(reset=reset)  # Build or update the index.
    mode = "Full rebuild" if reset else "Incremental update"  # Describe the selected mode.
    print(f"{mode}: updated {files} files and added {chunks} text chunks.")  # Print the result.
