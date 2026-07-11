import hashlib  # Import hashing utilities to detect changed files.
import sys  # Import command-line argument handling.
from pathlib import Path  # Import path utilities for document files.

import chromadb  # 导入 ChromaDB，用于保存和检索向量。
from pypdf import PdfReader  # 导入 PDF 阅读器，用于提取 PDF 文本。
from sentence_transformers import SentenceTransformer  # 导入文本向量模型。

from config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR  # 导入项目路径和集合配置。

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Use an English embedding model for the English corpus.
CHUNK_SIZE = 700  # Set the maximum size of each text chunk.
CHUNK_OVERLAP = 100  # Set the overlap between neighboring chunks.
INDEX_VERSION = "2"  # Increment this when the indexing schema changes.
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}  # Define the file types that can be indexed.


def read_file(path: Path) -> str:  # 定义读取单个文件的函数。
    if path.suffix.lower() == ".pdf":  # 判断文件是否为 PDF。
        reader = PdfReader(str(path))  # 打开 PDF 文件。
        text = "\n".join(page.extract_text() or "" for page in reader.pages)  # Extract text from each PDF page.
        if text.strip():  # Check whether the PDF contains selectable text.
            return text  # Return the extracted text.
        try:  # Try the optional OCR fallback for scanned PDFs.
            from pdf2image import convert_from_path  # Convert PDF pages to images.
            import pytesseract  # Run OCR on the page images.
        except ImportError as exc:  # Handle missing OCR dependencies.
            raise RuntimeError("Scanned PDF detected. Install pytesseract and pdf2image, plus the Tesseract and Poppler system tools.") from exc  # Explain how to enable OCR.
        images = convert_from_path(str(path), dpi=200)  # Render scanned pages at a readable resolution.
        return "\n".join(pytesseract.image_to_string(image) for image in images)  # Extract text from the scanned pages.
    return path.read_text(encoding="utf-8")  # 读取普通文本或 Markdown 文件。


def split_text(text: str) -> list[str]:  # 定义切分长文本的函数。
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())  # 清理空行和每行两端空格。
    chunks = []  # 创建列表，用于保存切分后的文本片段。
    start = 0  # 设置当前片段的起始位置。
    while start < len(text):  # 只要还没有处理完整段文字，就继续循环。
        end = min(start + CHUNK_SIZE, len(text))  # 计算当前片段的结束位置。
        chunks.append(text[start:end])  # 将当前片段加入列表。
        if end == len(text):  # 判断是否已经处理到文本末尾。
            break  # 到达末尾后退出循环。
        start = end - CHUNK_OVERLAP  # 从重叠位置开始下一个片段。
    return chunks  # 返回所有文本片段。


def open_collection(reset: bool):  # Open a persistent collection, optionally replacing it.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except ValueError:  # The collection may not exist on the first full rebuild.
            pass
        return client.create_collection(COLLECTION_NAME)
    return client.get_or_create_collection(COLLECTION_NAME)


def get_document_paths() -> list[Path]:  # List source files in a stable order.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return [
        path
        for path in sorted(DATA_DIR.iterdir())
        if path.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def get_metadata(collection, where: dict | None = None) -> list[dict]:  # Read non-empty metadata records.
    records = collection.get(where=where, include=["metadatas"]) if where else collection.get(include=["metadatas"])
    return [metadata for metadata in records.get("metadatas", []) if metadata]


def delete_stale_sources(collection, paths: list[Path]) -> None:  # Remove vectors for files deleted from DATA_DIR.
    current_sources = {path.name for path in paths}
    indexed_sources = {metadata.get("source") for metadata in get_metadata(collection)}
    for source in indexed_sources - current_sources:
        collection.delete(where={"source": source})


def create_index_metadata(text: str) -> dict:  # Describe the content and settings used to create an index.
    return {
        "file_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
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


def has_duplicate_source(collection, file_hash: str, source: str) -> bool:  # Check whether another file already has identical content.
    duplicate_sources = {
        metadata.get("source")
        for metadata in get_metadata(collection, where={"file_hash": file_hash})
    }
    return bool(duplicate_sources - {source})


def write_file_chunks(collection, model: SentenceTransformer, path: Path, chunks: list[str], metadata: dict) -> None:  # Embed and save one file.
    ids = [f"{path.name}-{number}" for number in range(len(chunks))]
    metadatas = [
        {"source": path.name, "chunk": number + 1, **metadata}
        for number in range(len(chunks))
    ]
    embeddings = model.encode(chunks, normalize_embeddings=True).tolist()
    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)


def build_index(reset: bool = False) -> tuple[int, int]:  # Build or incrementally update the vector index.
    collection = open_collection(reset)
    paths = get_document_paths()
    delete_stale_sources(collection, paths)

    model = None  # Load the embedding model only if a file actually needs indexing.
    changed_files = 0
    added_chunks = 0
    for path in paths:
        text = read_file(path)
        metadata = create_index_metadata(text)
        current_metadata = get_metadata(collection, where={"source": path.name})

        if is_current(current_metadata, metadata):
            continue
        if has_duplicate_source(collection, metadata["file_hash"], path.name):
            if current_metadata:
                collection.delete(where={"source": path.name})
            continue

        if current_metadata:
            collection.delete(where={"source": path.name})
        chunks = split_text(text)
        if not chunks:
            continue
        if model is None:
            model = SentenceTransformer(EMBEDDING_MODEL)
        write_file_chunks(collection, model, path, chunks, metadata)
        changed_files += 1
        added_chunks += len(chunks)

    return changed_files, added_chunks


if __name__ == "__main__":  # 判断当前文件是否被直接运行。
    reset = "--reset" in sys.argv  # Enable full rebuild only when --reset is provided.
    files, chunks = build_index(reset=reset)  # Run an incremental update or full rebuild.
    mode = "Full rebuild" if reset else "Incremental update"  # Describe the selected indexing mode.
    print(f"{mode}: updated {files} files and added {chunks} text chunks.")  # Print the indexing result.
