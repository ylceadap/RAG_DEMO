import hashlib  # Import hashing utilities to detect changed files.
import sys  # Import command-line argument handling.
from pathlib import Path  # Import path utilities for document files.

import chromadb  # 导入 ChromaDB，用于保存和检索向量。
from pypdf import PdfReader  # 导入 PDF 阅读器，用于提取 PDF 文本。
from sentence_transformers import SentenceTransformer  # 导入文本向量模型。

from config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR  # 导入项目路径和集合配置。

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # Use the embedding model configured for the corpus.
CHUNK_SIZE = 700  # Set the maximum size of each text chunk.
CHUNK_OVERLAP = 100  # Set the overlap between neighboring chunks.


def read_file(path: Path) -> str:  # 定义读取单个文件的函数。
    if path.suffix.lower() == ".pdf":  # 判断文件是否为 PDF。
        reader = PdfReader(str(path))  # 打开 PDF 文件。
        return "\n".join(page.extract_text() or "" for page in reader.pages)  # 提取并合并每一页文字。
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


def build_index(reset: bool = False) -> tuple[int, int]:  # Build or incrementally update the vector index.
    DATA_DIR.mkdir(parents=True, exist_ok=True)  # Ensure that the document directory exists.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))  # Create a persistent ChromaDB client.
    collection = client.get_or_create_collection(COLLECTION_NAME)  # Open or create the document collection.
    if reset:  # Check whether a full rebuild was explicitly requested.
        client.delete_collection(COLLECTION_NAME)  # Delete the existing collection.
        collection = client.create_collection(COLLECTION_NAME)  # Create a clean collection.

    paths = [  # Collect all supported document paths.
        path  # Keep the current path.
        for path in sorted(DATA_DIR.iterdir())  # Iterate through the document directory.
        if path.suffix.lower() in {".txt", ".md", ".pdf"}  # Keep only supported formats.
    ]  # Finish collecting document paths.
    current_sources = {path.name for path in paths}  # Record the files that currently exist.
    existing = collection.get(include=["metadatas"])  # Read existing metadata from the collection.
    existing_sources = {meta.get("source") for meta in existing.get("metadatas", []) if meta}  # Find indexed sources.
    for stale_source in existing_sources - current_sources:  # Find files removed from the document directory.
        collection.delete(where={"source": stale_source})  # Remove stale vectors from the index.

    model = None  # Delay loading the embedding model until a file needs indexing.
    changed_files = 0  # Count new or changed files.
    added_chunks = 0  # Count newly generated chunks.
    for path in paths:  # Process each current document.
        text = read_file(path)  # Read the document content.
        file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()  # Calculate a stable content hash.
        current_records = collection.get(where={"source": path.name}, include=["metadatas"])  # Read this file's existing records.
        current_metadata = current_records.get("metadatas", [])  # Extract this file's metadata.
        if current_metadata and all(meta.get("file_hash") == file_hash for meta in current_metadata):  # Check whether the file is unchanged.
            continue  # Skip unchanged files and avoid unnecessary embedding work.
        if current_metadata:  # Check whether an older version of this file is indexed.
            collection.delete(where={"source": path.name})  # Remove the older chunks before replacing them.
        chunks = split_text(text)  # Split the new or changed document into chunks.
        if not chunks:  # Skip empty documents.
            continue  # Continue with the next document.
        if model is None:  # Load the model only when indexing is required.
            model = SentenceTransformer(EMBEDDING_MODEL)  # Load the embedding model.
        ids = [f"{path.name}-{number}" for number in range(len(chunks))]  # Create stable IDs for the chunks.
        metadatas = [  # Create metadata for each chunk.
            {"source": path.name, "chunk": number + 1, "file_hash": file_hash}  # Store source, position, and content hash.
            for number in range(len(chunks))  # Iterate through chunk positions.
        ]  # Finish creating metadata.
        embeddings = model.encode(chunks, normalize_embeddings=True).tolist()  # Convert chunks into vectors.
        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)  # Add or replace only this file's vectors.
        changed_files += 1  # Count this new or changed file.
        added_chunks += len(chunks)  # Count the chunks written for this file.
    return changed_files, added_chunks  # Return the update statistics.


if __name__ == "__main__":  # 判断当前文件是否被直接运行。
    reset = "--reset" in sys.argv  # Enable full rebuild only when --reset is provided.
    files, chunks = build_index(reset=reset)  # Run an incremental update or full rebuild.
    mode = "Full rebuild" if reset else "Incremental update"  # Describe the selected indexing mode.
    print(f"{mode}: updated {files} files and added {chunks} text chunks.")  # Print the indexing result.
