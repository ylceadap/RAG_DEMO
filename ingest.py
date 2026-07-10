from pathlib import Path  # 导入路径工具，用于处理文档文件。

import chromadb  # 导入 ChromaDB，用于保存和检索向量。
from pypdf import PdfReader  # 导入 PDF 阅读器，用于提取 PDF 文本。
from sentence_transformers import SentenceTransformer  # 导入文本向量模型。

from config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR  # 导入项目路径和集合配置。

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 指定中文 Embedding 模型。
CHUNK_SIZE = 700  # 设置每个文本片段的最大字符数。
CHUNK_OVERLAP = 100  # 设置相邻文本片段之间重叠的字符数。


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


def build_index() -> tuple[int, int]:  # 定义建立向量索引的函数。
    DATA_DIR.mkdir(parents=True, exist_ok=True)  # 确保文档目录存在。
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))  # 创建持久化 ChromaDB 客户端。
    try:  # 尝试删除旧的向量集合。
        client.delete_collection(COLLECTION_NAME)  # 删除旧索引，避免重复添加数据。
    except Exception:  # 如果集合不存在，则忽略删除错误。
        pass  # 什么也不做，继续创建新集合。
    collection = client.create_collection(COLLECTION_NAME)  # 创建新的向量集合。
    model = SentenceTransformer(EMBEDDING_MODEL)  # 加载文本向量模型。

    documents, ids, metadatas = [], [], []  # 创建文本、ID 和元数据列表。
    for path in sorted(DATA_DIR.iterdir()):  # 遍历文档目录中的所有文件。
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:  # 判断文件是否为支持的格式。
            continue  # 跳过不支持的文件。
        for number, chunk in enumerate(split_text(read_file(path))):  # 读取文件并遍历切分后的片段。
            documents.append(chunk)  # 保存文本片段。
            ids.append(f"{path.name}-{number}")  # 为文本片段生成唯一 ID。
            metadatas.append({"source": path.name, "chunk": number + 1})  # 保存来源文件和片段编号。

    if documents:  # 只有存在文本片段时才生成向量。
        embeddings = model.encode(documents, normalize_embeddings=True).tolist()  # 将文本转换为归一化向量。
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)  # 写入 ChromaDB。
    return len({item["source"] for item in metadatas}), len(documents)  # 返回文件数和片段数。


if __name__ == "__main__":  # 判断当前文件是否被直接运行。
    files, chunks = build_index()  # 建立索引并接收统计结果。
    print(f"已导入 {files} 个文件，生成 {chunks} 个文本块。")  # 输出索引结果。
