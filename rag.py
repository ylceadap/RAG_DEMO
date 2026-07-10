import chromadb  # 导入 ChromaDB，用于搜索相关文档片段。
from openai import OpenAI  # 导入 OpenAI 兼容客户端，用于调用 DeepSeek。
from sentence_transformers import SentenceTransformer  # 导入文本向量模型。

from config import CHROMA_DIR, COLLECTION_NAME, DEEPSEEK_API_KEY, DEEPSEEK_MODEL  # 导入配置项。
from ingest import EMBEDDING_MODEL  # 复用建立索引时使用的向量模型名称。

SYSTEM_PROMPT = """你是一个公司内部资料问答助手。
请严格根据参考资料回答，不要凭空编造。若参考资料中没有答案，请明确说“资料中没有找到相关信息”。
回答尽量简洁，并在最后列出使用的来源文件。
"""


class RAG:
    def __init__(self):  # 初始化 RAG 问答对象。
        if not DEEPSEEK_API_KEY:  # 判断是否配置了 DeepSeek API Key。
            raise ValueError("请先在 .env 文件中配置 DEEPSEEK_API_KEY")  # 没有配置时给出提示。
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)  # 加载与索引相同的向量模型。
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))  # 连接本地向量数据库。
        self.collection = client.get_or_create_collection(COLLECTION_NAME)  # 获取文档向量集合。
        self.llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")  # 创建 DeepSeek 客户端。

    def answer(  # 定义根据当前问题和历史对话生成答案的方法。
        self,  # 传入当前 RAG 对象。
        question: str,  # 接收用户当前的问题。
        history: list[dict] | None = None,  # 接收之前的聊天记录。
        top_k: int = 4,  # 设置最多检索的文档片段数。
    ) -> tuple[str, list[dict]]:  # 返回答案和参考资料。
        history = history or []  # 没有历史记录时使用空列表。
        recent_history = history[-6:]  # 只保留最近三轮对话，避免上下文无限变长。
        history_text = "\n".join(  # 将历史消息整理成检索文本。
            f"{message['role']}: {message['content']}"  # 标记消息角色和内容。
            for message in recent_history  # 遍历最近的历史消息。
        )  # 完成历史消息拼接。
        retrieval_query = f"对话历史:\n{history_text}\n\n当前问题:\n{question}"  # 将上下文和当前问题合并用于检索。
        query_embedding = self.embedder.encode([retrieval_query], normalize_embeddings=True).tolist()  # 将带上下文的问题转换为向量。
        result = self.collection.query(  # 搜索最相关的文本片段。
            query_embeddings=query_embedding,  # 使用问题向量进行检索。
            n_results=top_k,  # 返回最相关的若干片段。
            include=["documents", "metadatas", "distances"],  # 同时返回原文、来源和距离。
        )  # 完成向量检索。
        docs = result.get("documents", [[]])[0]  # 获取检索到的文本内容。
        metas = result.get("metadatas", [[]])[0]  # 获取检索结果的来源信息。
        distances = result.get("distances", [[]])[0]  # 获取每个片段与问题的距离。
        references = [  # 组合成网页需要展示的参考资料列表。
            {**meta, "text": doc, "distance": distance}  # 保存来源、原文和相似度距离。
            for doc, meta, distance in zip(docs, metas, distances)  # 同时遍历文本、来源和距离。
        ]  # 完成参考资料列表。
        context = "\n\n".join(  # 将多个文本片段拼接成参考资料。
            f"[来源: {meta['source']}，第 {meta['chunk']} 个片段]\n{doc}"  # 给每段资料标记来源。
            for doc, meta in zip(docs, metas)  # 同时遍历文本和对应的元数据。
        )  # 完成参考资料拼接。
        messages = [  # 创建发送给模型的消息列表。
            {"role": "system", "content": SYSTEM_PROMPT},  # 设置模型的回答规则。
        ]  # 先放入系统规则。
        messages.extend(recent_history)  # 加入最近几轮历史对话。
        messages.append(  # 加入当前问题和检索资料。
            {"role": "user", "content": f"参考资料:\n{context}\n\n当前问题: {question}"}  # 发送资料和当前问题。
        )  # 完成当前问题消息。
        response = self.llm.chat.completions.create(  # 调用 DeepSeek 对话接口。
            model=DEEPSEEK_MODEL,  # 指定使用的模型。
            temperature=0.1,  # 设置较低随机性，让回答更稳定。
            messages=messages,  # 传入系统规则、历史对话和当前问题。
        )  # 完成模型调用。
        return response.choices[0].message.content, references  # 返回答案和完整参考资料。
