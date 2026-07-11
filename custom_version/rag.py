import chromadb  # 导入 ChromaDB，用于搜索相关文档片段。
from openai import OpenAI  # 导入 OpenAI 兼容客户端，用于调用 DeepSeek。
from sentence_transformers import SentenceTransformer  # 导入文本向量模型。

from config import CHROMA_DIR, COLLECTION_NAME, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, RETRIEVAL_DISTANCE_THRESHOLD  # Import configuration values.
from ingest import EMBEDDING_MODEL  # 复用建立索引时使用的向量模型名称。

SYSTEM_PROMPT = """You are a company knowledge assistant.
For company policies, employee handbook, and product questions, answer strictly from the reference material and do not invent facts.
If the reference material is unrelated but the question is general knowledge or casual conversation, answer directly.
If the question is company-specific and the reference material has no answer, say "No relevant information was found in the documents."
Keep answers concise and list the source files when reference material is used.
"""

SUMMARY_PROMPT = """You summarize a company knowledge assistant conversation for long-term memory.
Keep confirmed facts, topics of interest, references, and unresolved questions.
Remove greetings, repetition, and irrelevant details. Do not add information that was not discussed.
Write a concise English summary."""  # Set the summary-generation rules.

class RAG:
    def __init__(self):  # 初始化 RAG 问答对象。
        if not DEEPSEEK_API_KEY:  # 判断是否配置了 DeepSeek API Key。
            raise ValueError("Configure DEEPSEEK_API_KEY in the .env file first")  # Explain the missing configuration.
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)  # 加载与索引相同的向量模型。
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))  # 连接本地向量数据库。
        self.collection = client.get_or_create_collection(COLLECTION_NAME)  # 获取文档向量集合。
        self.llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")  # 创建 DeepSeek 客户端。

    def summarize_history(  # 定义把较早对话压缩成摘要的方法。
        self,  # 传入当前 RAG 对象。
        messages: list[dict],  # 接收需要被压缩的历史消息。
        existing_summary: str = "",  # 接收之前已经生成的摘要。
    ) -> str:  # 返回新的长期记忆摘要。
        history_text = "\n".join(  # 将待摘要的消息整理成文本。
            f"{message['role']}: {message['content']}"  # 标记消息角色和内容。
            for message in messages  # 遍历待摘要的历史消息。
        )  # 完成历史消息拼接。
        response = self.llm.chat.completions.create(  # 调用 DeepSeek 生成摘要。
            model=DEEPSEEK_MODEL,  # 指定使用的模型。
            temperature=0.1,  # 使用较低随机性，保持摘要稳定。
            messages=[  # 构造摘要请求消息。
                {"role": "system", "content": SUMMARY_PROMPT},  # 设置摘要规则。
                {  # 创建包含旧摘要和新增对话的用户消息。
                    "role": "user",  # 标记这是用户输入内容。
                    "content": f"已有摘要:\n{existing_summary}\n\n新增对话:\n{history_text}",  # 发送摘要和新增对话。
                },  # 结束用户消息。
            ],  # 结束消息列表。
        )  # 完成摘要生成。
        return response.choices[0].message.content or existing_summary  # 返回新摘要，异常为空时保留旧摘要。

    def answer(  # 定义根据当前问题和历史对话生成答案的方法。
        self,  # 传入当前 RAG 对象。
        question: str,  # 接收用户当前的问题。
        history: list[dict] | None = None,  # 接收之前的聊天记录。
        summary: str = "",  # 接收较早对话的摘要记忆。
        top_k: int = 4,  # 设置最多检索的文档片段数。
    ) -> tuple[str, list[dict]]:  # 返回答案和参考资料。
        history = history or []  # 没有历史记录时使用空列表。
        recent_history = history[-6:]  # 只保留最近三轮对话，避免上下文无限变长。
        history_text = "\n".join(  # 将历史消息整理成检索文本。
            f"{message['role']}: {message['content']}"  # 标记消息角色和内容。
            for message in recent_history  # 遍历最近的历史消息。
        )  # 完成历史消息拼接。
        retrieval_query = f"Long-term memory:\n{summary}\n\nConversation history:\n{history_text}\n\nCurrent question:\n{question}"  # Combine context for retrieval.
        query_embedding = self.embedder.encode([retrieval_query], normalize_embeddings=True).tolist()  # 将带上下文的问题转换为向量。
        result = self.collection.query(  # 搜索最相关的文本片段。
            query_embeddings=query_embedding,  # 使用问题向量进行检索。
            n_results=top_k,  # 返回最相关的若干片段。
            include=["documents", "metadatas", "distances"],  # 同时返回原文、来源和距离。
        )  # 完成向量检索。
        docs = result.get("documents", [[]])[0]  # 获取检索到的文本内容。
        metas = result.get("metadatas", [[]])[0]  # 获取检索结果的来源信息。
        distances = result.get("distances", [[]])[0]  # 获取每个片段与问题的距离。
        relevant_items = [  # Keep only results within the configured relevance threshold.
            (doc, meta, distance)  # Preserve the document, metadata, and distance.
            for doc, meta, distance in zip(docs, metas, distances)  # Iterate over retrieved results.
            if distance <= RETRIEVAL_DISTANCE_THRESHOLD  # Reject weakly related results.
        ]  # Finish relevance filtering.
        docs = [item[0] for item in relevant_items]  # Keep relevant document text only.
        metas = [item[1] for item in relevant_items]  # Keep relevant metadata only.
        distances = [item[2] for item in relevant_items]  # Keep relevant distances only.
        references = [  # 组合成网页需要展示的参考资料列表。
            {**meta, "text": doc, "distance": distance}  # 保存来源、原文和相似度距离。
            for doc, meta, distance in zip(docs, metas, distances)  # 同时遍历文本、来源和距离。
        ]  # 完成参考资料列表。
        context = "\n\n".join(  # 将多个文本片段拼接成参考资料。
            f"[Source: {meta['source']}, chunk {meta['chunk']}]\n{doc}"  # Label each document chunk.
            for doc, meta in zip(docs, metas)  # 同时遍历文本和对应的元数据。
        )  # 完成参考资料拼接。
        messages = [  # 创建发送给模型的消息列表。
            {"role": "system", "content": SYSTEM_PROMPT},  # 设置模型的回答规则。
        ]  # 先放入系统规则。
        messages.extend(recent_history)  # 加入最近几轮历史对话。
        messages.append(  # 加入当前问题和检索资料。
            {"role": "user", "content": f"Long-term memory:\n{summary}\n\nReference material:\n{context}\n\nCurrent question: {question}"}  # Send memory, references, and the current question.
        )  # 完成当前问题消息。
        response = self.llm.chat.completions.create(  # 调用 DeepSeek 对话接口。
            model=DEEPSEEK_MODEL,  # 指定使用的模型。
            temperature=0.1,  # 设置较低随机性，让回答更稳定。
            messages=messages,  # 传入系统规则、历史对话和当前问题。
        )  # 完成模型调用。
        return response.choices[0].message.content, references  # 返回答案和完整参考资料。
