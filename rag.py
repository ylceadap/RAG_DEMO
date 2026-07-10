import ast  # 导入语法树工具，用于安全解析简单数学表达式。
import operator  # 导入运算符函数，用于执行允许的数学运算。

import chromadb  # 导入 ChromaDB，用于搜索相关文档片段。
from openai import OpenAI  # 导入 OpenAI 兼容客户端，用于调用 DeepSeek。
from sentence_transformers import SentenceTransformer  # 导入文本向量模型。

from config import CHROMA_DIR, COLLECTION_NAME, DEEPSEEK_API_KEY, DEEPSEEK_MODEL  # 导入配置项。
from ingest import EMBEDDING_MODEL  # 复用建立索引时使用的向量模型名称。

SYSTEM_PROMPT = """You are a company knowledge assistant.
For company policies, employee handbook, and product questions, answer strictly from the reference material and do not invent facts.
If the reference material is unrelated but the question is simple math, general knowledge, or casual conversation, answer directly.
If the question is company-specific and the reference material has no answer, say "No relevant information was found in the documents."
Keep answers concise and list the source files when reference material is used.
"""

SUMMARY_PROMPT = """You summarize a company knowledge assistant conversation for long-term memory.
Keep confirmed facts, topics of interest, references, and unresolved questions.
Remove greetings, repetition, and irrelevant details. Do not add information that was not discussed.
Write a concise English summary."""  # Set the summary-generation rules.

ALLOWED_OPERATORS = {  # 设置允许执行的数学运算。
    ast.Add: operator.add,  # 允许加法。
    ast.Sub: operator.sub,  # 允许减法。
    ast.Mult: operator.mul,  # 允许乘法。
    ast.Div: operator.truediv,  # 允许除法。
    ast.Pow: operator.pow,  # 允许乘方。
}


def calculate_math(expression: str) -> int | float:  # 安全计算简单数学表达式。
    tree = ast.parse(expression, mode="eval")  # 将表达式解析成语法树。

    def evaluate(node: ast.AST) -> int | float:  # 递归计算语法树节点。
        if isinstance(node, ast.Expression):  # 判断是否为表达式根节点。
            return evaluate(node.body)  # 继续计算根节点中的实际内容。
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):  # 判断是否为数字。
            return node.value  # 返回数字值。
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):  # 判断是否为正负号。
            value = evaluate(node.operand)  # 计算正负号后面的数字。
            return value if isinstance(node.op, ast.UAdd) else -value  # 返回正数或负数。
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:  # 判断是否为允许的二元运算。
            left = evaluate(node.left)  # 计算左侧数字。
            right = evaluate(node.right)  # 计算右侧数字。
            return ALLOWED_OPERATORS[type(node.op)](left, right)  # 执行数学运算。
        raise ValueError("Only simple mathematical expressions are supported")  # Reject unsafe functions and variables.

    result = evaluate(tree)  # 计算完整表达式。
    if abs(result) > 10**12:  # 限制结果大小，避免异常计算。
        raise ValueError("The mathematical result is outside the supported range")  # Reject excessively large results.
    return result  # 返回数学结果。


def extract_math_expression(question: str) -> str | None:  # 从问题中提取简单数学表达式。
    cleaned = question.strip().replace("?", "")  # Remove surrounding spaces and question marks.
    for prefix in ("what is ", "calculate ", "compute "):  # Check common English math-question prefixes.
        if cleaned.lower().startswith(prefix):  # Detect a recognized prefix.
            cleaned = cleaned[len(prefix):].strip()  # Remove the question prefix.
            break  # Stop after finding a prefix.
    for suffix in (" equals", " equal to"):  # Check common English math-question suffixes.
        if cleaned.endswith(suffix):  # 判断问题是否以数学后缀结尾。
            cleaned = cleaned[: -len(suffix)].strip()  # Remove the question suffix.
            break  # Stop after finding a suffix.
    allowed = set("0123456789+-*/(). ")  # Define allowed expression characters.
    return cleaned if cleaned and set(cleaned) <= allowed else None  # Return only a pure math expression.


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
        math_expression = extract_math_expression(question)  # 判断当前问题是否为简单数学题。
        if math_expression:  # 如果识别出数学表达式，就直接本地计算。
            try:  # 尝试计算数学表达式。
                result = calculate_math(math_expression)  # 执行安全的数学计算。
                return f"{math_expression} = {result:g}", []  # 直接返回计算结果，不检索公司文档。
            except (SyntaxError, ValueError, ZeroDivisionError):  # 数学表达式不合法时继续走普通问答流程。
                pass  # 忽略计算错误，交给后续 RAG 流程处理。
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
