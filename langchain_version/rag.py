import ast
import operator

from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    EMBEDDING_MODEL,
    RETRIEVAL_DISTANCE_THRESHOLD,
)

SYSTEM_PROMPT = """You are a company knowledge assistant.
For company questions, answer only from the reference material and do not invent facts.
If the references do not answer a company-specific question, say that no relevant information was found.
For simple math, general knowledge, or casual conversation, answer directly when no company reference is needed.
Keep answers concise and mention the source files when references are used.
"""

SUMMARY_PROMPT = """Summarize this company knowledge conversation for long-term memory.
Keep confirmed facts, topics, references, and unresolved questions.
Remove repetition and do not add information that was not discussed.
Write a concise English summary."""

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def calculate_math(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            return ALLOWED_OPERATORS[type(node.op)](left, right)
        raise ValueError("Only simple mathematical expressions are supported")

    result = evaluate(tree)
    if abs(result) > 10**12:
        raise ValueError("The mathematical result is outside the supported range")
    return result


def extract_math_expression(question: str) -> str | None:
    cleaned = question.strip().replace("?", "")
    for prefix in ("what is ", "calculate ", "compute "):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    allowed = set("0123456789+-*/(). ")
    return cleaned if cleaned and set(cleaned) <= allowed else None


class LangChainRAG:
    def __init__(self) -> None:
        if not DEEPSEEK_API_KEY:
            raise ValueError("Configure DEEPSEEK_API_KEY in the .env file first")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=self.embeddings,
        )
        self.llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
            temperature=0.1,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                (
                    "human",
                    "Long-term memory:\n{summary}\n\nReference material:\n{context}\n\nCurrent question: {question}",
                ),
            ]
        )

    def summarize_history(self, messages: list[BaseMessage], existing_summary: str = "") -> str:
        summary_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SUMMARY_PROMPT),
                ("human", "Existing summary:\n{summary}\n\nNew conversation:\n{history}"),
            ]
        )
        history_text = "\n".join(f"{message.type}: {message.content}" for message in messages)
        response = self.llm.invoke(
            summary_prompt.invoke({"summary": existing_summary, "history": history_text})
        )
        return response.content or existing_summary

    def answer(
        self,
        question: str,
        history: list[BaseMessage] | None = None,
        summary: str = "",
    ) -> tuple[str, list[dict]]:
        math_expression = extract_math_expression(question)
        if math_expression:
            try:
                return f"{math_expression} = {calculate_math(math_expression):g}", []
            except (SyntaxError, ValueError, ZeroDivisionError):
                pass
        history = history or []
        recent_history = history[-6:]
        history_text = "\n".join(f"{message.type}: {message.content}" for message in recent_history)
        retrieval_query = f"Long-term memory:\n{summary}\n\nConversation history:\n{history_text}\n\nCurrent question:\n{question}"
        matches = self.vectorstore.similarity_search_with_score(retrieval_query, k=4)
        relevant = [(document, score) for document, score in matches if score <= RETRIEVAL_DISTANCE_THRESHOLD]
        context = "\n\n".join(
            f"[Source: {document.metadata.get('source', 'unknown')}]\n{document.page_content}"
            for document, _ in relevant
        )
        references = [
            {
                "source": document.metadata.get("source", "unknown"),
                "chunk": document.metadata.get("chunk", document.metadata.get("page", "unknown")),
                "text": document.page_content,
                "distance": score,
            }
            for document, score in relevant
        ]
        messages = self.prompt.invoke(
            {
                "chat_history": recent_history,
                "summary": summary,
                "context": context,
                "question": question,
            }
        )
        response = self.llm.invoke(messages)
        return response.content, references
