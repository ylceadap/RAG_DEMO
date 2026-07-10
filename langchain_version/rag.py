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
Keep answers concise and mention the source files when references are used.
"""


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
                    "Reference material:\n{context}\n\nCurrent question: {question}",
                ),
            ]
        )

    def answer(self, question: str, history: list[BaseMessage] | None = None) -> tuple[str, list[dict]]:
        history = history or []
        history_text = "\n".join(f"{message.type}: {message.content}" for message in history[-6:])
        retrieval_query = f"Conversation history:\n{history_text}\n\nCurrent question:\n{question}"
        matches = self.vectorstore.similarity_search_with_score(retrieval_query, k=4)
        relevant = [(document, score) for document, score in matches if score <= RETRIEVAL_DISTANCE_THRESHOLD]
        context = "\n\n".join(
            f"[Source: {document.metadata.get('source', 'unknown')}]\n{document.page_content}"
            for document, _ in relevant
        )
        references = [
            {
                "source": document.metadata.get("source", "unknown"),
                "chunk": document.metadata.get("page", "unknown"),
                "text": document.page_content,
                "distance": score,
            }
            for document, score in relevant
        ]
        messages = self.prompt.invoke(
            {"chat_history": history[-6:], "context": context, "question": question}
        )
        response = self.llm.invoke(messages)
        return response.content, references
