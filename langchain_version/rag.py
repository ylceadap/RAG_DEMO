from langchain_chroma import Chroma
from langchain_core.documents import Document
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
#vvvvcc
RETRIEVAL_TOP_K = 4
RECENT_MESSAGE_LIMIT = 6

SYSTEM_PROMPT = """You are a company knowledge assistant.
For company questions, answer only from the reference material and do not invent facts.
If the references do not answer a company-specific question, say that no relevant information was found.
For general knowledge or casual conversation, answer directly when no company reference is needed.
Keep answers concise and mention the source files when references are used.
"""

SUMMARY_PROMPT = """Summarize this company knowledge conversation for long-term memory.
Keep confirmed facts, topics, references, and unresolved questions.
Remove repetition and do not add information that was not discussed.
Write a concise English summary."""


def create_embeddings() -> HuggingFaceEmbeddings:
    """Create the embedding adapter shared by indexing and retrieval."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Connect to this version's persistent Chroma collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def create_chat_model() -> ChatOpenAI:
    """Configure DeepSeek through its OpenAI-compatible API."""
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0.1,
    )


def create_answer_prompt() -> ChatPromptTemplate:
    """Create the reusable prompt used to answer a question."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            (
                "human",
                "Long-term memory:\n{summary}\n\nReference material:\n{context}\n\nCurrent question: {question}",
            ),
        ]
    )


def create_summary_prompt() -> ChatPromptTemplate:
    """Create the prompt used to condense older conversation turns."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SUMMARY_PROMPT),
            ("human", "Existing summary:\n{summary}\n\nNew conversation:\n{history}"),
        ]
    )


def serialize_messages(messages: list[BaseMessage]) -> str:
    """Convert LangChain messages into readable text for retrieval or summarization."""
    return "\n".join(f"{message.type}: {message.content}" for message in messages)


def build_retrieval_query(question: str, history: list[BaseMessage], summary: str) -> str:
    """Combine the question with recent and long-term conversation context."""
    return (
        f"Long-term memory:\n{summary}\n\n"
        f"Conversation history:\n{serialize_messages(history)}\n\n"
        f"Current question:\n{question}"
    )


def build_context(matches: list[tuple[Document, float]]) -> str:
    """Format retrieved chunks as reference material for the chat model."""
    return "\n\n".join(
        f"[Source: {document.metadata.get('source', 'unknown')}]\n{document.page_content}"
        for document, _ in matches
    )


def build_references(matches: list[tuple[Document, float]]) -> list[dict]:
    """Create source records for the Streamlit interface."""
    return [
        {
            "source": document.metadata.get("source", "unknown"),
            "chunk": document.metadata.get("chunk", document.metadata.get("page", "unknown")),
            "text": document.page_content,
            "distance": score,
        }
        for document, score in matches
    ]


class LangChainRAG:
    """Coordinate conversation memory, document retrieval, and answer generation."""

    def __init__(self) -> None:
        if not DEEPSEEK_API_KEY:
            raise ValueError("Configure DEEPSEEK_API_KEY in the .env file first")

        self.embeddings = create_embeddings()
        self.vectorstore = create_vectorstore(self.embeddings)
        self.llm = create_chat_model()
        self.summary_prompt = create_summary_prompt()
        self.answer_prompt = create_answer_prompt()

    def summarize_history(self, messages: list[BaseMessage], existing_summary: str = "") -> str:
        """Compress older conversation turns into a persistent summary."""
        messages = self.summary_prompt.invoke(
            {
                "summary": existing_summary,
                "history": serialize_messages(messages),
            }
        )
        response = self.llm.invoke(messages)
        return response.content or existing_summary

    def retrieve(self, question: str, history: list[BaseMessage], summary: str) -> list[tuple[Document, float]]:
        """Return only document chunks that meet the relevance threshold."""
        query = build_retrieval_query(question, history, summary)
        matches = self.vectorstore.similarity_search_with_score(query, k=RETRIEVAL_TOP_K)
        return [
            (document, score)
            for document, score in matches
            if score <= RETRIEVAL_DISTANCE_THRESHOLD
        ]

    def answer(
        self,
        question: str,
        history: list[BaseMessage] | None = None,
        summary: str = "",
    ) -> tuple[str, list[dict]]:
        """Retrieve relevant context, then generate an answer and citations."""
        recent_history = (history or [])[-RECENT_MESSAGE_LIMIT:]
        matches = self.retrieve(question, recent_history, summary)
        references = build_references(matches)
        messages = self.answer_prompt.invoke(
            {
                "chat_history": recent_history,
                "summary": summary,
                "context": build_context(matches),
                "question": question,
            }
        )
        response = self.llm.invoke(messages)
        return response.content, references
