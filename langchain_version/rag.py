import ast  # Parse simple math expressions safely.
import operator  # Provide the allowed arithmetic operations.

from langchain_chroma import Chroma  # Connect the RAG pipeline to ChromaDB.
from langchain_core.messages import BaseMessage  # Type chat history messages.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Build reusable prompts.
from langchain_huggingface import HuggingFaceEmbeddings  # Load the embedding model through LangChain.
from langchain_openai import ChatOpenAI  # Call DeepSeek through its OpenAI-compatible API.

from config import (  # Import paths, model settings, and retrieval configuration.
    CHROMA_DIR,  # Location of the vector database.
    COLLECTION_NAME,  # Name of the Chroma collection.
    DEEPSEEK_API_KEY,  # API credential for DeepSeek.
    DEEPSEEK_MODEL,  # Chat model name.
    EMBEDDING_MODEL,  # Embedding model name.
    RETRIEVAL_DISTANCE_THRESHOLD,  # Maximum accepted retrieval distance.
)

SYSTEM_PROMPT = """You are a company knowledge assistant.
For company questions, answer only from the reference material and do not invent facts.
If the references do not answer a company-specific question, say that no relevant information was found.
For simple math, general knowledge, or casual conversation, answer directly when no company reference is needed.
Keep answers concise and mention the source files when references are used.
"""  # Tell the model how to ground its answers.

SUMMARY_PROMPT = """Summarize this company knowledge conversation for long-term memory.
Keep confirmed facts, topics, references, and unresolved questions.
Remove repetition and do not add information that was not discussed.
Write a concise English summary."""  # Tell the model how to compress older conversation.

ALLOWED_OPERATORS = {  # Restrict local math to safe arithmetic operators.
    ast.Add: operator.add,  # Allow addition.
    ast.Sub: operator.sub,  # Allow subtraction.
    ast.Mult: operator.mul,  # Allow multiplication.
    ast.Div: operator.truediv,  # Allow division.
    ast.Pow: operator.pow,  # Allow exponentiation.
}  # Finish the safe operator map.


def calculate_math(expression: str) -> int | float:  # Safely calculate a simple expression.
    tree = ast.parse(expression, mode="eval")  # Parse the expression without executing arbitrary code.

    def evaluate(node: ast.AST) -> int | float:  # Recursively evaluate allowed syntax-tree nodes.
        if isinstance(node, ast.Expression):  # Handle the expression root node.
            return evaluate(node.body)  # Evaluate the root's body.
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):  # Accept numeric constants.
            return node.value  # Return the numeric value.
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):  # Handle unary signs.
            value = evaluate(node.operand)  # Evaluate the signed value.
            return value if isinstance(node.op, ast.UAdd) else -value  # Apply the sign.
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:  # Handle allowed binary operations.
            left = evaluate(node.left)  # Evaluate the left operand.
            right = evaluate(node.right)  # Evaluate the right operand.
            return ALLOWED_OPERATORS[type(node.op)](left, right)  # Apply the selected operator.
        raise ValueError("Only simple mathematical expressions are supported")  # Reject unsafe syntax.

    result = evaluate(tree)  # Evaluate the complete expression.
    if abs(result) > 10**12:  # Prevent unreasonably large results.
        raise ValueError("The mathematical result is outside the supported range")  # Report the limit.
    return result  # Return the safe calculation result.


def extract_math_expression(question: str) -> str | None:  # Detect math questions that can bypass RAG.
    cleaned = question.strip().replace("?", "")  # Remove surrounding spaces and question marks.
    for prefix in ("what is ", "calculate ", "compute "):  # Check common question prefixes.
        if cleaned.lower().startswith(prefix):  # Detect a prefix at the beginning.
            cleaned = cleaned[len(prefix):].strip()  # Remove the prefix.
            break  # Stop after finding one prefix.
    allowed = set("0123456789+-*/(). ")  # Define allowed math characters.
    return cleaned if cleaned and set(cleaned) <= allowed else None  # Return only valid expressions.


class LangChainRAG:  # Encapsulate embeddings, retrieval, prompts, and generation.
    def __init__(self) -> None:  # Initialize the LangChain RAG components.
        if not DEEPSEEK_API_KEY:  # Check that the API credential exists.
            raise ValueError("Configure DEEPSEEK_API_KEY in the .env file first")  # Explain the missing setting.
        self.embeddings = HuggingFaceEmbeddings(  # Create the LangChain embedding adapter.
            model_name=EMBEDDING_MODEL,  # Select the English embedding model.
            encode_kwargs={"normalize_embeddings": True},  # Normalize vectors for consistent distance scores.
        )  # Finish the embedding adapter.
        self.vectorstore = Chroma(  # Connect to the persistent vector store.
            collection_name=COLLECTION_NAME,  # Use this version's collection.
            persist_directory=str(CHROMA_DIR),  # Keep the LangChain index separate.
            embedding_function=self.embeddings,  # Use the embedding adapter for queries.
        )  # Finish the vector-store connection.
        self.llm = ChatOpenAI(  # Configure the DeepSeek chat model.
            model=DEEPSEEK_MODEL,  # Select the configured model.
            api_key=DEEPSEEK_API_KEY,  # Pass the API credential.
            base_url="https://api.deepseek.com",  # Use DeepSeek's OpenAI-compatible endpoint.
            temperature=0.1,  # Keep answers relatively deterministic.
        )  # Finish the chat model configuration.
        self.prompt = ChatPromptTemplate.from_messages(  # Build the answer prompt template.
            [  # Define the prompt message sequence.
                ("system", SYSTEM_PROMPT),  # Add grounding instructions.
                MessagesPlaceholder("chat_history"),  # Insert recent conversation messages.
                (  # Add the current question and retrieved context.
                    "human",  # Mark this as the current user message.
                    "Long-term memory:\n{summary}\n\nReference material:\n{context}\n\nCurrent question: {question}",  # Define dynamic prompt fields.
                ),  # Finish the human message template.
            ]  # Finish the prompt sequence.
        )  # Store the reusable prompt.

    def summarize_history(self, messages: list[BaseMessage], existing_summary: str = "") -> str:  # Compress older turns.
        summary_prompt = ChatPromptTemplate.from_messages(  # Build the summary prompt.
            [  # Define summary instructions and input.
                ("system", SUMMARY_PROMPT),  # Add summary rules.
                ("human", "Existing summary:\n{summary}\n\nNew conversation:\n{history}"),  # Add summary inputs.
            ]  # Finish summary messages.
        )  # Store the summary prompt.
        history_text = "\n".join(f"{message.type}: {message.content}" for message in messages)  # Serialize old messages.
        response = self.llm.invoke(  # Ask the model to create a compact memory.
            summary_prompt.invoke({"summary": existing_summary, "history": history_text})  # Fill the prompt fields.
        )  # Complete the summary request.
        return response.content or existing_summary  # Keep the old summary if the response is empty.

    def answer(  # Retrieve references and generate an answer.
        self,  # Use this initialized RAG object.
        question: str,  # Receive the current user question.
        history: list[BaseMessage] | None = None,  # Receive recent raw conversation messages.
        summary: str = "",  # Receive the summary of older conversation.
    ) -> tuple[str, list[dict]]:  # Return the answer and displayed references.
        math_expression = extract_math_expression(question)  # Check for a simple math fast path.
        if math_expression:  # Handle simple math without document retrieval.
            try:  # Attempt safe local calculation.
                return f"{math_expression} = {calculate_math(math_expression):g}", []  # Return the calculation directly.
            except (SyntaxError, ValueError, ZeroDivisionError):  # Fall back to normal RAG on invalid math.
                pass  # Continue with retrieval and generation.
        history = history or []  # Use an empty history when none was provided.
        recent_history = history[-6:]  # Keep the latest three conversation turns raw.
        history_text = "\n".join(f"{message.type}: {message.content}" for message in recent_history)  # Serialize recent history.
        retrieval_query = f"Long-term memory:\n{summary}\n\nConversation history:\n{history_text}\n\nCurrent question:\n{question}"  # Build the retrieval query.
        matches = self.vectorstore.similarity_search_with_score(retrieval_query, k=4)  # Retrieve the four closest chunks.
        relevant = [(document, score) for document, score in matches if score <= RETRIEVAL_DISTANCE_THRESHOLD]  # Reject weak matches.
        context = "\n\n".join(  # Format relevant documents for the model.
            f"[Source: {document.metadata.get('source', 'unknown')}]\n{document.page_content}"  # Include source and text.
            for document, _ in relevant  # Iterate through relevant chunks.
        )  # Finish context formatting.
        references = [  # Build source data for the Streamlit UI.
            {  # Create one reference record.
                "source": document.metadata.get("source", "unknown"),  # Store the source filename.
                "chunk": document.metadata.get("chunk", document.metadata.get("page", "unknown")),  # Store chunk or page number.
                "text": document.page_content,  # Store the displayed source text.
                "distance": score,  # Store the retrieval distance.
            }  # Finish the reference record.
            for document, score in relevant  # Iterate through relevant chunks.
        ]  # Finish the reference list.
        messages = self.prompt.invoke(  # Fill the answer prompt with current state.
            {  # Provide every dynamic prompt field.
                "chat_history": recent_history,  # Add recent raw messages.
                "summary": summary,  # Add older conversation memory.
                "context": context,  # Add retrieved documents.
                "question": question,  # Add the current question.
            }  # Finish prompt inputs.
        )  # Create the model-ready message list.
        response = self.llm.invoke(messages)  # Generate the final answer.
        return response.content, references  # Return answer text and evidence.
