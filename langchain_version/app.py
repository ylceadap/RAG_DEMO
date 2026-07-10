import streamlit as st  # Build the interactive web interface.
from langchain_core.messages import AIMessage, HumanMessage  # Convert session messages to LangChain messages.

from config import APP_PASSWORD, DATA_DIR, DEEPSEEK_API_KEY  # Import UI and environment configuration.
from ingest import build_index  # Import the incremental indexing function.
from rag import LangChainRAG  # Import the LangChain RAG service.

st.set_page_config(page_title="LangChain Knowledge Assistant", page_icon="📚")  # Configure the browser page.


@st.cache_resource  # Cache heavyweight resources across Streamlit reruns.
def get_rag() -> LangChainRAG:  # Create the shared LangChain RAG service.
    return LangChainRAG()  # Load embeddings, Chroma, and the chat model once.


def require_password() -> None:  # Gate the app when a password is configured.
    if not APP_PASSWORD or st.session_state.get("authenticated"):  # Allow local development without a password.
        return  # Continue rendering the application.
    password = st.text_input("Application password", type="password")  # Ask the user for the password.
    if password == APP_PASSWORD:  # Check the submitted password.
        st.session_state.authenticated = True  # Mark this browser session as authenticated.
        st.rerun()  # Reload the page after successful authentication.
    st.info("Enter the application password to continue.")  # Explain why the app is waiting.
    st.stop()  # Stop rendering protected content.


require_password()  # Apply the optional password gate.
st.title("📚 LangChain Knowledge Assistant")  # Display the application title.
st.caption("LangChain loaders, embeddings, Chroma retrieval, and DeepSeek generation")  # Explain the stack.

with st.sidebar:  # Build the controls in the sidebar.
    st.subheader("Instructions")  # Display the sidebar heading.
    st.write(f"Place .txt, .md, or .pdf files in:\n`{DATA_DIR}`")  # Show the document directory.
    if st.button("Clear conversation"):  # Create the conversation reset button.
        st.session_state.messages = []  # Delete visible chat messages.
        st.session_state.conversation_summary = ""  # Delete the long-term summary.
        st.session_state.summarized_count = 0  # Reset the summary boundary.
        st.rerun()  # Refresh the page with an empty session.
    if st.button("Sync document index"):  # Create the incremental indexing button.
        with st.spinner("Checking documents and updating the LangChain index..."):  # Show indexing progress.
            files, chunks = build_index()  # Add, update, skip, or remove document vectors.
        st.success(f"Updated {files} files and added {chunks} chunks")  # Display indexing statistics.
    if st.button("Full rebuild index"):  # Create the full rebuild button.
        with st.spinner("Rebuilding the complete LangChain index..."):  # Show full rebuild progress.
            files, chunks = build_index(reset=True)  # Delete and recreate the entire index.
        st.success(f"Indexed {files} files and created {chunks} chunks")  # Display rebuild statistics.
    if not DEEPSEEK_API_KEY:  # Check whether the API key is configured.
        st.warning("DEEPSEEK_API_KEY is not configured")  # Display a configuration warning.

if "messages" not in st.session_state:  # Initialize visible conversation state.
    st.session_state.messages = []  # Store user and assistant messages.
if "conversation_summary" not in st.session_state:  # Initialize long-term conversation memory.
    st.session_state.conversation_summary = ""  # Store the summary of older turns.
if "summarized_count" not in st.session_state:  # Initialize the summary boundary.
    st.session_state.summarized_count = 0  # Track how many messages were summarized.

for message in st.session_state.messages:  # Replay the conversation on every rerun.
    with st.chat_message(message["role"]):  # Select the user or assistant message style.
        st.markdown(message["content"])  # Render the message content.

question = st.chat_input("Ask a question about the company documents")  # Render the user input box.
if question:  # Process only when the user submits a question.
    st.session_state.messages.append({"role": "user", "content": question})  # Save the current question.
    with st.chat_message("user"):  # Render the user message bubble.
        st.markdown(question)  # Display the user question.
    with st.chat_message("assistant"):  # Render the assistant response bubble.
        try:  # Protect the UI from runtime errors.
            history = [  # Convert stored dictionaries into LangChain message objects.
                HumanMessage(content=message["content"]) if message["role"] == "user" else AIMessage(content=message["content"])  # Map each role.
                for message in st.session_state.messages[:-1]  # Exclude the current question from prior history.
            ]  # Finish converting chat history.
            rag = get_rag()  # Reuse the cached RAG service.
            raw_history_start = st.session_state.summarized_count  # Find the first unsummarized message.
            if len(history) - raw_history_start > 6:  # Summarize older messages after three turns.
                summary_end = len(history) - 6  # Keep the latest six messages raw.
                old_messages = history[raw_history_start:summary_end]  # Select messages to summarize.
                st.session_state.conversation_summary = rag.summarize_history(  # Generate the new summary.
                    old_messages,  # Pass newly eligible older messages.
                    st.session_state.conversation_summary,  # Preserve the previous summary.
                )  # Finish summary generation.
                st.session_state.summarized_count = summary_end  # Save the new summary boundary.
            answer, sources = rag.answer(  # Retrieve context and generate an answer.
                question,  # Pass the current question.
                history=history[-6:],  # Pass the latest three raw turns.
                summary=st.session_state.conversation_summary,  # Pass older conversation memory.
            )  # Finish the RAG call.
            st.markdown(answer)  # Display the answer text.
            if sources:  # Check whether relevant references were found.
                st.caption("Sources: " + ", ".join(sorted({source["source"] for source in sources})))  # Display source names.
                with st.expander("View reference material"):  # Create an expandable evidence section.
                    for number, source in enumerate(sources, start=1):  # Iterate through source chunks.
                        st.markdown(f"**Reference {number}: {source['source']}**  \nDistance: `{source['distance']:.4f}`")  # Show source metadata.
                        st.code(source["text"], language="text")  # Show the retrieved source text.
            st.session_state.messages.append({"role": "assistant", "content": answer})  # Save the assistant answer.
        except Exception as exc:  # Catch errors from indexing, retrieval, or the API.
            st.error(str(exc))  # Show the error to the user.
