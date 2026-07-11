import streamlit as st

from config import APP_PASSWORD, DATA_DIR, DEEPSEEK_API_KEY
from ingest import build_index
from rag import RAG

RECENT_MESSAGE_LIMIT = 6

st.set_page_config(page_title="Company Knowledge Assistant", page_icon="📚")


@st.cache_resource
def get_rag() -> RAG:
    """Create one shared RAG service for the Streamlit process."""
    return RAG()


def initialize_session_state() -> None:
    """Create the conversation state used by this browser session."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("conversation_summary", "")
    st.session_state.setdefault("summarized_count", 0)


def clear_conversation() -> None:
    """Reset the visible history and its long-term summary."""
    st.session_state.messages = []
    st.session_state.conversation_summary = ""
    st.session_state.summarized_count = 0


def require_password() -> None:
    """Stop rendering when a configured application password is missing."""
    if not APP_PASSWORD or st.session_state.get("authenticated"):
        return

    password = st.text_input("Application password", type="password")
    if password == APP_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()

    st.info("Enter the application password to continue.")
    st.stop()


def render_sidebar() -> None:
    """Render conversation and indexing controls."""
    with st.sidebar:
        st.subheader("Instructions")
        st.write(f"Place .txt, .md, or .pdf files in:\n`{DATA_DIR}`")

        if st.button("Clear conversation"):
            clear_conversation()
            st.rerun()

        if st.button("Sync document index"):
            with st.spinner("Checking documents and updating the index. The embedding model may download on first run..."):
                files, chunks = build_index()
            st.success(f"Updated {files} files and added {chunks} text chunks")

        if not DEEPSEEK_API_KEY:
            st.warning("DEEPSEEK_API_KEY is not configured")


def render_sources(sources: list[dict]) -> None:
    """Display the retrieved chunks used to answer a question."""
    if not sources:
        return

    source_names = sorted({source["source"] for source in sources})
    st.caption("Sources: " + ", ".join(source_names))
    with st.expander("View reference material"):
        for number, source in enumerate(sources, start=1):
            st.markdown(
                f"**Reference {number}: {source['source']} · chunk {source['chunk']}**  "
                f"\nDistance: `{source['distance']:.4f}`"
            )
            st.code(source["text"], language="text")


def update_conversation_summary(rag: RAG, history: list[dict]) -> None:
    """Summarize older messages while retaining recent messages as raw context."""
    first_unsummarized = st.session_state.summarized_count
    summary_end = len(history) - RECENT_MESSAGE_LIMIT
    if summary_end <= first_unsummarized:
        return

    messages_to_summarize = history[first_unsummarized:summary_end]
    st.session_state.conversation_summary = rag.summarize_history(
        messages_to_summarize,
        st.session_state.conversation_summary,
    )
    st.session_state.summarized_count = summary_end


def answer_question(question: str) -> None:
    """Show and save one RAG answer for the submitted question."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            rag = get_rag()
            history = st.session_state.messages[:-1]
            update_conversation_summary(rag, history)
            answer, sources = rag.answer(
                question,
                history=history[-RECENT_MESSAGE_LIMIT:],
                summary=st.session_state.conversation_summary,
            )
            st.markdown(answer)
            render_sources(sources)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as exc:
            st.error(str(exc))


def render_chat_history() -> None:
    """Replay messages stored for the current browser session."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


initialize_session_state()
require_password()
render_sidebar()

st.title("📚 Company Knowledge Assistant")
st.caption("Local document retrieval powered by the DeepSeek API")
render_chat_history()

if question := st.chat_input("For example: What are the onboarding steps for a new employee?"):
    answer_question(question)
