import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from config import APP_PASSWORD, DATA_DIR, DEEPSEEK_API_KEY
from ingest import build_index
from rag import LangChainRAG

st.set_page_config(page_title="LangChain Knowledge Assistant", page_icon="📚")


@st.cache_resource
def get_rag() -> LangChainRAG:
    return LangChainRAG()


def require_password() -> None:
    if not APP_PASSWORD or st.session_state.get("authenticated"):
        return
    password = st.text_input("Application password", type="password")
    if password == APP_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    st.info("Enter the application password to continue.")
    st.stop()


require_password()
st.title("📚 LangChain Knowledge Assistant")
st.caption("LangChain loaders, embeddings, Chroma retrieval, and DeepSeek generation")

with st.sidebar:
    st.subheader("Instructions")
    st.write(f"Place .txt, .md, or .pdf files in:\n`{DATA_DIR}`")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.conversation_summary = ""
        st.session_state.summarized_count = 0
        st.rerun()
    if st.button("Sync document index"):
        with st.spinner("Checking documents and updating the LangChain index..."):
            files, chunks = build_index()
        st.success(f"Updated {files} files and added {chunks} chunks")
    if st.button("Full rebuild index"):
        with st.spinner("Rebuilding the complete LangChain index..."):
            files, chunks = build_index(reset=True)
        st.success(f"Indexed {files} files and created {chunks} chunks")
    if not DEEPSEEK_API_KEY:
        st.warning("DEEPSEEK_API_KEY is not configured")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""
if "summarized_count" not in st.session_state:
    st.session_state.summarized_count = 0

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the company documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            history = [
                HumanMessage(content=message["content"]) if message["role"] == "user" else AIMessage(content=message["content"])
                for message in st.session_state.messages[:-1]
            ]
            rag = get_rag()
            raw_history_start = st.session_state.summarized_count
            if len(history) - raw_history_start > 6:
                summary_end = len(history) - 6
                old_messages = history[raw_history_start:summary_end]
                st.session_state.conversation_summary = rag.summarize_history(
                    old_messages,
                    st.session_state.conversation_summary,
                )
                st.session_state.summarized_count = summary_end
            answer, sources = rag.answer(
                question,
                history=history[-6:],
                summary=st.session_state.conversation_summary,
            )
            st.markdown(answer)
            if sources:
                st.caption("Sources: " + ", ".join(sorted({source["source"] for source in sources})))
                with st.expander("View reference material"):
                    for number, source in enumerate(sources, start=1):
                        st.markdown(f"**Reference {number}: {source['source']}**  \nDistance: `{source['distance']:.4f}`")
                        st.code(source["text"], language="text")
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as exc:
            st.error(str(exc))
