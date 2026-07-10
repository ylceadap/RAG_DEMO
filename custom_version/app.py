import streamlit as st  # 导入 Streamlit，用于创建网页界面。

from config import APP_PASSWORD, DATA_DIR, DEEPSEEK_API_KEY  # Import the document directory and authentication settings.
from ingest import build_index  # 导入重新建立文档索引的函数。
from rag import RAG  # 导入 RAG 问答类。


@st.cache_resource  # Cache heavyweight resources for the lifetime of the Streamlit process.
def get_rag() -> RAG:  # Create one shared RAG instance instead of loading the model per question.
    return RAG()  # Load the embedding model and vector-store connection once.


def require_password() -> None:  # Protect the app when an application password is configured.
    if not APP_PASSWORD or st.session_state.get("authenticated"):  # Allow local development without a password.
        return  # Continue to the application.
    password = st.text_input("Application password", type="password")  # Ask for the configured password.
    if password == APP_PASSWORD:  # Check the submitted password.
        st.session_state.authenticated = True  # Mark this browser session as authenticated.
        st.rerun()  # Reload the page after successful authentication.
    st.info("Enter the application password to continue.")  # Explain why the app is waiting.
    st.stop()  # Do not render the knowledge assistant before authentication.


require_password()  # Apply the optional password gate.

st.set_page_config(page_title="Company Knowledge Assistant", page_icon="📚")  # Set the page title and icon.
st.title("📚 Company Knowledge Assistant")  # Display the main page title.
st.caption("Local document retrieval powered by the DeepSeek API")  # Display the page description.

with st.sidebar:  # 创建左侧边栏区域。
    st.subheader("Instructions")  # Display the sidebar heading.
    st.write(f"Place .txt, .md, or .pdf files in:\n`{DATA_DIR}`")  # Show the document directory.
    if st.button("Clear conversation"):  # Create the clear-history button.
        st.session_state.messages = []  # Delete the current chat history.
        st.session_state.conversation_summary = ""  # Delete the conversation summary.
        st.session_state.summarized_count = 0  # Reset the summarized-message counter.
        st.rerun()  # Refresh the page with an empty conversation.
    if st.button("Sync document index"):  # Create the incremental-index button.
        with st.spinner("Checking documents and updating the index. The embedding model may download on first run..."):  # Show progress text.
            files, chunks = build_index()  # Add new files and update changed files only.
        st.success(f"Updated {files} files and added {chunks} text chunks")  # Show the indexing result.
    if not DEEPSEEK_API_KEY:  # Check whether the API key is missing.
        st.warning("DEEPSEEK_API_KEY is not configured")  # Display a configuration warning.

if "messages" not in st.session_state:  # 判断当前会话是否已有聊天记录。
    st.session_state.messages = []  # 初始化聊天记录列表。
if "conversation_summary" not in st.session_state:  # 判断当前会话是否已有长期记忆摘要。
    st.session_state.conversation_summary = ""  # 初始化长期记忆摘要。
if "summarized_count" not in st.session_state:  # 判断是否记录过已摘要的消息数量。
    st.session_state.summarized_count = 0  # 初始化已摘要消息计数。

for message in st.session_state.messages:  # 遍历历史聊天记录。
    with st.chat_message(message["role"]):  # 按用户或助手身份显示消息。
        st.markdown(message["content"])  # 显示消息内容。

question = st.chat_input("For example: What are the onboarding steps for a new employee?")  # Show the user input box.
if question:  # Continue only after the user submits a question.
    st.session_state.messages.append({"role": "user", "content": question})  # Save the user question.
    with st.chat_message("user"):  # Create the user message bubble.
        st.markdown(question)  # Display the user question.
    with st.chat_message("assistant"):  # Create the assistant message bubble.
        try:  # Try to run retrieval and answer generation.
            rag = get_rag()  # Reuse the cached RAG object for this turn.
            history = st.session_state.messages[:-1]  # Get all messages before the current question.
            raw_history_start = st.session_state.summarized_count  # Get the first message not yet summarized.
            if len(history) - raw_history_start > 6:  # Check whether older messages need summarization.
                summary_end = len(history) - 6  # Keep the latest six messages as raw context.
                messages_to_summarize = history[raw_history_start:summary_end]  # Select messages to compress.
                st.session_state.conversation_summary = rag.summarize_history(  # Update the long-term summary.
                    messages_to_summarize,  # Pass the newly eligible older messages.
                    st.session_state.conversation_summary,  # Pass the existing summary as context.
                )  # Finish updating the summary.
                st.session_state.summarized_count = summary_end  # Record the summarized-message boundary.
            answer, sources = rag.answer(  # Retrieve documents and generate a context-aware answer.
                question,  # Pass the current user question.
                history=history[-6:],  # Pass the latest three raw conversation turns.
                summary=st.session_state.conversation_summary,  # Pass the summary of older messages.
            )  # Complete the conversational RAG call.
            st.markdown(answer)  # Display the assistant answer.
            if sources:  # Check whether sources were retrieved.
                st.caption("Sources: " + ", ".join(sorted({s["source"] for s in sources})))  # Display source filenames.
                with st.expander("View reference material"):  # Create an expandable references section.
                    for number, source in enumerate(sources, start=1):  # Iterate over retrieved sources.
                        st.markdown(  # Display the source and retrieval distance.
                            f"**Reference {number}: {source['source']} · chunk {source['chunk']}**  "
                            f"\nDistance: `{source['distance']:.4f}`"
                        )  # Finish displaying source metadata.
                        st.code(source["text"], language="text")  # Display the source text used by the model.
            st.session_state.messages.append({"role": "assistant", "content": answer})  # Save the assistant answer.
        except Exception as exc:  # Catch errors during processing.
            st.error(str(exc))  # Display the error in the web app.
