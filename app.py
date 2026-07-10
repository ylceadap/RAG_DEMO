import streamlit as st  # 导入 Streamlit，用于创建网页界面。

from config import DATA_DIR, DEEPSEEK_API_KEY  # 导入文档目录和 API Key 配置。
from ingest import build_index  # 导入重新建立文档索引的函数。
from rag import RAG  # 导入 RAG 问答类。

st.set_page_config(page_title="公司资料问答助手", page_icon="📚")  # 设置网页标题和图标。
st.title("📚 公司资料问答助手")  # 显示网页主标题。
st.caption("基于本地文档检索 + DeepSeek API")  # 显示网页说明文字。

with st.sidebar:  # 创建左侧边栏区域。
    st.subheader("使用说明")  # 显示边栏小标题。
    st.write(f"把 .txt、.md 或 .pdf 文件放入：\n`{DATA_DIR}`")  # 告诉用户文档放置位置。
    if st.button("清空当前对话"):  # 创建清空聊天记录按钮。
        st.session_state.messages = []  # 删除当前网页会话中的所有聊天记录。
        st.session_state.conversation_summary = ""  # 删除当前会话的长期记忆摘要。
        st.session_state.summarized_count = 0  # 重置已摘要消息计数。
        st.rerun()  # 立即刷新页面，让界面显示为空白对话。
    if st.button("重新导入文档"):  # 创建重新导入文档按钮。
        with st.spinner("正在切分文档并建立索引，首次运行会下载 Embedding 模型..."):  # 显示处理中的提示。
            files, chunks = build_index()  # 读取文档并建立向量索引。
        st.success(f"已导入 {files} 个文件，生成 {chunks} 个文本片段")  # 显示索引完成结果。
    if not DEEPSEEK_API_KEY:  # 判断是否缺少 API Key。
        st.warning("尚未配置 DEEPSEEK_API_KEY")  # 在网页中显示配置警告。

if "messages" not in st.session_state:  # 判断当前会话是否已有聊天记录。
    st.session_state.messages = []  # 初始化聊天记录列表。
if "conversation_summary" not in st.session_state:  # 判断当前会话是否已有长期记忆摘要。
    st.session_state.conversation_summary = ""  # 初始化长期记忆摘要。
if "summarized_count" not in st.session_state:  # 判断是否记录过已摘要的消息数量。
    st.session_state.summarized_count = 0  # 初始化已摘要消息计数。

for message in st.session_state.messages:  # 遍历历史聊天记录。
    with st.chat_message(message["role"]):  # 按用户或助手身份显示消息。
        st.markdown(message["content"])  # 显示消息内容。

question = st.chat_input("例如：新员工入职需要经过哪些步骤？")  # 显示用户输入框。
if question:  # 只有用户输入问题后才继续处理。
    st.session_state.messages.append({"role": "user", "content": question})  # 保存用户问题。
    with st.chat_message("user"):  # 创建用户消息气泡。
        st.markdown(question)  # 显示用户问题。
    with st.chat_message("assistant"):  # 创建助手消息气泡。
        try:  # 尝试执行检索和问答。
            rag = RAG()  # 创建本轮问答使用的 RAG 对象。
            history = st.session_state.messages[:-1]  # 获取当前问题之前的全部历史消息。
            raw_history_start = st.session_state.summarized_count  # 获取尚未纳入摘要的消息起点。
            if len(history) - raw_history_start > 6:  # 判断是否有超过三轮的旧对话需要摘要。
                summary_end = len(history) - 6  # 保留最近六条消息作为原始上下文。
                messages_to_summarize = history[raw_history_start:summary_end]  # 取出需要压缩的旧消息。
                st.session_state.conversation_summary = rag.summarize_history(  # 更新长期记忆摘要。
                    messages_to_summarize,  # 传入本次新增的旧消息。
                    st.session_state.conversation_summary,  # 传入已有摘要作为基础。
                )  # 完成摘要更新。
                st.session_state.summarized_count = summary_end  # 记录已经摘要到哪条消息。
            answer, sources = rag.answer(  # 检索资料并生成结合摘要和历史的回答。
                question,  # 传入当前用户问题。
                history=history[-6:],  # 传入最近三轮原始对话。
                summary=st.session_state.conversation_summary,  # 传入更早对话的摘要。
            )  # 完成多轮问答调用。
            st.markdown(answer)  # 显示助手回答。
            if sources:  # 判断是否检索到了来源。
                st.caption("检索到的来源：" + "、".join(sorted({s["source"] for s in sources})))  # 显示来源文件名。
                with st.expander("查看参考资料"):  # 创建可展开的参考资料区域。
                    for number, source in enumerate(sources, start=1):  # 遍历所有检索结果。
                        st.markdown(  # 显示参考资料的来源和检索距离。
                            f"**参考片段 {number}：{source['source']} · 第 {source['chunk']} 个片段**  "
                            f"\n检索距离：`{source['distance']:.4f}`"
                        )  # 结束来源信息展示。
                        st.code(source["text"], language="text")  # 展示模型实际使用的原文。
            st.session_state.messages.append({"role": "assistant", "content": answer})  # 保存助手回答。
        except Exception as exc:  # 捕获运行过程中的异常。
            st.error(str(exc))  # 在网页中显示错误信息。
