# DeepSeek RAG Demo

这是一个面向初学者的本地文档问答 Demo：文档和向量索引保存在本机，回答由 DeepSeek API 生成。

## 1. 安装依赖

建议使用 Python 3.10 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=你的_api_key
DEEPSEEK_MODEL=deepseek-chat
```

## 3. 建立文档索引

首次运行时会下载中文 Embedding 模型：

```bash
python ingest.py
```

## 4. 启动网页

```bash
streamlit run app.py
```

然后在浏览器打开 Streamlit 显示的地址。之后把自己的 `.txt`、`.md` 或 `.pdf` 文件放入 `data/documents/`，点击网页左侧的“重新导入文档”。

## 常见测试问题

- 新员工入职需要经过哪些步骤？
- 出差时每天最多可以报销多少餐费？
- 报销需要提交什么材料？
- 公司是否提供远程办公？

最后一个问题不在示例资料中，应该得到“资料中没有找到相关信息”的回答。

