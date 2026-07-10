import os  # 导入操作系统相关功能，用于读取环境变量。
from pathlib import Path  # 导入路径工具，用于安全地处理文件路径。

from dotenv import load_dotenv  # 导入 .env 文件加载函数。

BASE_DIR = Path(__file__).parent  # 获取当前项目的根目录。
DATA_DIR = BASE_DIR / "data" / "documents"  # 设置原始文档所在目录。
CHROMA_DIR = BASE_DIR / "chroma_db"  # 设置 ChromaDB 向量数据库目录。
COLLECTION_NAME = "company_documents"  # 设置向量集合的名称。

load_dotenv(BASE_DIR / ".env")  # 从项目根目录加载 API 配置。

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")  # 读取 DeepSeek API Key。
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # 读取模型名，没有配置时使用默认模型。
