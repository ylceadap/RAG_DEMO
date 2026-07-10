import os  # Read configuration values from environment variables.
from pathlib import Path  # Build paths that work across operating systems.

from dotenv import load_dotenv  # Load variables from .env files.

BASE_DIR = Path(__file__).parent  # Locate the LangChain version directory.
DATA_DIR = BASE_DIR / "data" / "documents"  # Store source documents here.
CHROMA_DIR = BASE_DIR / "chroma_db"  # Store this version's vector database here.
COLLECTION_NAME = "company_documents_langchain"  # Name the Chroma collection.

load_dotenv(BASE_DIR / ".env")  # Prefer a version-specific environment file.
load_dotenv(BASE_DIR.parent / ".env")  # Fall back to the repository-level environment file.

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")  # Read the DeepSeek API key.
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # Select the chat model.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")  # Read the optional app password.
RETRIEVAL_DISTANCE_THRESHOLD = float(os.getenv("RETRIEVAL_DISTANCE_THRESHOLD", "0.75"))  # Reject weak retrieval matches.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Use an English embedding model for the English corpus.
