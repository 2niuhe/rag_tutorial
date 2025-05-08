"""
配置文件

存储RAG系统的配置参数
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 本地模型配置
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# 本地LLM配置（如果需要）
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "")

# 向量数据库配置
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chroma")  # 'chroma' 或 'faiss'
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "intermediate_rag_collection")

# 文档处理配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# 检索配置
RETRIEVER_TYPE = os.getenv("RETRIEVER_TYPE", "vector")  # 'vector' 或 'hybrid'
TOP_K = int(os.getenv("TOP_K", "5"))
SEARCH_TYPE = os.getenv("SEARCH_TYPE", "similarity")  # 'similarity' 或 'mmr'

# 语言模型配置
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
STREAMING = os.getenv("STREAMING", "True").lower() == "true"

# 语料库路径
CORPUS_PATH = os.getenv("CORPUS_PATH", "../../corpus")
