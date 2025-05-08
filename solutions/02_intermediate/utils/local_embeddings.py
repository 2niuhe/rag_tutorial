"""
本地嵌入模型

使用sentence-transformers库实现本地嵌入模型
"""

from typing import List, Optional, Any
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

class LocalHuggingFaceEmbeddings(Embeddings):
    """使用HuggingFace的sentence-transformers实现的本地嵌入模型"""
    
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        cache_folder: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        encode_kwargs: Optional[dict] = None,
    ):
        """
        初始化本地嵌入模型
        
        参数:
            model_name: 模型名称，默认使用支持多语言的模型
            cache_folder: 模型缓存目录
            model_kwargs: 传递给SentenceTransformer的额外参数
            encode_kwargs: 传递给encode方法的额外参数
        """
        model_kwargs = model_kwargs or {}
        encode_kwargs = encode_kwargs or {}
        
        if cache_folder:
            model_kwargs["cache_folder"] = cache_folder
            
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs
        
        # 加载模型
        self.model = SentenceTransformer(model_name, **model_kwargs)
        
        print(f"已加载本地嵌入模型: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入文档列表
        
        参数:
            texts: 文档文本列表
            
        返回:
            嵌入向量列表
        """
        embeddings = self.model.encode(texts, **self.encode_kwargs)
        # 确保返回的是Python列表而不是numpy数组
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入查询文本
        
        参数:
            text: 查询文本
            
        返回:
            嵌入向量
        """
        embedding = self.model.encode(text, **self.encode_kwargs)
        # 确保返回的是Python列表而不是numpy数组
        return embedding.tolist()
