"""
嵌入生成工具模块

这个模块提供了使用Sentence-Transformers生成文本嵌入的功能。
"""

from typing import List, Dict, Union, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingGenerator:
    """文本嵌入生成器类"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化嵌入生成器
        
        参数:
            model_name: Sentence-Transformers模型名称
        """
        print(f"加载嵌入模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name  # 添加model_name属性
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"嵌入维度: {self.embedding_dim}")
    
    def generate_embeddings(self, 
                           chunks: List[Dict[str, str]], 
                           batch_size: int = 32,
                           show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        为文本块生成嵌入向量
        
        参数:
            chunks: 文本块字典列表，每个字典包含'content'和'metadata'
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            
        返回:
            包含文本、嵌入和元数据的字典列表
        """
        # 提取文本内容
        texts = [chunk['content'] for chunk in chunks]
        
        # 生成嵌入
        embeddings = []
        
        # 使用批处理提高效率
        for i in tqdm(range(0, len(texts), batch_size), 
                     desc="生成嵌入", 
                     disable=not show_progress):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts)
            embeddings.extend(batch_embeddings)
        
        # 组合文本、嵌入和元数据
        result = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            result.append({
                'content': chunk['content'],
                'embedding': embedding,
                'metadata': chunk['metadata']
            })
        
        return result
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        为查询文本生成嵌入向量
        
        参数:
            query: 查询文本
            
        返回:
            查询的嵌入向量
        """
        return self.model.encode(query)
