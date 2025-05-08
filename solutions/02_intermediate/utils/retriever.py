"""
检索器模块

提供高级检索策略，包括混合检索和重排序
"""

from typing import List, Dict, Any, Optional, Union
from langchain.schema import Document
from langchain.vectorstores import VectorStore, Chroma, FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# 导入本地嵌入模型
from utils.local_embeddings import LocalHuggingFaceEmbeddings
from config import LOCAL_EMBEDDING_MODEL


class AdvancedRetriever:
    """高级检索器类，提供多种检索策略"""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        初始化高级检索器
        
        参数:
            vector_store: 向量存储对象
            search_type: 搜索类型，'similarity'或'mmr'
            search_kwargs: 搜索参数
        """
        self.vector_store = vector_store
        self.search_type = search_type
        
        # 默认搜索参数
        if search_kwargs is None:
            if search_type == "similarity" or search_type == "mmr":
                self.search_kwargs = {"k": 5}
            else:
                self.search_kwargs = {"k": 5}
        else:
            # 移除fetch_k参数，因为它在新版本中不再被支持
            if "fetch_k" in search_kwargs:
                search_kwargs.pop("fetch_k")
            self.search_kwargs = search_kwargs
    
    def create_vector_store(
        self,
        documents: List[Document],
        embedding_model: str = LOCAL_EMBEDDING_MODEL,
        vector_store_type: str = "chroma",
        persist_directory: Optional[str] = None,
        collection_name: str = "intermediate_rag_collection"
    ) -> VectorStore:
        """
        创建向量存储
        
        参数:
            documents: 文档列表
            embedding_model: 嵌入模型名称
            vector_store_type: 向量存储类型，'chroma'或'faiss'
            persist_directory: 持久化目录
            collection_name: 集合名称
            
        返回:
            向量存储对象
        """
        # 使用本地嵌入模型
        print(f"使用本地嵌入模型: {embedding_model}")
        embeddings = LocalHuggingFaceEmbeddings(model_name=embedding_model)
        
        # 创建向量存储
        if vector_store_type.lower() == "chroma":
            if persist_directory:
                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=persist_directory,
                    collection_name=collection_name
                )
            else:
                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    collection_name=collection_name
                )
        elif vector_store_type.lower() == "faiss":
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=embeddings
            )
            # 如果指定了持久化目录，保存FAISS索引
            if persist_directory:
                import os
                os.makedirs(persist_directory, exist_ok=True)
                vector_store.save_local(persist_directory)
        else:
            raise ValueError(f"不支持的向量存储类型: {vector_store_type}")
        
        self.vector_store = vector_store
        return vector_store
    
    def load_vector_store(
        self,
        embedding_model: str = LOCAL_EMBEDDING_MODEL,
        vector_store_type: str = "chroma",
        persist_directory: str = "./chroma_db",
        collection_name: str = "intermediate_rag_collection"
    ) -> VectorStore:
        """
        加载现有向量存储
        
        参数:
            embedding_model: 嵌入模型名称
            vector_store_type: 向量存储类型，'chroma'或'faiss'
            persist_directory: 持久化目录
            collection_name: 集合名称
            
        返回:
            向量存储对象
        """
        # 使用本地嵌入模型
        print(f"使用本地嵌入模型: {embedding_model}")
        embeddings = LocalHuggingFaceEmbeddings(model_name=embedding_model)
        
        # 加载向量存储
        if vector_store_type.lower() == "chroma":
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name
            )
        elif vector_store_type.lower() == "faiss":
            vector_store = FAISS.load_local(
                persist_directory,
                embeddings
            )
        else:
            raise ValueError(f"不支持的向量存储类型: {vector_store_type}")
        
        self.vector_store = vector_store
        return vector_store
    
    def get_vector_retriever(
        self,
        filter_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        获取向量检索器
        
        参数:
            filter_metadata: 元数据过滤条件
            
        返回:
            向量检索器
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化，请先创建或加载向量存储")
        
        # 如果有过滤条件，添加到搜索参数中
        search_kwargs = self.search_kwargs.copy()
        if filter_metadata:
            search_kwargs["filter"] = filter_metadata
        
        # 创建检索器
        retriever = self.vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs=search_kwargs
        )
        
        return retriever
    
    # 移除get_compression_retriever方法，因为它依赖于OpenAI LLM
    
    def get_hybrid_retriever(
        self,
        documents: List[Document],
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        """
        获取混合检索器（向量 + BM25）
        
        参数:
            documents: 文档列表
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重
            
        返回:
            混合检索器
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化，请先创建或加载向量存储")
        
        # 创建向量检索器
        vector_retriever = self.get_vector_retriever()
        
        # 创建BM25检索器
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = self.search_kwargs.get("k", 5)
        
        # 创建集成检索器
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[vector_weight, bm25_weight]
        )
        
        return ensemble_retriever
    
    def retrieve(
        self,
        query: str,
        retriever_type: str = "vector",
        filter_metadata: Optional[Dict[str, Any]] = None,
        documents: Optional[List[Document]] = None
    ) -> List[Document]:
        """
        执行检索
        
        参数:
            query: 查询文本
            retriever_type: 检索器类型，'vector', 'compression', 或 'hybrid'
            filter_metadata: 元数据过滤条件
            documents: 用于BM25检索的文档列表（仅hybrid模式需要）
            
        返回:
            检索到的文档列表
        """
        # 根据类型选择检索器
        if retriever_type == "vector":
            retriever = self.get_vector_retriever(filter_metadata)
        elif retriever_type == "hybrid":
            if documents is None:
                raise ValueError("混合检索模式需要提供documents参数")
            retriever = self.get_hybrid_retriever(documents)
        else:
            # 默认使用向量检索器
            print(f"不支持的检索器类型: {retriever_type}，使用向量检索器代替")
            retriever = self.get_vector_retriever(filter_metadata)
        
        # 执行检索
        retrieved_docs = retriever.get_relevant_documents(query)
        
        return retrieved_docs
