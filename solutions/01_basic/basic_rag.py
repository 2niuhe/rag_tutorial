"""
基础RAG实现

这个脚本实现了一个简单的检索增强生成(RAG)系统，包括以下步骤：
1. 文档加载
2. 文本分块
3. 嵌入生成
4. 向量存储
5. 相似度检索
6. 回答生成

该实现不依赖于任何RAG框架，仅使用基本库来展示RAG的核心概念。
"""

import os
import numpy as np
import chromadb
from typing import List, Dict, Any
from chromadb.utils import embedding_functions

# 导入自定义工具模块
from utils.document_loader import load_documents
from utils.text_splitter import split_text
from utils.embeddings import EmbeddingGenerator


class SimpleRAG:
    """
    简单的检索增强生成(RAG)系统实现
    """
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 collection_name: str = "basic_rag_collection",
                 persist_directory: str = "./chroma_db"):
        """
        初始化RAG系统
        
        参数:
            embedding_model_name: 嵌入模型名称
            collection_name: ChromaDB集合名称
            persist_directory: ChromaDB持久化目录
        """
        print(f"初始化SimpleRAG系统...")
        
        # 初始化嵌入生成器
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model_name)
        
        # 初始化ChromaDB客户端
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # 获取或创建集合
        self.collection = self._get_or_create_collection(collection_name)
        
        print(f"SimpleRAG系统初始化完成。")
    
    def _get_or_create_collection(self, collection_name: str):
        """
        获取或创建ChromaDB集合
        
        参数:
            collection_name: 集合名称
            
        返回:
            ChromaDB集合对象
        """
        # 创建自定义嵌入函数
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_generator.model_name
        )
        
        # 先检查集合是否存在
        try:
            # 尝试获取现有集合
            collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=ef
            )
            print(f"已获取现有集合: {collection_name}")
            return collection
        except Exception:
            # 如果集合不存在，则创建新集合
            collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
            print(f"已创建新集合: {collection_name}")
            return collection
    
    def index_documents(self, 
                       documents_dir: str, 
                       chunk_size: int = 500, 
                       chunk_overlap: int = 50) -> None:
        """
        索引文档目录中的所有文本文件
        
        参数:
            documents_dir: 文档目录路径
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        print(f"开始索引文档目录: {documents_dir}")
        
        # 1. 加载文档
        documents = load_documents(documents_dir)
        print('开始分块')
        # 2. 文本分块
        chunks = split_text(
            documents, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        print('开始生成嵌入')
        # 3. 生成嵌入
        embedded_chunks = self.embedding_generator.generate_embeddings(chunks)
        
        # 4. 存储到ChromaDB
        # 准备批量添加数据
        ids = []
        documents_text = []
        embeddings = []
        metadatas = []
        
        for i, chunk in enumerate(embedded_chunks):
            # 使用唯一ID
            chunk_id = f"chunk_{i}_{hash(chunk['content'][:50])}"
            
            ids.append(chunk_id)
            documents_text.append(chunk['content'])
            embeddings.append(chunk['embedding'].tolist())
            metadatas.append(chunk['metadata'])
        
        # 添加到ChromaDB
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents_text,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
        print(f"索引完成，共添加 {len(ids)} 个文档块到向量数据库")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        检索与查询相关的文档
        
        参数:
            query: 查询文本
            top_k: 返回的最相关文档数量
            
        返回:
            相关文档列表
        """
        print(f"检索查询: '{query}'")
        
        # 生成查询嵌入
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        # 使用ChromaDB检索
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        print(f"找到 {len(formatted_results)} 个相关文档")
        return formatted_results
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        基于检索到的文档生成回答
        
        参数:
            query: 用户查询
            retrieved_docs: 检索到的文档列表
            
        返回:
            生成的回答
        """
        # 在基础版本中，我们简单地将检索到的文档拼接起来作为回答
        # 在更高级的实现中，这里会调用LLM生成更自然的回答
        
        if not retrieved_docs:
            return "抱歉，我没有找到相关信息。"
        
        # 构建回答
        answer = f"根据我找到的信息，关于'{query}'：\n\n"
        
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc['content']
            source = doc['metadata'].get('source', '未知来源')
            filename = os.path.basename(source)
            
            answer += f"{i}. {content}\n\n"
            answer += f"   来源: {filename}\n\n"
        
        answer += "以上是我找到的相关信息。"
        return answer
    
    def query(self, query: str, top_k: int = 3) -> str:
        """
        执行完整的RAG查询流程
        
        参数:
            query: 用户查询
            top_k: 返回的最相关文档数量
            
        返回:
            生成的回答
        """
        # 1. 检索相关文档
        retrieved_docs = self.retrieve(query, top_k=top_k)
        
        # 2. 生成回答
        answer = self.generate_answer(query, retrieved_docs)
        
        return answer


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='简单的RAG系统示例')
    parser.add_argument('--docs_dir', type=str, default='../../corpus',
                        help='文档目录路径')
    parser.add_argument('--chunk_size', type=int, default=500,
                        help='文本块大小')
    parser.add_argument('--chunk_overlap', type=int, default=50,
                        help='文本块重叠大小')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='嵌入模型名称')
    parser.add_argument('--index', action='store_true',
                        help='是否重新索引文档')
    
    args = parser.parse_args()
    
    # 初始化RAG系统
    rag = SimpleRAG(
        embedding_model_name=args.model,
        persist_directory="./chroma_db"
    )
    
    # 如果指定了索引，则重新索引文档
    if args.index:
        rag.index_documents(
            documents_dir=args.docs_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    
    # 简单的交互式查询循环
    print("\n=== 简单RAG系统演示 ===")
    print("输入问题进行查询，输入'exit'或'quit'退出")
    
    while True:
        query = input("\n请输入您的问题: ")
        if query.lower() in ['exit', 'quit', '退出']:
            break
        
        if not query.strip():
            continue
        
        # 执行查询
        answer = rag.query(query)
        print("\n回答:")
        print(answer)
