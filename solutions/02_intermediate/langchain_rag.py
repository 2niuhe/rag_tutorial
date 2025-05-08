"""
中级RAG实现

使用LangChain框架构建的更健壮、功能更丰富的RAG系统，包括：
1. 高级文档处理和分块策略
2. 元数据过滤和混合检索
3. 提示工程和上下文增强
4. 流式响应
"""

import os
from typing import List, Dict, Any, Optional, Union
from langchain.schema import Document

# 导入配置和工具模块
from config import (
    VECTOR_DB_TYPE, CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_TYPE, TOP_K,
    SEARCH_TYPE, TEMPERATURE, MAX_TOKENS, STREAMING,
    CORPUS_PATH, LOCAL_EMBEDDING_MODEL, LOCAL_LLM_MODEL
)
from utils.document_processor import DocumentProcessor
from utils.retriever import AdvancedRetriever


class LangChainRAG:
    """使用LangChain框架的中级RAG实现"""
    
    def __init__(
        self,
        embedding_model: str = LOCAL_EMBEDDING_MODEL,
        vector_db_type: str = VECTOR_DB_TYPE,
        persist_directory: str = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = COLLECTION_NAME,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        """
        初始化LangChain RAG系统
        
        参数:
            embedding_model: 本地嵌入模型名称
            vector_db_type: 向量存储类型
            persist_directory: 持久化目录
            collection_name: 集合名称
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        print(f"初始化LangChain RAG系统...")
        
        # 初始化文档处理器
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # 初始化检索器
        self.retriever = AdvancedRetriever(
            search_type=SEARCH_TYPE,
            search_kwargs={"k": TOP_K}
        )
        
        # 保存配置
        self.embedding_model = embedding_model
        self.vector_db_type = vector_db_type
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # 没有LLM，只能进行检索
        self.llm = None
        
        print(f"LangChain RAG系统初始化完成。")
    
    # 移除_init_llm方法，因为我们不再使用OpenAI LLM
    
    def index_documents(
        self,
        documents_dir: str = CORPUS_PATH,
        glob_pattern: str = "**/*.txt",
        additional_metadata: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False
    ):
        """
        索引文档
        
        参数:
            documents_dir: 文档目录
            glob_pattern: 文件匹配模式
            additional_metadata: 额外元数据
            force_reindex: 是否强制重新索引
        """
        # 检查向量存储是否已存在
        vector_store_exists = False
        if self.vector_db_type.lower() == "chroma" and os.path.exists(self.persist_directory):
            vector_store_exists = True
        
        # 如果向量存储已存在且不强制重新索引，则直接加载
        if vector_store_exists and not force_reindex:
            print(f"加载现有向量存储...")
            self.retriever.load_vector_store(
                embedding_model=self.embedding_model,
                vector_store_type=self.vector_db_type,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            return
        
        print(f"开始索引文档目录: {documents_dir}")
        
        # 处理文档
        documents = self.document_processor.process_documents(
            directory_path=documents_dir,
            glob_pattern=glob_pattern,
            additional_metadata=additional_metadata
        )
        
        # 创建向量存储
        self.retriever.create_vector_store(
            documents=documents,
            embedding_model=self.embedding_model,
            vector_store_type=self.vector_db_type,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        print(f"文档索引完成，共 {len(documents)} 个文档块")
    
    def retrieve(
        self,
        query: str,
        retriever_type: str = RETRIEVER_TYPE,
        filter_metadata: Optional[Dict[str, Any]] = None,
        top_k: int = TOP_K
    ) -> List[Document]:
        """
        检索文档
        
        参数:
            query: 查询文本
            retriever_type: 检索器类型
            filter_metadata: 元数据过滤条件
            top_k: 返回的文档数量
            
        返回:
            检索到的文档列表
        """
        # 更新检索参数
        self.retriever.search_kwargs["k"] = top_k
        
        # 执行检索
        documents = self.retriever.retrieve(
            query=query,
            retriever_type=retriever_type,
            filter_metadata=filter_metadata
        )
        
        return documents
    
    def format_documents(self, docs: List[Document]) -> str:
        """
        格式化文档为字符串
        
        参数:
            docs: 文档列表
            
        返回:
            格式化后的字符串
        """
        formatted_docs = []
        
        for i, doc in enumerate(docs):
            content = doc.page_content
            metadata = doc.metadata
            source = metadata.get("source", "未知来源")
            filename = os.path.basename(source) if source else "未知文件"
            
            formatted_doc = f"[文档 {i+1}] {content}\n"
            formatted_doc += f"来源: {filename}\n"
            
            formatted_docs.append(formatted_doc)
        
        return "\n".join(formatted_docs)
    
    # 移除answer_with_prompt方法，因为我们不再使用LLM
    
    def query(
        self,
        query: str,
        retriever_type: str = RETRIEVER_TYPE,
        filter_metadata: Optional[Dict[str, Any]] = None,
        top_k: int = TOP_K,
        return_docs: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        执行完整的RAG查询流程
        
        参数:
            query: 查询文本
            retriever_type: 检索器类型
            filter_metadata: 元数据过滤条件
            top_k: 返回的文档数量
            return_docs: 是否返回检索到的文档
            
        返回:
            检索结果，或包含检索结果和文档的字典
        """
        # 检索文档
        retrieved_docs = self.retrieve(
            query=query,
            retriever_type=retriever_type,
            filter_metadata=filter_metadata,
            top_k=top_k
        )
        
        # 只返回检索到的文档，因为我们只使用本地嵌入模型进行检索
        formatted_docs = self.format_documents(retrieved_docs)
        answer = f"使用本地嵌入模型检索结果：\n\n检索到的文档:\n{formatted_docs}"
        
        # 返回结果
        if return_docs:
            return {
                "answer": answer,
                "documents": retrieved_docs
            }
        else:
            return answer
    
    # 移除create_retrieval_chain方法，因为我们不再使用LLM和RetrievalQA


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='LangChain RAG系统示例')
    parser.add_argument('--docs_dir', type=str, default=CORPUS_PATH,
                        help='文档目录路径')
    parser.add_argument('--index', action='store_true',
                        help='是否重新索引文档')
    parser.add_argument('--retriever', type=str, default=RETRIEVER_TYPE,
                        choices=['vector', 'hybrid'],
                        help='检索器类型')
    parser.add_argument('--embedding_model', type=str, default=LOCAL_EMBEDDING_MODEL,
                        help='本地嵌入模型名称')
    
    args = parser.parse_args()
    
    # 初始化RAG系统
    rag = LangChainRAG(
        embedding_model=args.embedding_model
    )
    
    # 索引文档
    rag.index_documents(
        documents_dir=args.docs_dir,
        force_reindex=args.index
    )
    
    # 简单的交互式查询循环
    print("\n=== 本地嵌入模型RAG系统演示 ===")
    print(f"检索器: {args.retriever}")
    print(f"嵌入模型: {args.embedding_model}")
    print("输入问题进行查询，输入'exit'或'quit'退出")
    
    while True:
        query = input("\n请输入您的问题: ")
        if query.lower() in ['exit', 'quit', '退出']:
            break
        
        if not query.strip():
            continue
        
        # 执行查询
        print("\n检索结果:")
        answer = rag.query(
            query=query,
            retriever_type=args.retriever
        )
        
        print(answer)
