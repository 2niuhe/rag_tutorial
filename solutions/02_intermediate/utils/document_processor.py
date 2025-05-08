"""
文档处理工具

提供高级文档加载和分块功能
"""

from typing import List, Dict, Any, Optional
import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentProcessor:
    """高级文档处理器类"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        """
        初始化文档处理器
        
        参数:
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            separators: 分割文本的分隔符列表
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 默认分隔符，按优先级排序
        if separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]
        else:
            self.separators = separators
            
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
    
    def load_documents(self, directory_path: str, glob_pattern: str = "**/*.txt") -> List[Document]:
        """
        从目录加载文档
        
        参数:
            directory_path: 文档目录路径
            glob_pattern: 文件匹配模式
            
        返回:
            LangChain文档对象列表
        """
        # 确保目录存在
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        # 使用LangChain的DirectoryLoader加载文档
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        
        documents = loader.load()
        print(f"已加载 {len(documents)} 个文档")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档分割成块
        
        参数:
            documents: LangChain文档对象列表
            
        返回:
            分割后的文档块列表
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"已将 {len(documents)} 个文档分割成 {len(chunks)} 个文本块")
        
        return chunks
    
    def enrich_metadata(self, documents: List[Document], metadata: Dict[str, Any]) -> List[Document]:
        """
        为文档添加额外的元数据
        
        参数:
            documents: LangChain文档对象列表
            metadata: 要添加的元数据字典
            
        返回:
            添加了元数据的文档列表
        """
        enriched_docs = []
        
        for doc in documents:
            # 复制原始元数据
            new_metadata = doc.metadata.copy()
            # 添加新元数据
            new_metadata.update(metadata)
            
            # 创建新文档
            enriched_doc = Document(
                page_content=doc.page_content,
                metadata=new_metadata
            )
            
            enriched_docs.append(enriched_doc)
        
        return enriched_docs
    
    def process_documents(
        self, 
        directory_path: str, 
        glob_pattern: str = "**/*.txt",
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        完整的文档处理流程：加载、分块、添加元数据
        
        参数:
            directory_path: 文档目录路径
            glob_pattern: 文件匹配模式
            additional_metadata: 要添加的额外元数据
            
        返回:
            处理后的文档块列表
        """
        # 加载文档
        documents = self.load_documents(directory_path, glob_pattern)
        
        # 分割文档
        chunks = self.split_documents(documents)
        
        # 添加额外元数据
        if additional_metadata:
            chunks = self.enrich_metadata(chunks, additional_metadata)
        
        return chunks
