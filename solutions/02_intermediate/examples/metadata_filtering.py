"""
元数据过滤示例

演示如何使用元数据过滤来提高检索精度
"""

import os
import sys
from dotenv import load_dotenv

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_rag import LangChainRAG
from config import CORPUS_PATH

# 加载环境变量
load_dotenv()

def main():
    """元数据过滤示例主函数"""
    print("=== 元数据过滤示例 ===\n")
    
    # 初始化RAG系统
    rag = LangChainRAG()
    
    # 索引文档
    print("索引文档...\n")
    
    # 添加额外的元数据，用于过滤
    additional_metadata = {
        "category": "documentation",
        "version": "latest"
    }
    
    rag.index_documents(
        documents_dir=CORPUS_PATH,
        additional_metadata=additional_metadata
    )
    
    # 示例查询
    query = "如何创建虚拟机实例？"
    print(f"\n查询: {query}")
    
    # 1. 不使用过滤器的检索
    print("\n不使用过滤器的检索结果:")
    docs_without_filter = rag.retrieve(query=query, top_k=3)
    
    for i, doc in enumerate(docs_without_filter):
        filename = os.path.basename(doc.metadata.get("source", "未知"))
        print(f"文档 {i+1}: {filename}")
    
    # 2. 使用元数据过滤器的检索
    print("\n使用元数据过滤器的检索结果:")
    
    # 只检索特定文件
    filter_metadata = {
        "source": {"$eq": "../../corpus/nova_boot.txt"}
    }
    
    docs_with_filter = rag.retrieve(
        query=query,
        filter_metadata=filter_metadata,
        top_k=3
    )
    
    for i, doc in enumerate(docs_with_filter):
        filename = os.path.basename(doc.metadata.get("source", "未知"))
        print(f"文档 {i+1}: {filename}")
    
    # 3. 使用过滤器生成回答
    print("\n使用过滤器生成的回答:")
    try:
        answer = rag.query(
            query=query,
            filter_metadata=filter_metadata
        )
        print(answer)
    except Exception as e:
        print(f"\n错误: {e}\n注意: 我们已经移除了LLM功能，只保留了检索功能。")
    
    print("\n示例完成。")

if __name__ == "__main__":
    main()
