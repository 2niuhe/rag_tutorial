"""
混合搜索示例

演示如何结合向量检索和关键词检索来提高检索效果
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
    """混合搜索示例主函数"""
    print("=== 混合搜索示例 ===\n")
    
    # 初始化RAG系统
    rag = LangChainRAG()
    
    # 索引文档
    print("索引文档...\n")
    rag.index_documents(documents_dir=CORPUS_PATH)
    
    # 加载文档用于BM25检索器
    documents = rag.document_processor.process_documents(directory_path=CORPUS_PATH)
    
    # 示例查询
    query = "如何管理安全组？"
    print(f"\n查询: {query}")
    
    # 1. 使用纯向量检索
    print("\n使用纯向量检索的结果:")
    vector_docs = rag.retrieve(
        query=query,
        retriever_type="vector",
        top_k=3
    )
    
    for i, doc in enumerate(vector_docs):
        filename = os.path.basename(doc.metadata.get("source", "未知"))
        print(f"文档 {i+1}: {filename}")
    
    # 2. 使用混合检索
    print("\n使用混合检索的结果:")
    hybrid_docs = rag.retriever.retrieve(
        query=query,
        retriever_type="hybrid",
        documents=documents
    )
    
    for i, doc in enumerate(hybrid_docs):
        filename = os.path.basename(doc.metadata.get("source", "未知"))
        print(f"文档 {i+1}: {filename}")
    
    # 3. 使用混合检索生成回答
    print("\n使用混合检索生成的回答:")
    
    # 由于混合检索需要文档参数，我们需要手动检索然后生成回答
    answer = rag.answer_with_prompt(
        query=query,
        docs=hybrid_docs,
        prompt_type="cot"  # 使用思维链提示
    )
    
    if not rag.streaming:
        print(answer)
    
    print("\n示例完成。")

if __name__ == "__main__":
    main()
