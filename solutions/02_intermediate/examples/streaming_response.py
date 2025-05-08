"""
本地嵌入模型检索示例

演示如何使用本地嵌入模型进行文档检索
"""

import os
import sys
import time
from dotenv import load_dotenv

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_rag import LangChainRAG
from config import CORPUS_PATH

# 加载环境变量
load_dotenv()

def main():
    """本地嵌入模型检索示例主函数"""
    print("=== 本地嵌入模型检索示例 ===\n")
    
    # 初始化RAG系统
    rag = LangChainRAG()
    
    # 索引文档
    print("索引文档...\n")
    rag.index_documents(documents_dir=CORPUS_PATH)
    
    # 用户查询
    query = "详细解释一下如何使用nova命令创建和管理虚拟机实例？"
    print(f"用户查询: {query}\n")
    
    # 检索相关文档
    print("检索相关文档...")
    start_time = time.time()
    
    docs = rag.retrieve(query=query, top_k=5)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n检索完成！用时: {elapsed:.2f}秒, 检索到 {len(docs)} 个文档\n")
    
    # 显示检索结果
    print("检索结果:")
    for i, doc in enumerate(docs):
        print(f"\n[文档 {i+1}]\n{doc.page_content[:200]}...\n来源: {os.path.basename(doc.metadata.get('source', '未知'))}")

    
    print("\n示例完成。")

if __name__ == "__main__":
    main()
