"""
本地嵌入模型演示

演示如何使用本地嵌入模型而不是OpenAI的嵌入模型
"""

import sys
import os
import argparse

# 添加父目录到系统路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_rag import LangChainRAG
from config import CORPUS_PATH, LOCAL_EMBEDDING_MODEL

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='本地嵌入模型演示')
    parser.add_argument('--model', type=str, default=LOCAL_EMBEDDING_MODEL,
                        help='本地嵌入模型名称')
    parser.add_argument('--corpus', type=str, default=CORPUS_PATH,
                        help='文档目录路径')
    parser.add_argument('--reindex', action='store_true',
                        help='是否强制重新索引文档')
    parser.add_argument('--retriever', type=str, default='vector',
                        choices=['vector', 'hybrid'],
                        help='检索器类型')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("本地嵌入模型演示")
    print(f"使用模型: {args.model}")
    print(f"文档目录: {args.corpus}")
    print(f"检索器类型: {args.retriever}")
    print("=" * 50)
    
    try:
        # 初始化RAG系统，使用本地嵌入模型
        rag = LangChainRAG(
            embedding_model=args.model
        )
        
        # 索引文档
        print("\n开始索引文档...")
        rag.index_documents(
            documents_dir=args.corpus,
            force_reindex=args.reindex
        )
        
        # 简单的交互式查询循环
        print("\n=== 本地嵌入模型RAG系统演示 ===")
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
    
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请确保所有依赖项已安装，并检查文档目录是否存在。")
        print("确保 sentence-transformers 已安装，并检查指定的嵌入模型名称是否有效。")

if __name__ == "__main__":
    main()
