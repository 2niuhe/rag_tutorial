"""
简单问答示例

这个脚本展示了如何使用基础RAG系统进行简单的问答。
"""

import sys
import os

# 添加父目录到路径，以便导入基础RAG模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basic_rag import SimpleRAG


def main():
    """
    运行简单问答示例
    """
    print("=== 基础RAG系统 - 简单问答示例 ===\n")
    
    # 初始化RAG系统
    corpus_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "corpus")
    
    # 检查语料库目录是否存在
    if not os.path.exists(corpus_dir):
        print(f"错误: 语料库目录不存在: {corpus_dir}")
        print(f"当前工作目录: {os.getcwd()}")
        print("请确保您在正确的目录中运行此脚本，或者修改corpus_dir路径")
        return
    
    print(f"使用语料库目录: {os.path.abspath(corpus_dir)}")
    
    # 初始化RAG系统
    rag = SimpleRAG(
        embedding_model_name="all-MiniLM-L6-v2",
        persist_directory="./chroma_db"
    )
    
    # 检查是否需要索引文档
    collection_count = rag.collection.count()
    if collection_count == 0:
        print("集合为空，开始索引文档...")
        rag.index_documents(
            documents_dir=corpus_dir,
            chunk_size=500,
            chunk_overlap=50
        )
    else:
        print(f"集合已包含 {collection_count} 个文档，跳过索引步骤")
    
    # 预定义的示例问题
    example_questions = [
        "Nova是什么？",
        "如何使用Nova创建一个新的实例？",
        "Nova的flavor-list命令有什么用？",
        "如何管理Nova的安全组？",
        "如何使用Nova的keypair功能？"
    ]
    
    print("\n=== 示例问题 ===")
    for i, question in enumerate(example_questions, 1):
        print(f"{i}. {question}")
    
    # 交互式问答循环
    while True:
        print("\n选项:")
        print("1-5: 选择示例问题")
        print("q: 退出")
        print("或者输入您自己的问题")
        
        user_input = input("\n请选择或输入问题: ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit', '退出']:
            break
        
        # 处理用户选择的示例问题
        if user_input.isdigit() and 1 <= int(user_input) <= len(example_questions):
            question = example_questions[int(user_input) - 1]
        else:
            question = user_input
        
        if not question:
            continue
        
        print(f"\n问题: {question}")
        print("\n正在查询，请稍候...\n")
        
        # 执行查询
        answer = rag.query(question, top_k=3)
        
        print("回答:")
        print(answer)
        print("\n" + "-" * 80)


if __name__ == "__main__":
    main()
