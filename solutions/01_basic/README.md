# 基础RAG实现

## 概述

这是最基础的RAG（检索增强生成）实现，旨在帮助初学者理解RAG的核心概念和基本工作流程。本实现不依赖于任何专业RAG框架，而是使用原生Python和基本库来构建一个简单但功能完整的RAG系统。

## 技术栈

- **向量数据库**: ChromaDB (本地)
- **嵌入模型**: Sentence-Transformers (本地)
- **检索方法**: 简单向量相似度搜索
- **实现框架**: 原生Python实现 (无框架)

## 功能

- 基本文档加载和分块
- 文本嵌入生成
- 简单相似度检索
- 基础问答功能

## 文件结构

```
01_basic/
├── README.md                 # 本文档
├── requirements.txt          # 依赖项
├── basic_rag.py              # 主要实现代码
├── utils/
│   ├── document_loader.py    # 文档加载工具
│   ├── text_splitter.py      # 文本分块工具
│   └── embeddings.py         # 嵌入生成工具
└── examples/
    └── simple_qa.py          # 简单问答示例
```

## 实现步骤

本实现将RAG系统分解为以下几个关键步骤：

1. **文档加载**: 从文件系统加载文本文档
2. **文本分块**: 将文档分割成适当大小的块
3. **嵌入生成**: 使用Sentence-Transformers将文本块转换为向量
4. **向量存储**: 使用ChromaDB存储文本块及其向量
5. **查询处理**: 处理用户查询并生成嵌入
6. **相似度检索**: 基于向量相似度检索相关文本块
7. **回答生成**: 使用检索到的上下文生成回答

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
python examples/simple_qa.py
```

## 核心代码概览

```python
# 文档加载
documents = load_documents("path/to/documents")

# 文本分块
text_chunks = split_text(documents)

# 嵌入生成
embeddings = generate_embeddings(text_chunks)

# 向量存储
db = ChromaDB()
db.add(embeddings, text_chunks)

# 查询处理
query = "你的问题"
query_embedding = generate_embedding(query)

# 相似度检索
results = db.query(query_embedding, top_k=3)

# 回答生成
answer = generate_answer(query, results)
```

## 局限性

这个基础实现有以下局限性：

1. 仅支持简单的文本文档
2. 使用基本的分块策略，可能不适合复杂文档
3. 检索仅基于向量相似度，没有高级过滤或重排序
4. 没有考虑大规模数据的性能优化

## 后续进阶

完成这个基础实现后，建议探索以下进阶主题：

1. 使用更复杂的分块策略
2. 添加元数据过滤功能
3. 实现混合检索方法
4. 使用专业框架如LangChain简化实现

下一步，请查看中级RAG实现 (`02_intermediate/`)，了解如何使用专业框架构建更健壮的RAG系统。
