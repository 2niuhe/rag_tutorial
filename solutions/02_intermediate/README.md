# 中级RAG实现

## 概述

这是一个中级难度的RAG（检索增强生成）实现，使用LangChain框架构建更健壮、功能更丰富的RAG系统。本实现引入了更高级的检索策略和文档处理方法，适合需要更精确检索结果的应用场景。

## 技术栈

- **向量数据库**: ChromaDB/FAISS
- **嵌入模型**: OpenAI Embeddings
- **检索方法**: 向量检索 + 重排序
- **实现框架**: LangChain

## 功能

- 高级文档处理和分块策略
- 元数据过滤和混合检索
- 提示工程和上下文增强
- 流式响应

## 文件结构

```
02_intermediate/
├── README.md                    # 本文档
├── requirements.txt             # 依赖项
├── langchain_rag.py             # 主要实现代码
├── config.py                    # 配置文件
├── utils/
│   ├── document_processor.py    # 文档处理工具
│   ├── retriever.py             # 自定义检索器
│   └── prompt_templates.py      # 提示模板
└── examples/
    ├── metadata_filtering.py    # 元数据过滤示例
    ├── hybrid_search.py         # 混合搜索示例
    └── streaming_response.py    # 流式响应示例
```

## 实现步骤

本实现使用LangChain框架，将RAG系统分解为以下几个关键步骤：

1. **文档加载与处理**:
   - 使用LangChain的文档加载器加载多种格式文档
   - 应用高级分块策略（语义分块、重叠分块等）
   - 添加丰富的元数据

2. **向量存储与索引**:
   - 使用OpenAI嵌入模型生成高质量嵌入
   - 配置ChromaDB或FAISS作为向量存储
   - 实现元数据索引和过滤

3. **检索增强**:
   - 实现混合检索（向量 + 关键词）
   - 添加检索结果重排序
   - 实现最大边际相关性（MMR）检索

4. **提示工程**:
   - 设计高效的提示模板
   - 实现动态上下文窗口
   - 添加思维链（Chain-of-Thought）提示

5. **回答生成**:
   - 集成OpenAI模型
   - 实现流式响应
   - 添加引用和来源追踪

## 使用方法

### 环境设置

1. 创建`.env`文件并添加API密钥：
```
OPENAI_API_KEY=your_api_key_here
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

### 运行示例

```bash
# 元数据过滤示例
python examples/metadata_filtering.py

# 混合搜索示例
python examples/hybrid_search.py

# 流式响应示例
python examples/streaming_response.py
```

## 核心代码概览

```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 1. 文档加载与处理
loader = DirectoryLoader("path/to/documents", glob="**/*.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 2. 向量存储与索引
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 3. 检索增强
retriever = vectorstore.as_retriever(
    search_type="mmr",  # 最大边际相关性
    search_kwargs={"k": 5, "fetch_k": 10}
)

# 4. 上下文压缩
llm = ChatOpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# 5. 回答生成
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    chain_type="stuff",
    return_source_documents=True
)

# 6. 查询
result = qa_chain({"query": "你的问题"})
print(result["result"])
print("Sources:", [doc.metadata for doc in result["source_documents"]])
```

## 进阶技巧

1. **元数据过滤**:
   ```python
   retriever = vectorstore.as_retriever(
       search_kwargs={"filter": {"category": "documentation"}}
   )
   ```

2. **混合检索**:
   ```python
   from langchain.retrievers import BM25Retriever, EnsembleRetriever
   
   bm25_retriever = BM25Retriever.from_documents(chunks)
   ensemble_retriever = EnsembleRetriever(
       retrievers=[retriever, bm25_retriever],
       weights=[0.7, 0.3]
   )
   ```

3. **流式响应**:
   ```python
   from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
   
   llm = ChatOpenAI(
       streaming=True,
       callbacks=[StreamingStdOutCallbackHandler()]
   )
   ```

## 后续进阶

完成这个中级实现后，建议探索以下进阶主题：

1. 查询重写和分解
2. 多模态RAG（文本+图像）
3. 自定义检索器和重排序
4. 使用LlamaIndex或Haystack框架

下一步，请查看高级RAG实现 (`03_advanced/`)，了解如何构建更复杂、更精确的RAG系统。
