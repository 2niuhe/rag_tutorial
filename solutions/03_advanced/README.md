# 高级RAG实现

## 概述

这是一个高级难度的RAG（检索增强生成）实现，使用LlamaIndex/Haystack框架构建复杂、高精度的RAG系统。本实现引入了先进的检索技术和优化策略，适合处理大规模异构数据和复杂查询需求的应用场景。

## 技术栈

- **向量数据库**: 多数据库集成 (Pinecone, Weaviate, Qdrant等)
- **嵌入模型**: 多模型集成或自定义模型
- **检索方法**: 多查询扩展、语义分块、查询路由
- **实现框架**: LlamaIndex/Haystack

## 功能

- 查询重写和分解
- 多模态RAG (文本+图像)
- 上下文压缩和优化
- 高级提示模板
- 自定义检索器和重排序

## 文件结构

```
03_advanced/
├── README.md                      # 本文档
├── requirements.txt               # 依赖项
├── llamaindex_rag.py              # LlamaIndex实现
├── haystack_rag.py                # Haystack实现
├── config.yaml                    # 配置文件
├── modules/
│   ├── query_transformers.py      # 查询转换器
│   ├── custom_retrievers.py       # 自定义检索器
│   ├── rerankers.py               # 重排序器
│   ├── context_compressors.py     # 上下文压缩器
│   └── multimodal_processor.py    # 多模态处理器
└── examples/
    ├── query_decomposition.py     # 查询分解示例
    ├── multimodal_rag.py          # 多模态RAG示例
    ├── custom_retrieval.py        # 自定义检索示例
    └── advanced_evaluation.py     # 高级评估示例
```

## 实现步骤

本实现使用LlamaIndex/Haystack框架，将RAG系统分解为以下几个高级组件：

1. **高级文档处理**:
   - 语义分块和层次化索引
   - 多模态内容处理（文本、图像、表格）
   - 结构化数据提取和索引

2. **查询优化**:
   - 查询重写和扩展
   - 查询分解（子查询生成）
   - 查询路由到专门的检索器

3. **高级检索**:
   - 混合检索策略
   - 多阶段检索管道
   - 自适应检索（基于查询类型）

4. **上下文处理**:
   - 检索结果合并和去重
   - 上下文压缩和总结
   - 动态上下文窗口调整

5. **增强生成**:
   - 高级提示工程
   - 结构化输出格式化
   - 引用和来源追踪

## 使用方法

### 环境设置

1. 创建`.env`文件并添加必要的API密钥：
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
WEAVIATE_API_KEY=your_weaviate_api_key
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置`config.yaml`文件，根据需要调整参数。

### 运行示例

```bash
# 查询分解示例
python examples/query_decomposition.py

# 多模态RAG示例
python examples/multimodal_rag.py

# 自定义检索示例
python examples/custom_retrieval.py

# 高级评估示例
python examples/advanced_evaluation.py
```

## 核心技术详解

### 1. 查询转换和分解

查询转换和分解是高级RAG系统的关键优化技术，可以显著提高复杂查询的检索质量。

```python
# 使用LlamaIndex实现查询分解
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.tools import QueryEngineTool
from llama_index.indices.vector_store import VectorStoreIndex

# 创建专门的索引
general_index = VectorStoreIndex.from_documents(general_docs)
technical_index = VectorStoreIndex.from_documents(technical_docs)

# 创建查询工具
query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=general_index.as_query_engine(),
        description="用于一般信息查询"
    ),
    QueryEngineTool.from_defaults(
        query_engine=technical_index.as_query_engine(),
        description="用于技术细节查询"
    )
]

# 创建子查询引擎
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    verbose=True
)

# 执行复杂查询
response = sub_question_engine.query(
    "Nova的flavor-create命令有哪些参数，以及如何使用这些参数创建一个高内存实例？"
)
```

### 2. 多模态RAG

多模态RAG扩展了传统RAG的能力，允许系统处理和检索图像、音频等非文本数据。

```python
# 使用LlamaIndex实现多模态RAG
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.schema import ImageDocument
from PIL import Image

# 加载图像文档
image_documents = []
for image_path in image_paths:
    image = Image.open(image_path)
    image_document = ImageDocument(
        image=image,
        metadata={"source": image_path}
    )
    image_documents.append(image_document)

# 创建多模态索引
from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex

multi_modal_index = MultiModalVectorStoreIndex.from_documents(
    documents=text_documents + image_documents
)

# 创建多模态查询引擎
multi_modal_llm = OpenAIMultiModal(model="gpt-4-vision-preview")
query_engine = multi_modal_index.as_query_engine(multi_modal_llm=multi_modal_llm)

# 执行多模态查询
response = query_engine.query(
    "描述这些图像中显示的Nova界面并解释如何使用它"
)
```

### 3. 高级检索策略

高级RAG系统通常使用多阶段检索和重排序策略来提高检索质量。

```python
# 使用Haystack实现高级检索管道
from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes import EmbeddingRetriever, BM25Retriever, JoinDocuments
from haystack.nodes import FARMReader, TransformersRanker
from haystack.pipelines import Pipeline

# 设置文档存储
document_store = WeaviateDocumentStore(
    host="localhost", port=8080, embedding_dim=768
)
document_store.write_documents(documents)

# 创建多种检索器
embedding_retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)
bm25_retriever = BM25Retriever(document_store=document_store)

# 创建重排序器
ranker = TransformersRanker(
    model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2"
)

# 创建阅读器
reader = FARMReader(
    model_name_or_path="deepset/roberta-base-squad2"
)

# 创建高级检索管道
pipeline = Pipeline()
pipeline.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["Query"])
pipeline.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
pipeline.add_node(component=JoinDocuments(join_mode="concatenate"), name="JoinResults", 
                 inputs=["EmbeddingRetriever", "BM25Retriever"])
pipeline.add_node(component=ranker, name="Ranker", inputs=["JoinResults"])
pipeline.add_node(component=reader, name="Reader", inputs=["Ranker"])

# 执行查询
results = pipeline.run(
    query="Nova如何管理安全组？",
    params={"EmbeddingRetriever": {"top_k": 10}, "BM25Retriever": {"top_k": 10}}
)
```

### 4. 上下文优化

上下文优化技术可以提高LLM的回答质量，特别是在处理长文档或复杂查询时。

```python
# 使用LlamaIndex实现上下文压缩
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.indices.query.schema import QueryBundle

# 创建重排序器
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_n=5
)

# 创建检索器
retriever = vector_index.as_retriever(similarity_top_k=20)

# 执行检索和重排序
query_bundle = QueryBundle(query_str="如何使用Nova创建一个新的实例？")
retrieved_nodes = retriever.retrieve(query_bundle)
reranked_nodes = reranker.postprocess_nodes(
    retrieved_nodes, query_bundle
)

# 使用压缩后的上下文生成回答
response = llm.complete(
    prompt=f"基于以下上下文回答问题：\n{reranked_nodes}\n\n问题：{query_bundle.query_str}"
)
```

## 评估和优化

高级RAG系统需要全面的评估和持续优化：

1. **评估指标**:
   - 相关性: nDCG, MRR
   - 准确性: 精确度/召回率
   - 回答质量: ROUGE, BLEU
   - 幻觉比例: 自定义评估

2. **优化策略**:
   - 嵌入模型选择和微调
   - 分块参数优化
   - 检索参数调整
   - 提示模板优化

## 后续进阶

完成这个高级实现后，建议探索以下进阶主题：

1. 分布式索引和检索
2. 实时更新和同步
3. 监控和评估系统
4. 用户反馈循环

下一步，请查看生产级RAG实现 (`04_production/`)，了解如何构建可扩展、可监控的生产级RAG系统。
