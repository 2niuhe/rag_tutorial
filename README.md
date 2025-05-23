# RAG 检索增强生成 教程

这个项目是一个从零开始学习RAG（检索增强生成）技术的教程。项目包含多个示例，从简单到复杂，帮助你理解和实现RAG系统。

## 项目结构

```
rag_tutorial/
├── corpus/                 # 文档集合（Nova命令文档）
├── solutions/              # 不同复杂度的RAG实现方案
│   ├── 01_basic/           # 基础RAG实现
│   ├── 02_intermediate/    # 中级RAG实现
│   ├── 03_advanced/        # 高级RAG实现
│   └── 04_production/      # 生产级RAG实现
├── README.md               # 项目说明
└── requirements.txt        # 项目依赖
```

## RAG实现方案提纲

### 1. 基础RAG实现 (01_basic)

**目标**: 理解RAG的基本概念和简单实现

- **技术栈**:
  - 向量数据库: ChromaDB (本地)
  - 嵌入模型: Sentence-Transformers (本地)
  - 检索方法: 简单向量相似度搜索
  - 实现框架: 原生Python实现 (无框架)

- **功能**:
  - 基本文档加载和分块
  - 文本嵌入生成
  - 简单相似度检索
  - 基础问答功能

- **适用场景**:
  - 小型文档集
  - 本地开发和学习
  - 概念验证

### 2. 中级RAG实现 (02_intermediate)

**目标**: 学习使用专业框架构建更健壮的RAG系统

- **技术栈**:
  - 向量数据库: ChromaDB/FAISS
  - 嵌入模型: OpenAI Embeddings
  - 检索方法: 向量检索 + 重排序
  - 实现框架: LangChain

- **功能**:
  - 高级文档处理和分块策略
  - 元数据过滤和混合检索
  - 提示工程和上下文增强
  - 流式响应

- **适用场景**:
  - 中等规模文档集
  - 需要更精确检索的应用
  - 需要框架支持的项目

### 3. 高级RAG实现 (03_advanced)

**目标**: 掌握高级RAG技术和优化方法

- **技术栈**:
  - 向量数据库: 多数据库集成
  - 嵌入模型: 多模型集成或自定义模型
  - 检索方法: 多查询扩展、语义分块
  - 实现框架: LlamaIndex/Haystack

- **功能**:
  - 查询重写和分解
  - 多模态RAG (文本+图像)
  - 上下文压缩和优化
  - 高级提示模板
  - 自定义检索器和重排序

- **适用场景**:
  - 大规模异构数据
  - 复杂查询需求
  - 需要高精度的应用

### 4. 生产级RAG实现 (04_production)

**目标**: 构建可扩展、可监控的生产级RAG系统

- **技术栈**:
  - 向量数据库: 分布式向量存储
  - 嵌入模型: 微调的领域特定模型
  - 检索方法: 混合检索策略
  - 实现框架: 完整应用框架 (如Dify、RAGFlow)

- **功能**:
  - 分布式索引和检索
  - 实时更新和同步
  - 监控和评估系统
  - 缓存和性能优化
  - 用户反馈循环
  - Web界面和API

- **适用场景**:
  - 企业级应用
  - 高并发系统
  - 需要持续改进的应用

## 安装

1. 创建并激活虚拟环境:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

## 使用方法

每个解决方案目录包含:
- README.md: 详细说明和教程
- 完整的源代码
- 示例数据和配置
- 运行说明

按照难度顺序学习每个解决方案，逐步掌握RAG技术。
