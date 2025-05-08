"""
自定义检索器模块

提供高级检索策略，包括混合检索、多阶段检索和自适应检索
"""

from typing import List, Dict, Any, Optional, Union
import yaml
import numpy as np
from llama_index.schema import Document, NodeWithScore
from llama_index.retrievers import BaseRetriever, VectorIndexRetriever, KeywordTableRetriever
from llama_index.retrievers.bm25_retriever import BM25Retriever
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.indices.keyword_table import KeywordTableIndex
from llama_index.vector_stores import PineconeVectorStore, WeaviateVectorStore, QdrantVectorStore
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding


class HybridRetriever(BaseRetriever):
    """混合检索器，结合向量检索和关键词检索"""
    
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        keyword_retriever: BaseRetriever,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_k: int = 10
    ):
        """
        初始化混合检索器
        
        参数:
            vector_retriever: 向量检索器
            keyword_retriever: 关键词检索器
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            top_k: 返回的最大文档数量
        """
        super().__init__()
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.top_k = top_k
    
    def _retrieve(self, query: str) -> List[NodeWithScore]:
        """
        执行混合检索
        
        参数:
            query: 查询文本
            
        返回:
            检索到的节点列表
        """
        # 执行向量检索
        vector_results = self.vector_retriever.retrieve(query)
        
        # 执行关键词检索
        keyword_results = self.keyword_retriever.retrieve(query)
        
        # 合并结果
        combined_results = self._merge_results(
            vector_results, 
            keyword_results,
            self.vector_weight,
            self.keyword_weight
        )
        
        # 返回前top_k个结果
        return combined_results[:self.top_k]
    
    def _merge_results(
        self,
        vector_results: List[NodeWithScore],
        keyword_results: List[NodeWithScore],
        vector_weight: float,
        keyword_weight: float
    ) -> List[NodeWithScore]:
        """
        合并两组检索结果
        
        参数:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            
        返回:
            合并后的结果
        """
        # 创建节点ID到结果的映射
        node_to_result = {}
        
        # 处理向量检索结果
        for result in vector_results:
            node_id = result.node.node_id
            node_to_result[node_id] = {
                "node": result.node,
                "vector_score": result.score or 0.0,
                "keyword_score": 0.0
            }
        
        # 处理关键词检索结果
        for result in keyword_results:
            node_id = result.node.node_id
            if node_id in node_to_result:
                node_to_result[node_id]["keyword_score"] = result.score or 0.0
            else:
                node_to_result[node_id] = {
                    "node": result.node,
                    "vector_score": 0.0,
                    "keyword_score": result.score or 0.0
                }
        
        # 计算组合分数
        combined_results = []
        for node_data in node_to_result.values():
            combined_score = (
                vector_weight * node_data["vector_score"] +
                keyword_weight * node_data["keyword_score"]
            )
            combined_results.append(
                NodeWithScore(
                    node=node_data["node"],
                    score=combined_score
                )
            )
        
        # 按分数排序
        combined_results.sort(key=lambda x: x.score or 0.0, reverse=True)
        
        return combined_results


class MultiStageRetriever(BaseRetriever):
    """多阶段检索器，先广泛检索，再精确过滤"""
    
    def __init__(
        self,
        first_stage_retriever: BaseRetriever,
        second_stage_llm: Optional[OpenAI] = None,
        first_stage_k: int = 20,
        final_k: int = 5
    ):
        """
        初始化多阶段检索器
        
        参数:
            first_stage_retriever: 第一阶段检索器
            second_stage_llm: 第二阶段LLM，用于过滤结果
            first_stage_k: 第一阶段检索的文档数量
            final_k: 最终返回的文档数量
        """
        super().__init__()
        self.first_stage_retriever = first_stage_retriever
        self.second_stage_llm = second_stage_llm or OpenAI(model="gpt-3.5-turbo")
        self.first_stage_k = first_stage_k
        self.final_k = final_k
    
    def _retrieve(self, query: str) -> List[NodeWithScore]:
        """
        执行多阶段检索
        
        参数:
            query: 查询文本
            
        返回:
            检索到的节点列表
        """
        # 第一阶段：广泛检索
        first_stage_results = self.first_stage_retriever.retrieve(query)
        
        # 如果结果太少，直接返回
        if len(first_stage_results) <= self.final_k:
            return first_stage_results
        
        # 第二阶段：使用LLM过滤
        filtered_results = self._filter_with_llm(query, first_stage_results)
        
        return filtered_results[:self.final_k]
    
    def _filter_with_llm(self, query: str, results: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        使用LLM过滤检索结果
        
        参数:
            query: 查询文本
            results: 第一阶段检索结果
            
        返回:
            过滤后的结果
        """
        # 准备LLM输入
        context_texts = []
        for i, result in enumerate(results):
            context_texts.append(f"[{i+1}] {result.node.get_content()}")
        
        context = "\n\n".join(context_texts)
        
        prompt = f"""你是一个专业的文档过滤器。你的任务是从以下文档中选择最相关的文档来回答用户的查询。
        
        用户查询: {query}
        
        文档:
        {context}
        
        请列出最相关的文档编号（最多{self.final_k}个），按相关性从高到低排序。只需返回编号列表，例如：
        1, 4, 7, 2, 5
        """
        
        # 使用LLM过滤
        response = self.second_stage_llm.complete(prompt)
        
        # 解析LLM响应
        try:
            # 尝试解析编号列表
            selected_indices = []
            for part in response.text.strip().split(','):
                try:
                    idx = int(part.strip()) - 1  # 转换为0索引
                    if 0 <= idx < len(results):
                        selected_indices.append(idx)
                except ValueError:
                    continue
            
            # 如果没有有效索引，返回原始结果的前k个
            if not selected_indices:
                return results[:self.final_k]
            
            # 按照LLM给出的顺序返回结果
            filtered_results = [results[idx] for idx in selected_indices]
            return filtered_results
            
        except Exception:
            # 解析失败，返回原始结果的前k个
            return results[:self.final_k]


class AdaptiveRetriever(BaseRetriever):
    """自适应检索器，根据查询类型选择不同的检索策略"""
    
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        keyword_retriever: BaseRetriever,
        hybrid_retriever: BaseRetriever,
        llm: Optional[OpenAI] = None,
        top_k: int = 10
    ):
        """
        初始化自适应检索器
        
        参数:
            vector_retriever: 向量检索器
            keyword_retriever: 关键词检索器
            hybrid_retriever: 混合检索器
            llm: 用于查询分类的LLM
            top_k: 返回的最大文档数量
        """
        super().__init__()
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.hybrid_retriever = hybrid_retriever
        self.llm = llm or OpenAI(model="gpt-3.5-turbo")
        self.top_k = top_k
    
    def _retrieve(self, query: str) -> List[NodeWithScore]:
        """
        执行自适应检索
        
        参数:
            query: 查询文本
            
        返回:
            检索到的节点列表
        """
        # 确定查询类型
        query_type = self._classify_query(query)
        
        # 根据查询类型选择检索器
        if query_type == "keyword":
            print(f"查询 '{query}' 分类为关键词查询，使用关键词检索器")
            results = self.keyword_retriever.retrieve(query)
        elif query_type == "semantic":
            print(f"查询 '{query}' 分类为语义查询，使用向量检索器")
            results = self.vector_retriever.retrieve(query)
        else:  # "hybrid"
            print(f"查询 '{query}' 分类为混合查询，使用混合检索器")
            results = self.hybrid_retriever.retrieve(query)
        
        return results[:self.top_k]
    
    def _classify_query(self, query: str) -> str:
        """
        分类查询类型
        
        参数:
            query: 查询文本
            
        返回:
            查询类型: "keyword", "semantic", 或 "hybrid"
        """
        prompt = f"""你是一个专业的查询分类器。你的任务是将用户的查询分类为以下三种类型之一：
        
        1. 关键词查询(keyword): 包含明确的专业术语、命令名称或参数，适合关键词匹配
        2. 语义查询(semantic): 表达概念性问题或需要理解上下文的查询，适合语义向量检索
        3. 混合查询(hybrid): 同时包含关键词和语义元素的复杂查询，适合混合检索策略
        
        用户查询: {query}
        
        请只回答一个类型: keyword, semantic, 或 hybrid
        """
        
        # 使用LLM分类
        response = self.llm.complete(prompt)
        
        # 解析响应
        response_text = response.text.strip().lower()
        
        if "keyword" in response_text:
            return "keyword"
        elif "semantic" in response_text:
            return "semantic"
        else:
            return "hybrid"


class RetrieverFactory:
    """检索器工厂，用于创建各种检索器"""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        初始化检索器工厂
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化嵌入模型
        self.embedding_model = OpenAIEmbedding(
            model=self.config["embeddings"]["model"]
        )
        
        # 初始化LLM
        self.llm = OpenAI(
            model=self.config["llm"]["model"],
            temperature=self.config["llm"]["temperature"]
        )
    
    def create_vector_store(
        self,
        documents: List[Document],
        vector_store_type: str = "pinecone"
    ) -> Union[PineconeVectorStore, WeaviateVectorStore, QdrantVectorStore]:
        """
        创建向量存储
        
        参数:
            documents: 文档列表
            vector_store_type: 向量存储类型
            
        返回:
            向量存储对象
        """
        if vector_store_type == "pinecone":
            # 创建Pinecone向量存储
            config = self.config["vector_stores"]["primary"]
            vector_store = PineconeVectorStore(
                index_name=config["index_name"],
                namespace=config["namespace"],
                environment="gcp-starter"  # 根据实际情况修改
            )
        
        elif vector_store_type == "weaviate":
            # 创建Weaviate向量存储
            config = self.config["vector_stores"]["secondary"]
            vector_store = WeaviateVectorStore(
                class_name=config["class_name"]
            )
        
        elif vector_store_type == "qdrant":
            # 创建Qdrant向量存储
            config = self.config["vector_stores"]["local"]
            vector_store = QdrantVectorStore(
                collection_name=config["collection_name"],
                path=config["location"]
            )
        
        else:
            raise ValueError(f"不支持的向量存储类型: {vector_store_type}")
        
        return vector_store
    
    def create_vector_retriever(
        self,
        index: VectorStoreIndex,
        top_k: int = 10
    ) -> VectorIndexRetriever:
        """
        创建向量检索器
        
        参数:
            index: 向量索引
            top_k: 返回的最大文档数量
            
        返回:
            向量检索器
        """
        return VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k
        )
    
    def create_keyword_retriever(
        self,
        index: KeywordTableIndex,
        top_k: int = 10
    ) -> KeywordTableRetriever:
        """
        创建关键词检索器
        
        参数:
            index: 关键词索引
            top_k: 返回的最大文档数量
            
        返回:
            关键词检索器
        """
        return KeywordTableRetriever(
            index=index,
            top_k=top_k
        )
    
    def create_bm25_retriever(
        self,
        documents: List[Document],
        top_k: int = 10
    ) -> BM25Retriever:
        """
        创建BM25检索器
        
        参数:
            documents: 文档列表
            top_k: 返回的最大文档数量
            
        返回:
            BM25检索器
        """
        return BM25Retriever.from_defaults(
            documents=documents,
            similarity_top_k=top_k
        )
    
    def create_hybrid_retriever(
        self,
        vector_retriever: BaseRetriever,
        keyword_retriever: BaseRetriever,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_k: int = 10
    ) -> HybridRetriever:
        """
        创建混合检索器
        
        参数:
            vector_retriever: 向量检索器
            keyword_retriever: 关键词检索器
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            top_k: 返回的最大文档数量
            
        返回:
            混合检索器
        """
        return HybridRetriever(
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            top_k=top_k
        )
    
    def create_multi_stage_retriever(
        self,
        first_stage_retriever: BaseRetriever,
        first_stage_k: int = 20,
        final_k: int = 5
    ) -> MultiStageRetriever:
        """
        创建多阶段检索器
        
        参数:
            first_stage_retriever: 第一阶段检索器
            first_stage_k: 第一阶段检索的文档数量
            final_k: 最终返回的文档数量
            
        返回:
            多阶段检索器
        """
        return MultiStageRetriever(
            first_stage_retriever=first_stage_retriever,
            second_stage_llm=self.llm,
            first_stage_k=first_stage_k,
            final_k=final_k
        )
    
    def create_adaptive_retriever(
        self,
        vector_retriever: BaseRetriever,
        keyword_retriever: BaseRetriever,
        hybrid_retriever: BaseRetriever,
        top_k: int = 10
    ) -> AdaptiveRetriever:
        """
        创建自适应检索器
        
        参数:
            vector_retriever: 向量检索器
            keyword_retriever: 关键词检索器
            hybrid_retriever: 混合检索器
            top_k: 返回的最大文档数量
            
        返回:
            自适应检索器
        """
        return AdaptiveRetriever(
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            hybrid_retriever=hybrid_retriever,
            llm=self.llm,
            top_k=top_k
        )
