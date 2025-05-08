"""
重排序器模块

提供高级重排序功能，用于提高检索结果的质量
"""

from typing import List, Dict, Any, Optional, Callable
import yaml
import numpy as np
from llama_index.schema import Document, NodeWithScore
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate
from sentence_transformers import CrossEncoder


class BaseReranker:
    """重排序器基类"""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        初始化重排序器
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_n: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        重排序检索结果
        
        参数:
            query: 查询文本
            nodes: 检索到的节点列表
            top_n: 返回的最大节点数量
            
        返回:
            重排序后的节点列表
        """
        raise NotImplementedError("子类必须实现此方法")


class LLMReranker(BaseReranker):
    """使用LLM的重排序器"""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """初始化LLM重排序器"""
        super().__init__(config_path)
        
        # 初始化LLM
        self.llm = OpenAI(
            model=self.config["llm"]["model"],
            temperature=0.0  # 使用低温度以获得确定性结果
        )
        
        # 重排序提示模板
        self.rerank_template = PromptTemplate(
            """你是一个专业的文档重排序专家。你的任务是评估以下文档与用户查询的相关性，并给每个文档一个0到10的分数。
            10分表示文档与查询高度相关，0分表示完全不相关。
            
            用户查询: {query}
            
            文档:
            {documents}
            
            请为每个文档评分，只返回文档编号和分数，每行一个，格式如下:
            1: 8
            2: 5
            3: 9
            ...
            """
        )
    
    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_n: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        使用LLM重排序检索结果
        
        参数:
            query: 查询文本
            nodes: 检索到的节点列表
            top_n: 返回的最大节点数量
            
        返回:
            重排序后的节点列表
        """
        if not nodes:
            return []
        
        if top_n is None:
            top_n = len(nodes)
        
        # 准备文档文本
        documents_text = ""
        for i, node in enumerate(nodes):
            content = node.node.get_content()
            # 截断过长的内容
            if len(content) > 500:
                content = content[:500] + "..."
            documents_text += f"[{i+1}] {content}\n\n"
        
        # 使用LLM评分
        response = self.llm.complete(
            self.rerank_template.format(
                query=query,
                documents=documents_text
            )
        )
        
        # 解析LLM响应
        try:
            scores = {}
            for line in response.text.strip().split('\n'):
                if ':' in line:
                    parts = line.split(':')
                    try:
                        idx = int(parts[0].strip()) - 1  # 转换为0索引
                        score = float(parts[1].strip())
                        if 0 <= idx < len(nodes):
                            scores[idx] = score
                    except (ValueError, IndexError):
                        continue
            
            # 如果没有解析到有效分数，返回原始顺序
            if not scores:
                return nodes[:top_n]
            
            # 创建新的节点列表，带有LLM评分
            reranked_nodes = []
            for i, node in enumerate(nodes):
                score = scores.get(i, 0.0)
                reranked_nodes.append(
                    NodeWithScore(
                        node=node.node,
                        score=score
                    )
                )
            
            # 按分数排序
            reranked_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
            
            return reranked_nodes[:top_n]
            
        except Exception as e:
            print(f"LLM重排序失败: {e}")
            return nodes[:top_n]


class CrossEncoderReranker(BaseReranker):
    """使用CrossEncoder的重排序器"""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        config_path: str = "../config.yaml"
    ):
        """
        初始化CrossEncoder重排序器
        
        参数:
            model_name: CrossEncoder模型名称
            config_path: 配置文件路径
        """
        super().__init__(config_path)
        
        # 加载CrossEncoder模型
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_n: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        使用CrossEncoder重排序检索结果
        
        参数:
            query: 查询文本
            nodes: 检索到的节点列表
            top_n: 返回的最大节点数量
            
        返回:
            重排序后的节点列表
        """
        if not nodes:
            return []
        
        if top_n is None:
            top_n = len(nodes)
        
        # 准备模型输入
        pairs = [(query, node.node.get_content()) for node in nodes]
        
        # 计算相关性分数
        scores = self.model.predict(pairs)
        
        # 创建新的节点列表，带有CrossEncoder分数
        reranked_nodes = []
        for i, (node, score) in enumerate(zip(nodes, scores)):
            reranked_nodes.append(
                NodeWithScore(
                    node=node.node,
                    score=float(score)
                )
            )
        
        # 按分数排序
        reranked_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        
        return reranked_nodes[:top_n]


class MaximalMarginalRelevanceReranker(BaseReranker):
    """使用最大边际相关性(MMR)的重排序器"""
    
    def __init__(
        self,
        lambda_mult: float = 0.5,
        config_path: str = "../config.yaml"
    ):
        """
        初始化MMR重排序器
        
        参数:
            lambda_mult: 多样性权重，0表示最大多样性，1表示最大相关性
            config_path: 配置文件路径
        """
        super().__init__(config_path)
        self.lambda_mult = lambda_mult
    
    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_n: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        使用MMR重排序检索结果
        
        参数:
            query: 查询文本
            nodes: 检索到的节点列表
            top_n: 返回的最大节点数量
            
        返回:
            重排序后的节点列表
        """
        if not nodes:
            return []
        
        if top_n is None:
            top_n = len(nodes)
        
        # 确保所有节点都有嵌入
        nodes_with_embeddings = []
        for node in nodes:
            if hasattr(node.node, 'embedding') and node.node.embedding is not None:
                nodes_with_embeddings.append(node)
        
        # 如果没有足够的节点有嵌入，返回原始顺序
        if len(nodes_with_embeddings) < 2:
            return nodes[:top_n]
        
        # 执行MMR重排序
        mmr_nodes = self._mmr_rerank(nodes_with_embeddings, top_n)
        
        return mmr_nodes
    
    def _mmr_rerank(
        self,
        nodes: List[NodeWithScore],
        top_n: int
    ) -> List[NodeWithScore]:
        """
        执行MMR重排序算法
        
        参数:
            nodes: 检索到的节点列表
            top_n: 返回的最大节点数量
            
        返回:
            重排序后的节点列表
        """
        # 如果节点数量小于等于top_n，直接返回
        if len(nodes) <= top_n:
            return nodes
        
        # 获取所有节点的嵌入
        embeddings = np.array([node.node.embedding for node in nodes])
        
        # 计算相似度矩阵
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        # 初始化已选择和未选择的节点
        selected_indices = []
        unselected_indices = list(range(len(nodes)))
        
        # 选择第一个节点（最相关的）
        first_idx = max(range(len(nodes)), key=lambda i: nodes[i].score or 0.0)
        selected_indices.append(first_idx)
        unselected_indices.remove(first_idx)
        
        # 迭代选择剩余节点
        while len(selected_indices) < top_n and unselected_indices:
            # 计算MMR分数
            mmr_scores = []
            for i in unselected_indices:
                # 相关性项
                relevance = nodes[i].score or 0.0
                
                # 多样性项
                if selected_indices:
                    diversity = min(1.0 - similarity_matrix[i, j] for j in selected_indices)
                else:
                    diversity = 1.0
                
                # MMR分数
                mmr_score = self.lambda_mult * relevance + (1 - self.lambda_mult) * diversity
                mmr_scores.append((i, mmr_score))
            
            # 选择MMR分数最高的节点
            next_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(next_idx)
            unselected_indices.remove(next_idx)
        
        # 返回重排序后的节点
        return [nodes[i] for i in selected_indices]


class RerankerFactory:
    """重排序器工厂，用于创建各种重排序器"""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        初始化重排序器工厂
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def create_reranker(
        self,
        reranker_type: str = "cross_encoder",
        **kwargs
    ) -> BaseReranker:
        """
        创建重排序器
        
        参数:
            reranker_type: 重排序器类型
            **kwargs: 其他参数
            
        返回:
            重排序器对象
        """
        if reranker_type == "llm":
            return LLMReranker(config_path=kwargs.get("config_path", "../config.yaml"))
        
        elif reranker_type == "cross_encoder":
            model_name = kwargs.get(
                "model_name",
                self.config["retrieval"]["reranking"].get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            )
            return CrossEncoderReranker(
                model_name=model_name,
                config_path=kwargs.get("config_path", "../config.yaml")
            )
        
        elif reranker_type == "mmr":
            lambda_mult = kwargs.get("lambda_mult", 0.5)
            return MaximalMarginalRelevanceReranker(
                lambda_mult=lambda_mult,
                config_path=kwargs.get("config_path", "../config.yaml")
            )
        
        else:
            raise ValueError(f"不支持的重排序器类型: {reranker_type}")


if __name__ == "__main__":
    # 测试重排序器
    import random
    from llama_index.schema import TextNode
    
    # 创建测试节点
    test_nodes = []
    for i in range(10):
        node = TextNode(
            text=f"这是测试文档 {i+1}，包含一些关于Nova的信息。",
            id_=f"node_{i}"
        )
        # 随机分数
        score = random.random()
        test_nodes.append(NodeWithScore(node=node, score=score))
    
    # 创建重排序器
    reranker_factory = RerankerFactory()
    
    # 测试LLM重排序器
    llm_reranker = reranker_factory.create_reranker("llm")
    llm_results = llm_reranker.rerank("Nova命令", test_nodes, top_n=5)
    
    print("LLM重排序结果:")
    for i, result in enumerate(llm_results):
        print(f"{i+1}. 分数: {result.score:.4f}, 内容: {result.node.get_content()}")
