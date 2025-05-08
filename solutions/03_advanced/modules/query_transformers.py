"""
查询转换器模块

提供高级查询转换功能，包括查询重写、扩展和分解
"""

from typing import List, Dict, Any, Optional, Union
import yaml
import os
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate


class QueryTransformer:
    """查询转换基类"""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        初始化查询转换器
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化LLM
        self.llm = OpenAI(
            model=self.config["llm"]["model"],
            temperature=self.config["llm"]["temperature"]
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def transform(self, query: str) -> Union[str, List[str]]:
        """
        转换查询
        
        参数:
            query: 原始查询
            
        返回:
            转换后的查询或查询列表
        """
        raise NotImplementedError("子类必须实现此方法")


class QueryRewriter(QueryTransformer):
    """查询重写器，用于改进原始查询的表达"""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """初始化查询重写器"""
        super().__init__(config_path)
        
        # 查询重写提示模板
        self.rewrite_template = PromptTemplate(
            """你是一个专业的查询重写专家。你的任务是将用户的原始查询重写为更清晰、更具体、更容易检索的形式。
            重写时，请保持查询的原始意图，但可以添加更多专业术语和上下文，使其更精确。
            不要添加原始查询中不存在的新问题或改变查询的本质含义。
            
            原始查询: {query}
            
            重写后的查询:"""
        )
    
    def transform(self, query: str) -> str:
        """
        重写查询
        
        参数:
            query: 原始查询
            
        返回:
            重写后的查询
        """
        # 使用LLM重写查询
        response = self.llm.complete(
            self.rewrite_template.format(query=query)
        )
        
        rewritten_query = response.text.strip()
        print(f"原始查询: {query}")
        print(f"重写后的查询: {rewritten_query}")
        
        return rewritten_query


class QueryExpander(QueryTransformer):
    """查询扩展器，生成多个相关查询变体"""
    
    def __init__(self, config_path: str = "../config.yaml", num_expansions: int = 3):
        """
        初始化查询扩展器
        
        参数:
            config_path: 配置文件路径
            num_expansions: 扩展查询的数量
        """
        super().__init__(config_path)
        self.num_expansions = num_expansions
        
        # 查询扩展提示模板
        self.expand_template = PromptTemplate(
            """你是一个专业的查询扩展专家。你的任务是基于用户的原始查询，生成{num_expansions}个不同的相关查询变体。
            这些变体应该涵盖原始查询的不同方面，使用不同的表达方式，或者关注不同的细节。
            所有变体都应该与原始查询的主题相关，但提供不同的视角或强调不同的方面。
            
            原始查询: {query}
            
            请生成{num_expansions}个查询变体，每个变体一行:"""
        )
    
    def transform(self, query: str) -> List[str]:
        """
        扩展查询
        
        参数:
            query: 原始查询
            
        返回:
            扩展后的查询列表
        """
        # 使用LLM扩展查询
        response = self.llm.complete(
            self.expand_template.format(
                query=query,
                num_expansions=self.num_expansions
            )
        )
        
        # 解析响应
        expanded_queries = [
            q.strip() for q in response.text.strip().split('\n')
            if q.strip()
        ]
        
        # 确保至少有一个查询
        if not expanded_queries:
            expanded_queries = [query]
        
        print(f"原始查询: {query}")
        print(f"扩展后的查询: {expanded_queries}")
        
        return expanded_queries


class QueryDecomposer(QueryTransformer):
    """查询分解器，将复杂查询分解为多个简单子查询"""
    
    def __init__(self, config_path: str = "../config.yaml", max_sub_questions: int = 3):
        """
        初始化查询分解器
        
        参数:
            config_path: 配置文件路径
            max_sub_questions: 最大子问题数量
        """
        super().__init__(config_path)
        self.max_sub_questions = max_sub_questions
        
        # 查询分解提示模板
        self.decompose_template = PromptTemplate(
            """你是一个专业的查询分解专家。你的任务是将用户的复杂查询分解为最多{max_sub_questions}个简单的子查询。
            这些子查询应该：
            1. 涵盖原始复杂查询的所有方面
            2. 每个子查询都应该是独立的、明确的问题
            3. 子查询的组合应该能够完整回答原始查询
            4. 子查询应该按照逻辑顺序排列
            
            复杂查询: {query}
            
            请将其分解为子查询，每个子查询一行:"""
        )
    
    def transform(self, query: str) -> List[str]:
        """
        分解查询
        
        参数:
            query: 原始复杂查询
            
        返回:
            分解后的子查询列表
        """
        # 使用LLM分解查询
        response = self.llm.complete(
            self.decompose_template.format(
                query=query,
                max_sub_questions=self.max_sub_questions
            )
        )
        
        # 解析响应
        sub_queries = [
            q.strip() for q in response.text.strip().split('\n')
            if q.strip()
        ]
        
        # 确保至少有一个查询
        if not sub_queries:
            sub_queries = [query]
        
        print(f"原始复杂查询: {query}")
        print(f"分解后的子查询: {sub_queries}")
        
        return sub_queries


class QueryTransformationPipeline:
    """查询转换管道，组合多种查询转换策略"""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        初始化查询转换管道
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化转换器
        self.transformers = []
        
        # 根据配置添加转换器
        if self.config["retrieval"]["query_transformation"]["enabled"]:
            strategies = self.config["retrieval"]["query_transformation"]["strategies"]
            
            for strategy in strategies:
                if strategy["type"] == "rewrite":
                    self.transformers.append(QueryRewriter(config_path))
                
                elif strategy["type"] == "expand":
                    num_expansions = strategy.get("num_expansions", 3)
                    self.transformers.append(QueryExpander(config_path, num_expansions))
                
                elif strategy["type"] == "decompose":
                    max_sub_questions = strategy.get("max_sub_questions", 3)
                    self.transformers.append(QueryDecomposer(config_path, max_sub_questions))
    
    def transform(self, query: str) -> Dict[str, Any]:
        """
        执行查询转换
        
        参数:
            query: 原始查询
            
        返回:
            包含各种转换结果的字典
        """
        results = {
            "original": query,
            "rewritten": None,
            "expanded": None,
            "decomposed": None
        }
        
        # 应用各种转换器
        for transformer in self.transformers:
            if isinstance(transformer, QueryRewriter):
                results["rewritten"] = transformer.transform(query)
            
            elif isinstance(transformer, QueryExpander):
                results["expanded"] = transformer.transform(query)
            
            elif isinstance(transformer, QueryDecomposer):
                results["decomposed"] = transformer.transform(query)
        
        return results


if __name__ == "__main__":
    # 测试查询转换
    pipeline = QueryTransformationPipeline("../config.yaml")
    
    test_query = "Nova的flavor是什么以及如何创建和管理flavor？"
    results = pipeline.transform(test_query)
    
    print("\n转换结果:")
    for key, value in results.items():
        if value:
            print(f"\n{key.capitalize()}:")
            if isinstance(value, list):
                for i, item in enumerate(value, 1):
                    print(f"  {i}. {item}")
            else:
                print(f"  {value}")
