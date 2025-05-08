"""
提示模板模块

提供高级提示工程模板，用于增强RAG系统的回答质量
"""

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate


# 基础RAG提示模板
RAG_PROMPT_TEMPLATE = """
你是一个智能助手，专门回答关于OpenStack Nova命令的问题。
请基于以下检索到的上下文信息回答用户的问题。
如果上下文中没有足够的信息来回答问题，请直接说"我没有足够的信息来回答这个问题"，不要编造信息。

上下文信息:
{context}

用户问题: {question}

请提供详细、准确的回答，并引用相关的命令和参数。
"""

BASIC_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE
)


# 带有思维链(Chain-of-Thought)的提示模板
COT_PROMPT_TEMPLATE = """
你是一个智能助手，专门回答关于OpenStack Nova命令的问题。
请基于以下检索到的上下文信息回答用户的问题。
如果上下文中没有足够的信息来回答问题，请直接说"我没有足够的信息来回答这个问题"，不要编造信息。

上下文信息:
{context}

用户问题: {question}

请按照以下步骤思考:
1. 分析用户问题，确定他们想要了解的具体Nova命令或功能
2. 从上下文中找出与问题最相关的信息
3. 组织这些信息，形成一个清晰、结构化的回答
4. 确保包含相关的命令语法、参数和示例

现在，请提供你的详细回答:
"""

COT_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=COT_PROMPT_TEMPLATE
)


# 带有引用的提示模板
CITATION_PROMPT_TEMPLATE = """
你是一个智能助手，专门回答关于OpenStack Nova命令的问题。
请基于以下检索到的上下文信息回答用户的问题。
如果上下文中没有足够的信息来回答问题，请直接说"我没有足够的信息来回答这个问题"，不要编造信息。

上下文信息:
{context}

用户问题: {question}

请提供详细、准确的回答，并在回答中明确引用你使用的信息来源（使用[1]、[2]等引用标记）。
在回答的最后，列出所有引用的来源文件。
"""

CITATION_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=CITATION_PROMPT_TEMPLATE
)


# Chat格式的提示模板
CHAT_RAG_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """你是一个智能助手，专门回答关于OpenStack Nova命令的问题。
        请基于检索到的上下文信息回答用户的问题。
        如果上下文中没有足够的信息，请直接说你不知道，不要编造信息。
        
        上下文信息:
        {context}"""
    ),
    HumanMessagePromptTemplate.from_template("{question}")
])


# 动态上下文窗口提示模板
def get_dynamic_context_prompt(max_tokens=3800):
    """
    获取动态上下文窗口提示模板
    
    参数:
        max_tokens: 上下文的最大token数量
        
    返回:
        提示模板
    """
    template = f"""
    你是一个智能助手，专门回答关于OpenStack Nova命令的问题。
    请基于以下检索到的上下文信息回答用户的问题。
    上下文信息已经按相关性排序，你应该优先考虑靠前的信息。
    如果上下文中没有足够的信息来回答问题，请直接说"我没有足够的信息来回答这个问题"，不要编造信息。
    
    上下文信息 (限制在{max_tokens}个tokens内):
    {{context}}
    
    用户问题: {{question}}
    
    请提供详细、准确的回答，并引用相关的命令和参数。
    """
    
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
