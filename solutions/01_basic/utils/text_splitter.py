"""
文本分块工具模块

这个模块提供了将长文本分割成较小块的功能，用于向量化和检索。
"""

from typing import List, Dict, Union


def split_text(documents: List[Dict[str, str]], 
               chunk_size: int = 500, 
               chunk_overlap: int = 50) -> List[Dict[str, str]]:
    """
    将文档列表分割成较小的文本块
    
    参数:
        documents: 文档字典列表，每个字典包含'content'和'metadata'
        chunk_size: 每个块的最大字符数
        chunk_overlap: 相邻块之间的重叠字符数
        
    返回:
        包含文本块和元数据的字典列表
    """
    chunks = []
    
    for doc in documents:
        content = doc['content']
        metadata = doc['metadata']
        
        # 如果内容长度小于chunk_size，直接作为一个块
        if len(content) <= chunk_size:
            chunk = {
                'content': content,
                'metadata': metadata
            }
            chunks.append(chunk)
            continue
        
        # 分割长文本
        doc_chunks = _split_long_text(content, chunk_size, chunk_overlap)
        
        # 为每个块添加元数据
        for i, chunk_text in enumerate(doc_chunks):
            # 复制原始元数据并添加块信息
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = i
            chunk_metadata['total_chunks'] = len(doc_chunks)
            
            chunk = {
                'content': chunk_text,
                'metadata': chunk_metadata
            }
            chunks.append(chunk)
    
    print(f"已将 {len(documents)} 个文档分割成 {len(chunks)} 个文本块")
    return chunks


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    将长文本分割成重叠的块
    
    参数:
        text: 要分割的文本
        chunk_size: 每个块的最大字符数
        chunk_overlap: 相邻块之间的重叠字符数
        
    返回:
        文本块列表
    """
    # 确保chunk_overlap小于chunk_size
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 2
    
    chunks = []
    start = 0
    
    while start < len(text):
        # 计算当前块的结束位置
        end = start + chunk_size
        
        # 如果不是最后一个块，尝试在句子或段落边界处分割
        if end < len(text):
            # 尝试在句号、问号或感叹号后分割
            for i in range(min(end + 20, len(text) - 1), end - 20, -1):
                if i >= len(text):
                    continue
                if text[i] in ['.', '?', '!', '\n'] and (i + 1 >= len(text) or text[i + 1] == ' ' or text[i + 1] == '\n'):
                    end = i + 1
                    break
        else:
            end = len(text)
        
        # 添加当前块
        chunks.append(text[start:end])
        
        # 更新下一个块的起始位置
        start = end - chunk_overlap
    
    return chunks


def split_by_separator(text: str, separator: str = "\n\n") -> List[str]:
    """
    按指定分隔符分割文本
    
    参数:
        text: 要分割的文本
        separator: 分隔符，默认为两个换行符（段落分隔）
        
    返回:
        文本块列表
    """
    chunks = text.split(separator)
    # 过滤掉空块
    return [chunk.strip() for chunk in chunks if chunk.strip()]
