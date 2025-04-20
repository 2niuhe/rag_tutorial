"""
文档加载工具模块

这个模块提供了从文件系统加载文本文档的功能。
"""

import os
from typing import List, Dict, Optional


def load_documents(directory_path: str, file_extension: str = ".txt") -> List[Dict[str, str]]:
    """
    从指定目录加载文本文档
    
    参数:
        directory_path: 文档目录的路径
        file_extension: 要加载的文件扩展名，默认为.txt
        
    返回:
        包含文档内容和元数据的字典列表
    """
    documents = []
    
    # 确保目录存在
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"目录不存在: {directory_path}")
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith(file_extension):
            file_path = os.path.join(directory_path, filename)
            
            # 读取文件内容
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # 创建文档字典，包含内容和元数据
                document = {
                    'content': content,
                    'metadata': {
                        'source': file_path,
                        'filename': filename,
                        'file_extension': file_extension
                    }
                }
                
                documents.append(document)
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {str(e)}")
    
    print(f"已加载 {len(documents)} 个文档")
    return documents


def load_single_document(file_path: str) -> Optional[Dict[str, str]]:
    """
    加载单个文本文档
    
    参数:
        file_path: 文档的文件路径
        
    返回:
        包含文档内容和元数据的字典，如果加载失败则返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        filename = os.path.basename(file_path)
        file_extension = os.path.splitext(filename)[1]
        
        document = {
            'content': content,
            'metadata': {
                'source': file_path,
                'filename': filename,
                'file_extension': file_extension
            }
        }
        
        return document
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {str(e)}")
        return None
