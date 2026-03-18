import os
from dataclasses import dataclass
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv()

@dataclass
class Config:
    """RAG系统的配置设置"""
    # Anthropic API设置
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")  # Anthropic API密钥
    ANTHROPIC_MODEL: str = "claude-3-5-haiku-20241022"  # Anthropic模型名称
    
    # 嵌入模型设置
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # 语义嵌入模型
    
    # 文档处理设置
    CHUNK_SIZE: int = 800       # 向量存储的文本块大小
    CHUNK_OVERLAP: int = 100    # 块之间的字符重叠量
    MAX_RESULTS: int = 5         # 最大返回搜索结果数量
    MAX_HISTORY: int = 2         # 记住的对话消息数量
    
    # 数据库路径
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB存储位置

config = Config()  # 全局配置实例


