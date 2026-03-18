from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Message:
    """表示对话中的单条消息"""
    role: str     # "user" 或 "assistant"
    content: str  # 消息内容

class SessionManager:
    """管理对话会话和消息历史"""
    
    def __init__(self, max_history: int = 5):
        """
        初始化会话管理器
        
        Args:
            max_history: 最大历史消息数量
        """
        self.max_history = max_history
        self.sessions: Dict[str, List[Message]] = {}
        self.session_counter = 0
    
    def create_session(self) -> str:
        """创建新的对话会话"""
        self.session_counter += 1
        session_id = f"session_{self.session_counter}"
        self.sessions[session_id] = []
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str):
        """向对话历史添加消息"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        message = Message(role=role, content=content)
        self.sessions[session_id].append(message)
        
        # 保持对话历史在限制范围内
        if len(self.sessions[session_id]) > self.max_history * 2:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history * 2:]
    
    def add_exchange(self, session_id: str, user_message: str, assistant_message: str):
        """添加完整的问答交流"""
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", assistant_message)
    
    def get_conversation_history(self, session_id: Optional[str]) -> Optional[str]:
        """获取会话的格式化对话历史"""
        if not session_id or session_id not in self.sessions:
            return None
        
        messages = self.sessions[session_id]
        if not messages:
            return None
        
        # 为上下文格式化消息
        formatted_messages = []
        for msg in messages:
            formatted_messages.append(f"{msg.role.title()}: {msg.content}")
        
        return "\n".join(formatted_messages)
    
    def clear_session(self, session_id: str):
        """清除会话中的所有消息"""
        if session_id in self.sessions:
            self.sessions[session_id] = []