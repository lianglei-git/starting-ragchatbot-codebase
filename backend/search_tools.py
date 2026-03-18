from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """所有工具的抽象基类"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """返回此工具的Anthropic工具定义"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """使用给定参数执行工具"""
        pass


class CourseSearchTool(Tool):
    """用于搜索课程内容的工具，支持语义课程名称匹配"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # 跟踪上次搜索的来源
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """返回此工具的Anthropic工具定义"""
        # Message(id='msg_1773728609594074837', 
        #         content=[ToolUseBlock(id='tooluse_mzMjgoCEA72c7y1tTw1sMd', 
        #                               input={'course_name': 'MCP', 'query': 'what is MCP'}, 
        #                               name='search_course_content', type='tool_use')],
        #                                 model='claude-haiku-4-5-20251001', role='assistant', 
        #                                 stop_reason='tool_use', 
        #                                 stop_sequence=None, 
        #                                 ype='message', 
        #                                 usage=Usage(cache_creation_input_tokens=None, cache_read_input_tokens=None, input_tokens=24, output_tokens=30, server_tool_use=None, service_tier=None))
        return {
            "name": "search_course_content",
            "description": "使用智能课程名称匹配和课程过滤搜索课程材料",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "在课程内容中搜索什么"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "课程标题（支持部分匹配，例如'MCP', 'Introduction'）"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "要搜索的特定课程编号（例如1, 2, 3）"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        使用给定参数执行搜索工具
        
        Args:
            query: 搜索内容
            course_name: 可选的课程过滤器
            lesson_number: 可选的课程过滤器
            
        Returns:
            格式化的搜索结果或错误消息
        """
        
        # 使用向量存储的统一搜索接口
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # 处理错误
        if results.error:
            return results.error
        
        # 处理空结果
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # 格式化并返回结果
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """使用课程和课程上下文格式化搜索结果"""
        formatted = []
        sources = []  # 为UI跟踪来源
        
        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')
            
            # 构建上下文头
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"
            
            # 为UI跟踪来源
            source = course_title
            if lesson_num is not None:
                source += f" - Lesson {lesson_num}"
            sources.append(source)
            
            formatted.append(f"{header}\n{doc}")
        
        # 存储来源供检索
        self.last_sources = sources
        
        return "\n\n".join(formatted)

class ToolManager:
    """管理AI可用的工具"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """注册任何实现Tool接口的工具"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """获取所有工具定义用于Anthropic工具调用"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """按名称使用给定参数执行工具"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """从最后一次搜索操作获取来源"""
        # 检查所有工具的last_sources属性
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """重置所有跟踪来源的工具的来源"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []