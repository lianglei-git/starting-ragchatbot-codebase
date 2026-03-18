from typing import List, Tuple, Optional, Dict
import os
from document_processor import DocumentProcessor
from vector_store import VectorStore
from ai_generator import AIGenerator
from session_manager import SessionManager
from search_tools import ToolManager, CourseSearchTool
from models import Course, Lesson, CourseChunk

class RAGSystem:
    """检索增强生成系统的主要协调器"""
    
    def __init__(self, config):
        """
        初始化RAG系统
        
        Args:
            config: 配置对象，包含系统所有配置参数
        """
        self.config = config
        
        # 初始化核心组件
        self.document_processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        self.ai_generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        self.session_manager = SessionManager(config.MAX_HISTORY)
        
        # 初始化搜索工具
        self.tool_manager = ToolManager()
        self.search_tool = CourseSearchTool(self.vector_store)
        self.tool_manager.register_tool(self.search_tool)
    
    def add_course_document(self, file_path: str) -> Tuple[Course, int]:
        """
        将单个课程文档添加到知识库
        
        Args:
            file_path: 课程文档路径
            
        Returns:
            Tuple of (Course对象, 创建的块数量)
        """
        try:
            # 处理文档
            course, course_chunks = self.document_processor.process_course_document(file_path)
            
            # 将课程元数据添加到向量存储以便语义搜索
            self.vector_store.add_course_metadata(course)
            
            # 将课程内容块添加到向量存储
            self.vector_store.add_course_content(course_chunks)
            
            return course, len(course_chunks)
        except Exception as e:
            print(f"Error processing course document {file_path}: {e}")
            return None, 0
    
    def add_course_folder(self, folder_path: str, clear_existing: bool = False) -> Tuple[int, int]:
        """
        从文件夹添加所有课程文档
        
        Args:
            folder_path: 包含课程文档的文件夹路径
            clear_existing: 是否先清除现有数据
            
        Returns:
            Tuple of (添加的总课程数, 创建的总块数)
        """
        total_courses = 0
        total_chunks = 0
        
        # 如果需要，清除现有数据
        if clear_existing:
            print("Clearing existing data for fresh rebuild...")
            self.vector_store.clear_all_data()
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist")
            return 0, 0
        
        # 获取现有课程标题以避免重复处理
        existing_course_titles = set(self.vector_store.get_existing_course_titles())
        
        # 处理文件夹中的每个文件
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.pdf', '.docx', '.txt')):
                try:
                    # 检查此课程是否可能已存在
                    # 我们将处理文档以获取课程ID，但仅在新课程时添加
                    course, course_chunks = self.document_processor.process_course_document(file_path)
                    
                    if course and course.title not in existing_course_titles:
                        # 这是一个新课程 - 添加到向量存储
                        self.vector_store.add_course_metadata(course)
                        self.vector_store.add_course_content(course_chunks)
                        total_courses += 1
                        total_chunks += len(course_chunks)
                        print(f"Added new course: {course.title} ({len(course_chunks)} chunks)")
                        existing_course_titles.add(course.title)
                    elif course:
                        print(f"Course already exists: {course.title} - skipping")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
        
        return total_courses, total_chunks
    
    def query(self, query: str, session_id: Optional[str] = None) -> Tuple[str, List[str]]:
        """
        使用基于工具搜索的RAG系统处理用户查询
        
        Args:
            query: 用户的问题
            session_id: 可选的会话ID用于对话上下文
            
        Returns:
            Tuple of (响应, 来源列表 - 基于工具的方法为空)
        """
        # 为AI创建带有清晰说明的提示
        prompt = f"""Answer this question about course materials: {query}"""
        
        # 如果会话存在，获取对话历史
        history = None
        if session_id:
            history = self.session_manager.get_conversation_history(session_id)
        
        # 使用AI和工具生成响应
        response = self.ai_generator.generate_response(
            query=prompt,
            conversation_history=history,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager
        )
        
        # 从搜索工具获取来源
        sources = self.tool_manager.get_last_sources()

        # 检索后重置来源
        self.tool_manager.reset_sources()
        
        # 更新对话历史
        if session_id:
            self.session_manager.add_exchange(session_id, query, response)
        
        # 返回来搜索工具的响应和来源
        return response, sources
    
    def get_course_analytics(self) -> Dict:
        """获取关于课程目录的分析"""
        return {
            "total_courses": self.vector_store.get_course_count(),
            "course_titles": self.vector_store.get_existing_course_titles()
        }