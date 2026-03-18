from typing import List, Dict, Optional
from pydantic import BaseModel

class Lesson(BaseModel):
    """表示课程中的一节课"""
    lesson_number: int  # 顺序课程编号（1、2、3等）
    title: str         # 课程标题
    lesson_link: Optional[str] = None  # 课程链接URL

class Course(BaseModel):
    """表示一个完整的课程及其所有课程"""
    title: str                 # 完整课程标题（用作唯一标识符）
    course_link: Optional[str] = None  # 课程链接URL
    instructor: Optional[str] = None  # 课程讲师名称（可选元数据）
    lessons: List[Lesson] = [] # 此课程中的课程列表

class CourseChunk(BaseModel):
    """表示来自课程的文本块，用于向量存储"""
    content: str                        # 实际文本内容
    course_title: str                   # 此块所属的课程标题
    lesson_number: Optional[int] = None # 此块来自的课程编号
    chunk_index: int                    # 此块在文档中的位置