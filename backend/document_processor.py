import os
import re
from typing import List, Tuple
from models import Course, Lesson, CourseChunk

class DocumentProcessor:
    """处理课程文档并提取结构化信息"""
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        初始化文档处理器
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 块之间的重叠量
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def read_file(self, file_path: str) -> str:
        """使用UTF-8编码从文件读取内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试使用错误处理
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
    


    def chunk_text(self, text: str) -> List[str]:
        """使用配置设置将文本分割成基于句子的块并添加重叠"""
        
        # 清理文本
        text = re.sub(r'\s+', ' ', text.strip())  # 标准化空白字符
        
        # 更好的句子分割，处理缩写
        # 这个正则表达式查找句点后跟空白和大写字母的情况
        # 但忽略常见缩写
        sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+(?=[A-Z])')
        sentences = sentence_endings.split(text)
        
        # 清理句子
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        i = 0
        
        while i < len(sentences):
            current_chunk = []
            current_size = 0
            
            # 从句子i开始构建块
            for j in range(i, len(sentences)):
                sentence = sentences[j]
                
                # 计算带空格的尺寸
                space_size = 1 if current_chunk else 0
                total_addition = len(sentence) + space_size
                
                # 检查添加此句子是否会超过块大小
                if current_size + total_addition > self.chunk_size and current_chunk:
                    break
                
                current_chunk.append(sentence)
                current_size += total_addition
            
            # 如果有内容，添加块
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # 计算下一个块的重叠
                if hasattr(self, 'chunk_overlap') and self.chunk_overlap > 0:
                    # 找到要重叠的句子数量
                    overlap_size = 0
                    overlap_sentences = 0
                    
                    # 从当前块的末尾开始向后计数
                    for k in range(len(current_chunk) - 1, -1, -1):
                        sentence_len = len(current_chunk[k]) + (1 if k < len(current_chunk) - 1 else 0)
                        if overlap_size + sentence_len <= self.chunk_overlap:
                            overlap_size += sentence_len
                            overlap_sentences += 1
                        else:
                            break
                    
                    # 考虑重叠移动起始位置
                    next_start = i + len(current_chunk) - overlap_sentences
                    i = max(next_start, i + 1)  # 确保有进展
                else:
                    # 无重叠 - 移动到当前块后的下一个句子
                    i += len(current_chunk)
            else:
                # 没有句子适合，移动到下一个
                i += 1
        
        return chunks




    
    def process_course_document(self, file_path: str) -> Tuple[Course, List[CourseChunk]]:
        """
        处理具有预期格式的课程文档：
        第1行：Course Title: [标题]
        第2行：Course Link: [url]
        第3行：Course Instructor: [讲师]
        后续行：课程标记和内容
        """
        content = self.read_file(file_path)
        filename = os.path.basename(file_path)
        
        lines = content.strip().split('\n')
        
        # 从前三行提取课程元数据
        course_title = filename  # 默认回退值
        course_link = None
        instructor_name = "Unknown"
        
        # 从第一行解析课程标题
        if len(lines) >= 1 and lines[0].strip():
            title_match = re.match(r'^Course Title:\s*(.+)$', lines[0].strip(), re.IGNORECASE)
            if title_match:
                course_title = title_match.group(1).strip()
            else:
                course_title = lines[0].strip()
        
        # 解析剩余行获取课程元数据
        for i in range(1, min(len(lines), 4)):  # 检查前4行获取元数据
            line = lines[i].strip()
            if not line:
                continue
                
            # 尝试匹配课程链接
            link_match = re.match(r'^Course Link:\s*(.+)$', line, re.IGNORECASE)
            if link_match:
                course_link = link_match.group(1).strip()
                continue
                
            # 尝试匹配讲师
            instructor_match = re.match(r'^Course Instructor:\s*(.+)$', line, re.IGNORECASE)
            if instructor_match:
                instructor_name = instructor_match.group(1).strip()
                continue
        
        # 使用标题作为ID创建课程对象
        course = Course(
            title=course_title,
            course_link=course_link,
            instructor=instructor_name if instructor_name != "Unknown" else None
        )
        # 处理课程并创建块
        course_chunks = []
        current_lesson = None
        lesson_title = None
        lesson_link = None
        lesson_content = []
        chunk_counter = 0
        
        # 从第4行开始处理（元数据之后）
        start_index = 3
        if len(lines) > 3 and not lines[3].strip():
            start_index = 4  # 跳过讲师后的空行
        
        i = start_index
        while i < len(lines):
            line = lines[i]
            
            # 检查课程标记（例如"Lesson 0: Introduction"）
            lesson_match = re.match(r'^Lesson\s+(\d+):\s*(.+)$', line.strip(), re.IGNORECASE)
            
            if lesson_match:
                # 如果存在前一个课程，处理它
                if current_lesson is not None and lesson_content:
                    lesson_text = '\n'.join(lesson_content).strip()
                    if lesson_text:
                        # 将课程添加到课程对象
                        lesson = Lesson(
                            lesson_number=current_lesson,
                            title=lesson_title,
                            lesson_link=lesson_link
                        )
                        course.lessons.append(lesson)
                        
                        # 为此课程创建块
                        chunks = self.chunk_text(lesson_text)
                        for idx, chunk in enumerate(chunks):
                            # 对于每个课程的第一个块，添加上下文
                            if idx == 0:
                                chunk_with_context = f"Lesson {current_lesson} content: {chunk}"
                            else:
                                chunk_with_context = chunk
                            
                            course_chunk = CourseChunk(
                                content=chunk_with_context,
                                course_title=course.title,
                                lesson_number=current_lesson,
                                chunk_index=chunk_counter
                            )
                            course_chunks.append(course_chunk)
                            chunk_counter += 1
                
                # 开始新课程
                current_lesson = int(lesson_match.group(1))
                lesson_title = lesson_match.group(2).strip()
                lesson_link = None
                
                # 检查下一行是否是课程链接
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    link_match = re.match(r'^Lesson Link:\s*(.+)$', next_line, re.IGNORECASE)
                    if link_match:
                        lesson_link = link_match.group(1).strip()
                        i += 1  # 跳过链接行，不添加到内容中
                
                lesson_content = []
            else:
                # 将行添加到当前课程内容
                lesson_content.append(line)
                
            i += 1
        
        # 处理最后一个课程
        if current_lesson is not None and lesson_content:
            lesson_text = '\n'.join(lesson_content).strip()
            if lesson_text:
                lesson = Lesson(
                    lesson_number=current_lesson,
                    title=lesson_title,
                    lesson_link=lesson_link
                )
                course.lessons.append(lesson)
                
                chunks = self.chunk_text(lesson_text)
                for idx, chunk in enumerate(chunks):
                    # 对于每个课程的每个块，添加上下文和课程标题
                  
                    chunk_with_context = f"Course {course_title} Lesson {current_lesson} content: {chunk}"
                    
                    course_chunk = CourseChunk(
                        content=chunk_with_context,
                        course_title=course.title,
                        lesson_number=current_lesson,
                        chunk_index=chunk_counter
                    )
                    course_chunks.append(course_chunk)
                    chunk_counter += 1
        
        # 如果没有找到课程，将整个内容视为一个文档
        if not course_chunks and len(lines) > 2:
            remaining_content = '\n'.join(lines[start_index:]).strip()
            if remaining_content:
                chunks = self.chunk_text(remaining_content)
                for chunk in chunks:
                    course_chunk = CourseChunk(
                        content=chunk,
                        course_title=course.title,
                        chunk_index=chunk_counter
                    )
                    course_chunks.append(course_chunk)
                    chunk_counter += 1
        
        return course, course_chunks
