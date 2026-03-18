import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

from config import config
from rag_system import RAGSystem

# 初始化 FastAPI 应用
app = FastAPI(title="Course Materials RAG System", root_path="")

# 添加信任主机中间件用于代理
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# 启用 CORS 并配置代理设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 初始化 RAG 系统
rag_system = RAGSystem(config)

# Pydantic 请求/响应模型
class QueryRequest(BaseModel):
    """课程查询请求模型"""
    query: str  # 用户查询内容
    session_id: Optional[str] = None  # 可选的会话ID

class QueryResponse(BaseModel):
    """课程查询响应模型"""
    answer: str  # AI生成的回答
    sources: List[str]  # 回答的来源信息
    session_id: str  # 会话ID（用于后续对话）

class CourseStats(BaseModel):
    """课程统计响应模型"""
    total_courses: int  # 总课程数量
    course_titles: List[str]  # 所有课程标题列表

# API 端点

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """处理查询并返回带来源的响应"""
    try:
        # 如果没有提供会话ID，创建新会话
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()
        
        # 使用 RAG 系统处理查询
        answer, sources = rag_system.query(request.query, session_id)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """获取课程分析和统计信息"""
    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """启动事件：加载初始文档"""
    docs_path = "../docs"
    if os.path.exists(docs_path):
        print("Loading initial documents...")
        try:
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
            print(f"Loaded {courses} courses with {chunks} chunks")
        except Exception as e:
            print(f"Error loading documents: {e}")

# 自定义静态文件处理器，开发环境下禁用缓存
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path

class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # 开发环境下添加无缓存头
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    
# 为前端提供静态文件服务
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")