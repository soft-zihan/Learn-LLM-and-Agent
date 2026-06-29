"""
FastAPI 服务入口 - 对应教程章节：
- 13-生产部署实践：API 服务化
- 12-测试与可观测性：日志与追踪
- 05-提示词与上下文工程：上下文管理 API
- 06-记忆系统设计：会话隔离 API

本模块将 LangGraph Agent 包装为 Web 服务，提供 REST API：

API 接口：
- POST /api/chat    - 对话接口（SSE 流式响应）
- GET  /api/tools   - 列出可用工具
- GET  /api/health  - 健康检查
- POST /api/mode    - 切换 Agent 模式

会话管理（教程06）：
- GET    /api/threads              - 列出所有对话线程
- POST   /api/threads              - 创建新线程
- DELETE /api/threads/{thread_id}  - 删除线程

上下文管理（教程05）：
- GET  /api/context/{thread_id}           - 获取上下文统计
- POST /api/context/{thread_id}/compress  - 压缩上下文

技能管理（教程04）：
- GET /api/skills - 列出所有技能

MCP管理（教程03）：
- GET /api/mcp/status - MCP 连接状态
- GET /api/mcp/tools  - MCP 工具详情

核心技术：
- FastAPI：高性能异步 Web 框架
- SSE（Server-Sent Events）：服务端推送，实现流式响应
- 生命周期管理：应用启动/关闭时的资源初始化/清理

启动方式：
    cd LangGraph
    ./venv/bin/uvicorn main:app --reload --port 8000

教程对应：
- 13-生产部署实践：FastAPI 服务化、CORS 配置、健康检查
- 12-测试与可观测性：请求日志、错误追踪
- 05-提示词与上下文工程：上下文裁剪、压缩、优先级管理
- 06-记忆系统设计：thread_id 实现会话隔离
"""

import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# 从 agent 模块导入核心功能
from agent import (
    chat, get_agent, init_agent,
    shutdown_mcp, get_mcp_tools,
    set_mode, current_mode, checkpointer
)
from context_manager import get_context_stats, summarize_conversation, trim_context, build_compressed_context
from skills import SkillRegistry


# ── 全局状态 ─────────────────────────────────────────────────

# 线程注册表：跟踪所有活跃的对话线程
# 对应教程06：通过 thread_id 实现记忆隔离
thread_registry: dict[str, dict] = {
    "default": {
        "thread_id": "default",
        "name": "默认对话",
        "created_at": datetime.now().isoformat(),
        "message_count": 0
    }
}

# 技能注册表（教程04）
skill_registry = SkillRegistry()
# 尝试加载技能目录
skills_dir = os.path.join(os.path.dirname(__file__), "skills")
if os.path.exists(skills_dir):
    skill_registry.register_directory(skills_dir)

# MCP 连接状态
mcp_status = {"connected": False, "tools_count": 0}


# ── 生命周期管理（教程13：生产部署）──────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理器
    
    在应用启动时初始化资源，在关闭时清理资源。
    这是 FastAPI 推荐的资源管理方式（替代 on_event）。
    
    启动流程：
    1. 初始化 Agent（连接 MCP、构建图）
    2. yield（应用运行中）
    3. 关闭时清理 MCP 连接
    """
    global mcp_status
    print("[Server] 正在初始化 Agent...")
    await init_agent()
    mcp_status["connected"] = True
    mcp_status["tools_count"] = len(get_mcp_tools())
    print("[Server] Agent 就绪 ✓")
    yield
    print("[Server] 正在关闭...")
    await shutdown_mcp()
    print("[Server] 已关闭 ✓")


# ── 创建 FastAPI 应用 ────────────────────────────────────────

app = FastAPI(
    title="Agent App",
    description="LangGraph + MCP 智能体服务",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS 配置（教程13：跨域安全）─────────────────────────────
# CORS（Cross-Origin Resource Sharing）控制哪些网页可以访问 API
# 生产环境应该限制为具体的前端域名，而非 ["*"]

app.add_middleware(
    CORSMiddleware,
    # 开发环境允许所有来源，生产环境应改为具体域名：
    # allow_origins=["http://localhost:5173", "https://your-domain.com"]
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["GET", "POST"],  # 只允许需要的 HTTP 方法
    allow_headers=["Content-Type"],  # 只允许必要的请求头
)


# ── 请求模型 ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    对话请求模型
    
    Attributes:
        message: 用户输入的消息
        thread_id: 对话线程ID（教程06：通过 thread_id 实现记忆隔离）
        mode: Agent 模式（"react" 或 "plan_execute"）
    """
    message: str
    thread_id: str = "default"
    mode: str = "react"


# ── API 路由 ─────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    对话接口（SSE 流式响应）
    
    前端通过 EventSource 接收实时事件流：
    - token: LLM 流式输出的文字片段
    - tool_call: Agent 决定调用工具
    - tool_result: 工具执行结果
    - done: 对话完成
    
    SSE 优势：
    - 单向推送，比 WebSocket 简单
    - 自动重连
    - 兼容 HTTP/2
    
    Args:
        req: 对话请求（包含消息、线程ID、模式）
    
    Returns:
        EventSourceResponse：SSE 事件流
    """
    async def event_generator():
        """SSE 事件生成器"""
        async for event in chat(req.message, req.thread_id, mode=req.mode):
            # 将事件字典序列化为 JSON，通过 SSE 发送
            yield {"data": json.dumps(event, ensure_ascii=False)}

    return EventSourceResponse(event_generator())


@app.get("/api/tools")
async def list_tools():
    """
    列出当前可用的工具
    
    包括 MCP 工具和内置工具。
    前端可以用这个接口展示可用工具列表。
    
    Returns:
        工具列表，每项包含 name 和 description
    """
    tools = get_mcp_tools()
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
            }
            for t in tools
        ]
    }


@app.get("/api/health")
async def health():
    """
    健康检查接口（教程13：生产部署）
    
    用于负载均衡器、监控系统检查服务是否存活。
    返回当前 Agent 模式信息。
    
    Returns:
        {"status": "ok", "mode": "react"|"plan_execute"}
    """
    return {"status": "ok", "mode": current_mode}


@app.post("/api/mode")
async def switch_mode(req: dict):
    """
    切换 Agent 模式
    
    支持在 ReAct 和 Plan-and-Execute 两种模式之间切换。
    切换后立即生效，影响后续的对话请求。
    
    Args:
        req: {"mode": "react"|"plan_execute"}
    
    Returns:
        切换结果
    """
    mode = req.get("mode", "react")
    try:
        set_mode(mode)
        return {"status": "ok", "mode": mode}
    except ValueError as e:
        return {"status": "error", "message": str(e)}


# ── 会话管理 API（教程06：记忆隔离）──────────────────────────

@app.get("/api/threads")
async def list_threads():
    """
    列出所有对话线程
    
    对应教程06：通过 thread_id 实现记忆隔离
    每个线程有独立的上下文和记忆
    
    Returns:
        线程列表
    """
    return {"threads": list(thread_registry.values())}


@app.post("/api/threads")
async def create_thread(req: dict = None):
    """
    创建新对话线程
    
    生成唯一的 thread_id，前端可以用这个 ID 开始新对话。
    
    Args:
        req: 可选的 {"name": "对话名称"}
    
    Returns:
        新创建的线程信息
    """
    thread_id = str(uuid.uuid4())[:8]
    name = (req or {}).get("name", f"对话 {len(thread_registry) + 1}")
    
    thread_registry[thread_id] = {
        "thread_id": thread_id,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "message_count": 0
    }
    
    return {"status": "ok", "thread_id": thread_id, "name": name}


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """
    删除对话线程
    
    清除该线程的所有记忆和上下文。
    注意：默认线程不能被删除。
    
    Args:
        thread_id: 要删除的线程 ID
    
    Returns:
        删除结果
    """
    if thread_id == "default":
        raise HTTPException(status_code=400, detail="默认线程不能删除")
    
    if thread_id not in thread_registry:
        raise HTTPException(status_code=404, detail="线程不存在")
    
    del thread_registry[thread_id]
    # 注意：LangGraph 的 MemorySaver 中的检查点无法直接删除
    # 实际应用中可以使用其他 checkpointer 支持删除
    
    return {"status": "ok", "deleted": thread_id}


# ── 上下文管理 API（教程05：上下文工程）──────────────────────

@app.get("/api/context/{thread_id}")
async def get_context(thread_id: str):
    """
    获取指定线程的上下文统计
    
    对应教程05：上下文窗口管理策略
    返回消息数、估算 token 数、使用率等统计信息
    
    Args:
        thread_id: 线程 ID
    
    Returns:
        上下文统计信息
    """
    if thread_id not in thread_registry:
        raise HTTPException(status_code=404, detail="线程不存在")
    
    # 获取该线程的消息历史
    # 注意：实际消息存储在 LangGraph 的 checkpointer 中
    # 这里返回线程注册表中的统计
    thread = thread_registry[thread_id]
    
    # 模拟消息列表（实际应用中从 checkpointer 获取）
    messages = [{"role": "user", "content": "示例消息"}] * thread["message_count"]
    
    stats = get_context_stats(messages)
    return {
        "thread_id": thread_id,
        "stats": stats
    }


@app.post("/api/context/{thread_id}/compress")
async def compress_context(thread_id: str):
    """
    压缩指定线程的上下文
    
    对应教程05：策略 2 - 上下文压缩（Summarization）
    用 LLM 生成对话摘要，替代原始消息
    
    Args:
        thread_id: 线程 ID
    
    Returns:
        压缩结果
    """
    if thread_id not in thread_registry:
        raise HTTPException(status_code=404, detail="线程不存在")
    
    # 实际应用中：
    # 1. 从 checkpointer 获取完整消息历史
    # 2. 调用 summarize_conversation() 生成摘要
    # 3. 用摘要替换原始消息
    
    return {
        "status": "ok",
        "thread_id": thread_id,
        "message": "上下文压缩功能需要在实际对话中使用"
    }


# ── 技能管理 API（教程04：Skills）────────────────────────────

@app.get("/api/skills")
async def list_skills():
    """
    列出所有已加载的技能
    
    对应教程04：Skills核心概念
    技能是预定义的能力模块，帮助 LLM 执行特定任务
    
    Returns:
        技能列表
    """
    skills = skill_registry.list_all()
    return {
        "skills": [s.to_dict() for s in skills],
        "count": len(skills)
    }


# ── MCP 管理 API（教程03：MCP协议）───────────────────────────

@app.get("/api/mcp/status")
async def mcp_status_endpoint():
    """
    获取 MCP 连接状态
    
    对应教程03：MCP客户端接入
    返回 MCP Server 是否连接、可用工具数等信息
    
    Returns:
        MCP 状态信息
    """
    return {
        "connected": mcp_status["connected"],
        "tools_count": mcp_status["tools_count"]
    }


@app.get("/api/mcp/tools")
async def mcp_tools_detail():
    """
    获取 MCP 工具详情
    
    对应教程03：MCP工具发现
    返回所有 MCP 工具的详细信息
    
    Returns:
        MCP 工具列表
    """
    tools = get_mcp_tools()
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "args_schema": str(t.args_schema) if hasattr(t, "args_schema") else None
            }
            for t in tools
        ]
    }
