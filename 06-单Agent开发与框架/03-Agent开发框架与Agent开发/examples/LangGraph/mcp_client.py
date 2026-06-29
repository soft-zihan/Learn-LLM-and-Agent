"""
MCP 客户端接入 - 对应教程 03-MCP客户端接入

MCP（Model Context Protocol）是 Anthropic 提出的标准协议，
用于连接 LLM 和外部工具/资源。本模块实现了 MCP 客户端：

核心功能：
1. 连接 MCP Server（通过 streamable-http 通信）
2. 发现 Server 提供的工具列表
3. 将 MCP 工具桥接为 LangChain StructuredTool
4. 管理 MCP 连接的生命周期

架构说明：
    LangGraph Agent ←→ LangChain StructuredTool ←→ MCP Client ←→ MCP Server
    （决策层）         （工具接口层）              （协议层）     （工具实现层）

传输方式：
    使用 streamable-http（主流方式），MCP Server 作为 HTTP 服务运行，
    Client 通过 HTTP POST 请求与 Server 通信。
    
    相比 stdio 的优势：
    - Server 可以部署在远程机器上
    - 支持负载均衡和水平扩展
    - 更容易监控和调试（标准 HTTP 日志）

教程对应：
- 03-MCP客户端接入：连接 Server、发现工具、桥接工具
- 02-MCP协议详解：streamable-http 传输、JSON-RPC 通信
"""

import os
from typing import Optional

from langchain_core.tools import StructuredTool
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# ── MCP Server 地址配置 ──────────────────────────────────────
# 使用 streamable-http 通信，MCP Server 作为 HTTP 服务运行
# 默认连接本地运行的 hello_world MCP Server

MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "http://localhost:8000/mcp"  # 默认本地 MCP Server 地址
)

# ── 全局 MCP 会话状态 ────────────────────────────────────────
# 应用生命周期内保持连接，避免每次调用都重新连接

_mcp_session: Optional[ClientSession] = None    # MCP 会话对象
_mcp_session_cm = None                          # Session 的 context manager
_mcp_cm = None                                  # streamablehttp_client 的 context manager
_mcp_tools_raw = []                             # 从 Server 发现的原始工具列表


async def init_mcp():
    """
    启动时连接 MCP Server，发现并缓存工具列表（教程03核心）
    
    执行流程：
    1. 配置 streamable-http 连接参数（Server URL）
    2. 建立 HTTP 连接（MCP Server 已在运行）
    3. 创建 MCP ClientSession 并初始化
    4. 调用 list_tools() 发现 Server 提供的所有工具
    5. 缓存工具列表供后续使用
    
    安全设计：
    - 使用 try/except 确保异常时正确清理资源
    - 保存 context manager 引用以便在异常时调用 __aexit__
    """
    global _mcp_session, _mcp_cm, _mcp_session_cm, _mcp_tools_raw

    server_url = MCP_SERVER_URL
    print(f"[MCP] 连接 Server: {server_url}")

    import asyncio
    try:
        # 超时控制，避免无限等待
        await asyncio.wait_for(_do_init_mcp(server_url), timeout=10.0)
    except asyncio.TimeoutError:
        print("[MCP] 连接超时（10秒），MCP 工具不可用")
        _mcp_tools_raw = []
    except Exception as e:
        print(f"[MCP] 连接失败（可选）: {e}")
        print("[MCP] 应用将继续运行，但 MCP 工具不可用")
        _mcp_tools_raw = []


async def _do_init_mcp(server_url: str):
    """执行 MCP 初始化"""
    global _mcp_session, _mcp_cm, _mcp_session_cm, _mcp_tools_raw

    # 建立 streamable-http 连接（分步进行，确保异常时能正确清理）
    _mcp_cm = streamablehttp_client(server_url)
    try:
        # 步骤1：建立 HTTP 连接
        read, write, _ = await _mcp_cm.__aenter__()

        # 步骤2：创建 MCP 客户端会话
        _mcp_session_cm = ClientSession(read, write)
        _mcp_session = await _mcp_session_cm.__aenter__()
        
        # 步骤3：初始化会话（握手）
        await _mcp_session.initialize()

        # 步骤4：发现工具
        result = await _mcp_session.list_tools()
        _mcp_tools_raw = result.tools
        
        print(f"[MCP] 连接成功，发现 {len(_mcp_tools_raw)} 个工具: "
              f"{[t.name for t in _mcp_tools_raw]}")
    except Exception as e:
        print(f"[MCP] 初始化失败: {e}")
        raise


async def shutdown_mcp():
    """
    关闭时清理 MCP 连接（应用关闭时调用）
    
    清理顺序：
    1. 关闭 ClientSession（发送关闭请求）
    2. 关闭 HTTP 连接
    """
    global _mcp_session, _mcp_cm, _mcp_session_cm
    
    # 先关闭 Session
    if _mcp_session_cm:
        try:
            await _mcp_session_cm.__aexit__(None, None, None)
        except Exception:
            pass
        _mcp_session_cm = None
    
    # 再关闭 HTTP 连接
    if _mcp_cm:
        try:
            await _mcp_cm.__aexit__(None, None, None)
        except Exception:
            pass
        _mcp_cm = None
    
    _mcp_session = None
    _mcp_tools_raw = []


def get_mcp_tools() -> list:
    """
    将 MCP 工具转换为 LangChain StructuredTool（教程03核心）
    
    这是 MCP 和 LangGraph 之间的桥梁：
    - MCP Server 提供的工具使用 MCP 协议格式（JSON Schema 参数定义）
    - LangGraph Agent 需要 LangChain StructuredTool 格式
    - 这个函数完成格式转换
    
    转换过程：
    1. 遍历 MCP 工具列表
    2. 从 JSON Schema 提取参数定义
    3. 动态创建 Pydantic Model 作为参数 schema
    4. 创建 coroutine 函数调用 MCP 工具
    5. 组装为 LangChain StructuredTool
    
    Returns:
        LangChain StructuredTool 列表，可直接传给 Agent 使用
    """
    if not _mcp_session:
        return []

    tools = []
    for mcp_tool in _mcp_tools_raw:
        session = _mcp_session  # 闭包捕获当前会话

        # ── 从 MCP JSON Schema 提取参数定义 ──
        # MCP 工具的 inputSchema 遵循 JSON Schema 规范
        # 例如：{"type": "object", "properties": {"name": {"type": "string"}}}
        input_schema = mcp_tool.inputSchema or {"type": "object", "properties": {}}
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        # ── 创建 coroutine 工厂函数 ──
        # 使用工厂函数避免循环变量捕获问题（Python 闭包陷阱）
        tool_name = mcp_tool.name
        tool_desc = mcp_tool.description or ""

        def make_coro_factory(name):
            """
            工厂函数：为每个 MCP 工具创建独立的 coroutine
            
            为什么需要工厂？
            如果直接在循环中定义 async def，所有工具会共享同一个 name 变量
            （因为 Python 闭包捕获的是变量引用，而非值）
            """
            async def make_coro(**kwargs):
                # 通过 MCP 协议调用 Server 端的工具
                result = await session.call_tool(name, kwargs)
                # 提取文本内容
                return "\n".join(c.text for c in result.content if hasattr(c, "text"))
            return make_coro

        coro = make_coro_factory(tool_name)

        # ── 动态构建 Pydantic Schema ──
        # LangChain 的 StructuredTool 需要一个 Pydantic Model 来验证参数
        # 这里根据 MCP 的 JSON Schema 动态创建
        from pydantic import create_model, Field
        from typing import Optional as Opt
        
        field_defs = {}
        for prop_name, prop_def in properties.items():
            # JSON Schema 类型 → Python 类型映射
            prop_type = prop_def.get("type", "string")
            python_type = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool
            }.get(prop_type, str)
            
            is_required = prop_name in required
            field_defs[prop_name] = (
                python_type if is_required else Opt[python_type],
                Field(default=None, description=prop_def.get("description", ""))
            )

        # 动态创建 Pydantic Model（类名格式：{tool_name}_Input）
        Model = create_model(f"{tool_name}_Input", **field_defs)

        # ── 组装 StructuredTool ──
        tool = StructuredTool(
            name=tool_name,
            description=tool_desc,
            args_schema=Model,
            func=lambda **kwargs: None,  # 占位符，实际使用 coroutine
            coroutine=coro,              # 异步调用函数
        )
        tools.append(tool)

    return tools


def list_mcp_tools_info() -> list:
    """
    列出 MCP 工具信息（用于 API 返回）
    
    Returns:
        工具信息列表，每项包含 name 和 description
    """
    if not _mcp_session:
        return []

    return [
        {
            "name": t.name,
            "description": t.description,
        }
        for t in _mcp_tools_raw
    ]
