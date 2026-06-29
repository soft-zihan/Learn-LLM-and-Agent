"""
共享 MCP 实例 - 避免循环导入

所有工具模块和资源模块都从这里导入 mcp 实例，
确保注册到同一个 FastMCP Server 上。

架构：
    mcp_instance.py  →  创建 FastMCP 实例
         ↑
    ├── tools/repos.py   （导入 mcp，注册工具）
    ├── tools/issues.py  （导入 mcp，注册工具）
    ├── tools/pulls.py   （导入 mcp，注册工具）
    └── resources/repo_info.py （导入 mcp，注册资源）
         ↑
    server.py  →  导入工具/资源模块（触发注册），然后运行 mcp
"""

from fastmcp import FastMCP

# 全局共享的 MCP Server 实例
# 所有工具和资源都注册到这个实例上
mcp = FastMCP(
    "github-mcp-server",
    instructions="GitHub MCP Server，提供仓库查询、Issue 管理、PR 操作等工具",
)
