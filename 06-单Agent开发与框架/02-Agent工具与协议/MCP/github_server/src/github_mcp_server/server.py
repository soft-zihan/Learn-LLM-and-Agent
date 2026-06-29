"""
GitHub MCP Server - 主入口

本模块是 GitHub MCP Server 的启动入口，提供以下功能：
1. 仓库查询（get_repository, search_repositories）
2. Issue 管理（list_issues, get_issue_detail, create_issue）
3. PR 操作（list_pull_requests, get_pull_request_detail）
4. 仓库资源（github://repo/{owner}/{repo}）

运行方式:
    # 设置 Token（可选，无 Token 时受 API 限流 60次/小时）
    export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxx

    # 启动（stdio 模式，用于本地集成）
    python -m github_mcp_server.server

    # 启动（HTTP 模式，用于远程部署）
    python -m github_mcp_server.server --transport streamable-http --port 8000

架构说明:
    server.py          → 启动入口，解析命令行参数
    mcp_instance.py    → 创建共享的 FastMCP 实例
    client.py          → GitHub API 客户端封装
    tools/repos.py     → 仓库相关工具
    tools/issues.py    → Issue 管理工具
    tools/pulls.py     → PR 操作工具
    resources/repo_info.py → 仓库信息资源
"""

import os
import sys

# 确保包路径正确
sys.path.insert(0, os.path.dirname(__file__))

from .mcp_instance import mcp

# 从环境变量读取 Token
GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


def register_tools():
    """
    注册所有工具模块
    
    导入工具模块会触发 @mcp.tool 装饰器执行，
    将工具注册到共享的 mcp 实例上。
    """
    from .tools import repos, issues, pulls  # noqa: F401


def register_resources():
    """
    注册所有资源模块
    
    导入资源模块会触发 @mcp.resource 装饰器执行，
    将资源注册到共享的 mcp 实例上。
    """
    from .resources import repo_info  # noqa: F401


def parse_args() -> tuple[str, int]:
    """
    解析命令行参数
    
    支持的参数：
        --transport <stdio|streamable-http>  传输方式（默认 stdio）
        --port <1-65535>                     HTTP 端口（默认 8000）
    
    Returns:
        (transport, port) 元组
    """
    transport = "stdio"
    port = 8000
    
    try:
        if "--transport" in sys.argv:
            idx = sys.argv.index("--transport")
            if idx + 1 < len(sys.argv):
                transport = sys.argv[idx + 1]
                # 校验传输方式
                valid_transports = {"stdio", "streamable-http", "sse"}
                if transport not in valid_transports:
                    print(f"警告: 不支持的传输方式 '{transport}'，使用默认 stdio", file=sys.stderr)
                    transport = "stdio"
            else:
                print("警告: --transport 参数缺少值，使用默认 stdio", file=sys.stderr)
        
        if "--port" in sys.argv:
            idx = sys.argv.index("--port")
            if idx + 1 < len(sys.argv):
                try:
                    port = int(sys.argv[idx + 1])
                    # 端口范围校验
                    if not (1 <= port <= 65535):
                        print(f"警告: 端口 {port} 超出范围(1-65535)，使用默认 8000", file=sys.stderr)
                        port = 8000
                except ValueError:
                    print(f"警告: 端口 '{sys.argv[idx + 1]}' 不是有效数字，使用默认 8000", file=sys.stderr)
            else:
                print("警告: --port 参数缺少值，使用默认 8000", file=sys.stderr)
    except (IndexError, ValueError) as e:
        print(f"警告: 参数解析错误 ({e})，使用默认值", file=sys.stderr)
    
    return transport, port


def main():
    """主函数 - 启动 MCP Server"""
    print(f"Starting GitHub MCP Server... (Token: {'configured' if GITHUB_TOKEN else 'not set'})", file=sys.stderr)

    # 导入工具与资源模块（触发注册）
    register_tools()
    register_resources()

    # 解析命令行参数
    transport, port = parse_args()

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=transport, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
