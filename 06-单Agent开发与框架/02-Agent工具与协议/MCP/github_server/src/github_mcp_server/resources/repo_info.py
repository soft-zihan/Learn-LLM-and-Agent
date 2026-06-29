"""
仓库信息资源 - 提供 MCP Resource 接口查询仓库信息

资源列表：
- github://repo/{owner}/{repo}: 获取仓库基本信息

所有资源注册到共享的 mcp 实例。
"""

from ..mcp_instance import mcp
from ..client import GitHubClient

# 共享的 GitHub 客户端实例
github = GitHubClient()


@mcp.resource("github://repo/{owner}/{repo}")
async def repo_info(owner: str, repo: str) -> str:
    """获取仓库基本信息

    Args:
        owner: 仓库所有者
        repo: 仓库名称
    """
    try:
        data = await github.get_repo(owner, repo)
        return (
            f"名称: {data['full_name']}\n"
            f"描述: {data.get('description', 'N/A')}\n"
            f"Stars: {data['stargazers_count']} | Forks: {data['forks_count']}\n"
            f"语言: {data.get('language', 'Unknown')}\n"
            f"默认分支: {data.get('default_branch', 'main')}\n"
            f"URL: {data['html_url']}"
        )
    except Exception as e:
        return f"获取仓库信息失败: {e}"
"""仓库信息资源"""

from fastmcp import FastMCP
from ..client import GitHubClient

mcp = FastMCP("github-server")
github = GitHubClient()


@mcp.resource("github://repo/{owner}/{repo}")
async def repo_info(owner: str, repo: str) -> str:
    """获取仓库基本信息

    Args:
        owner: 仓库所有者
        repo: 仓库名称
    """
    try:
        data = await github.get_repo(owner, repo)
        return (
            f"名称: {data['full_name']}\n"
            f"描述: {data.get('description', 'N/A')}\n"
            f"Stars: {data['stargazers_count']} | Forks: {data['forks_count']}\n"
            f"语言: {data.get('language', 'Unknown')}\n"
            f"默认分支: {data.get('default_branch', 'main')}\n"
            f"URL: {data['html_url']}"
        )
    except Exception as e:
        return f"获取仓库信息失败: {e}"
