"""
仓库相关工具 - 提供仓库查询和搜索功能

工具列表：
- get_repository: 获取仓库详情
- search_repositories: 搜索仓库

所有工具注册到 server.py 中共享的 mcp 实例。
"""

from ..mcp_instance import mcp
from ..client import GitHubClient

# 共享的 GitHub 客户端实例
github = GitHubClient()


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def get_repository(owner: str, repo: str) -> str:
    """获取仓库信息

    Args:
        owner: 仓库所有者（用户名或组织名）
        repo: 仓库名称
    """
    try:
        data = await github.get_repo(owner, repo)
        return (
            f"# {data['full_name']}\n\n"
            f"{data.get('description', 'No description')}\n\n"
            f"Stars: {data['stargazers_count']} | Forks: {data['forks_count']}\n"
            f"语言: {data.get('language', 'Unknown')}\n"
            f"URL: {data['html_url']}"
        )
    except Exception as e:
        return f"获取仓库信息失败: {e}"


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def search_repositories(query: str, sort: str = "stars", limit: int = 10) -> str:
    """搜索仓库

    Args:
        query: 搜索关键词
        sort: 排序方式 (stars/forks/help-wanted-issues/recent/updated)
        limit: 返回结果数（1-100）
    """
    try:
        data = await github.search_repos(query, sort, limit)
        lines = [f"找到 {data['total_count']} 个仓库\n"]
        for i, r in enumerate(data["items"][:limit], 1):
            lines.append(
                f"{i}. **{r['full_name']}** - Stars: {r['stargazers_count']}\n"
                f"   {r.get('description', '')}\n"
                f"   URL: {r['html_url']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"搜索仓库失败: {e}"
"""仓库相关工具"""

from fastmcp import FastMCP
from ..client import GitHubClient

mcp = FastMCP("github-server")
github = GitHubClient()


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def get_repository(owner: str, repo: str) -> str:
    """获取仓库信息

    Args:
        owner: 仓库所有者
        repo: 仓库名称
    """
    try:
        data = await github.get_repo(owner, repo)
        return (
            f"# {data['full_name']}\n\n"
            f"{data.get('description', 'No description')}\n\n"
            f"Stars: {data['stargazers_count']} | Forks: {data['forks_count']}\n"
            f"语言: {data.get('language', 'Unknown')}\n"
            f"URL: {data['html_url']}"
        )
    except Exception as e:
        return f"获取仓库信息失败: {e}"


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def search_repositories(query: str, sort: str = "stars", limit: int = 10) -> str:
    """搜索仓库

    Args:
        query: 搜索关键词
        sort: 排序方式 (stars/forks/help)
        limit: 返回结果数
    """
    try:
        data = await github.search_repos(query, sort, limit)
        lines = [f"找到 {data['total_count']} 个仓库\n"]
        for i, r in enumerate(data["items"][:limit], 1):
            lines.append(
                f"{i}. **{r['full_name']}** - Stars: {r['stargazers_count']}\n"
                f"   {r.get('description', '')}\n"
                f"   URL: {r['html_url']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"搜索仓库失败: {e}"
