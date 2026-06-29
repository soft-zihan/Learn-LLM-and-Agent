"""
Pull Request 工具 - 提供 PR 查询功能

工具列表：
- list_pull_requests: 列出 PR
- get_pull_request_detail: 获取 PR 详情

所有工具注册到共享的 mcp 实例。
"""

from ..mcp_instance import mcp
from ..client import GitHubClient

# 共享的 GitHub 客户端实例
github = GitHubClient()


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def list_pull_requests(owner: str, repo: str, state: str = "open") -> str:
    """列出 Pull Requests

    Args:
        owner: 仓库所有者
        repo: 仓库名称
        state: 状态 (open/closed/all)
    """
    try:
        prs = await github.list_pulls(owner, repo, state)
        lines = [f"**{owner}/{repo}** Pull Requests ({state}):\n"]
        for pr in prs[:10]:
            lines.append(
                f"- **#{pr['number']}** {pr['title']}\n"
                f"  状态: {pr['state']} | 作者: {pr['user']['login']}\n"
                f"  分支: {pr['head']['ref']} -> {pr['base']['ref']}\n"
                f"  URL: {pr['html_url']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"列出 PRs 失败: {e}"


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def get_pull_request_detail(owner: str, repo: str, pr_number: int) -> str:
    """获取 PR 详情

    Args:
        owner: 仓库所有者
        repo: 仓库名称
        pr_number: PR 编号（必须 > 0）
    """
    try:
        pr = await github.get_pull(owner, repo, pr_number)
        return (
            f"# #{pr['number']} - {pr['title']}\n\n"
            f"**状态**: {pr['state']} | **作者**: {pr['user']['login']}\n\n"
            f"`{pr['head']['ref']}` -> `{pr['base']['ref']}`\n\n"
            f"{pr.get('body', 'No description')}\n\n"
            f"+{pr['additions']} -{pr['deletions']}\n"
            f"URL: {pr['html_url']}"
        )
    except Exception as e:
        return f"获取 PR 详情失败: {e}"
"""Pull Request 工具"""

from fastmcp import FastMCP
from ..client import GitHubClient

mcp = FastMCP("github-server")
github = GitHubClient()


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def list_pull_requests(owner: str, repo: str, state: str = "open") -> str:
    """列出 Pull Requests

    Args:
        owner: 仓库所有者
        repo: 仓库名称
        state: 状态 (open/closed/all)
    """
    try:
        prs = await github.list_pulls(owner, repo, state)
        lines = [f"**{owner}/{repo}** Pull Requests ({state}):\n"]
        for pr in prs[:10]:
            lines.append(
                f"- **#{pr['number']}** {pr['title']}\n"
                f"  状态: {pr['state']} | 作者: {pr['user']['login']}\n"
                f"  分支: {pr['head']['ref']} -> {pr['base']['ref']}\n"
                f"  URL: {pr['html_url']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"列出 PRs 失败: {e}"


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def get_pull_request_detail(owner: str, repo: str, pr_number: int) -> str:
    """获取 PR 详情

    Args:
        owner: 仓库所有者
        repo: 仓库名称
        pr_number: PR 编号
    """
    try:
        pr = await github.get_pull(owner, repo, pr_number)
        return (
            f"# #{pr['number']} - {pr['title']}\n\n"
            f"**状态**: {pr['state']} | **作者**: {pr['user']['login']}\n\n"
            f"`{pr['head']['ref']}` -> `{pr['base']['ref']}`\n\n"
            f"{pr.get('body', 'No description')}\n\n"
            f"+{pr['additions']} -{pr['deletions']}\n"
            f"URL: {pr['html_url']}"
        )
    except Exception as e:
        return f"获取 PR 详情失败: {e}"
