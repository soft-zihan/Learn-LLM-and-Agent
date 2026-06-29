"""
Issue 管理工具 - 提供 Issue 的查询、创建等功能

工具列表：
- list_issues: 列出仓库的 Issues
- get_issue_detail: 获取单个 Issue 详情
- create_issue: 创建新 Issue（需要 Token）

所有工具注册到共享的 mcp 实例。
"""

from typing import Optional
from ..mcp_instance import mcp
from ..client import GitHubClient

# 共享的 GitHub 客户端实例
github = GitHubClient()


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def list_issues(owner: str, repo: str, state: str = "open", limit: int = 10) -> str:
    """列出 Issues

    Args:
        owner: 仓库所有者
        repo: 仓库名称
        state: 状态 (open/closed/all)
        limit: 返回结果数（1-100）
    """
    try:
        issues = await github.list_issues(owner, repo, state, limit)
        lines = [f"**{owner}/{repo}** Issues ({state}):\n"]
        for issue in issues[:limit]:
            labels = ", ".join(l["name"] for l in issue.get("labels", []))
            lines.append(
                f"- **#{issue['number']}** {issue['title']}\n"
                f"  状态: {issue['state']} | 评论: {issue['comments']}\n"
                f"  标签: {labels or '无'}\n"
                f"  URL: {issue['html_url']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"列出 Issues 失败: {e}"


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def get_issue_detail(owner: str, repo: str, issue_number: int) -> str:
    """获取 Issue 详情

    Args:
        owner: 仓库所有者
        repo: 仓库名称
        issue_number: Issue 编号（必须 > 0）
    """
    try:
        issue = await github.get_issue(owner, repo, issue_number)
        labels = ", ".join(l["name"] for l in issue.get("labels", []))
        return (
            f"# #{issue['number']} - {issue['title']}\n\n"
            f"**状态**: {issue['state']} | **创建者**: {issue['user']['login']}\n"
            f"**标签**: {labels or '无'}\n\n"
            f"{issue.get('body', 'No description')}\n\n"
            f"URL: {issue['html_url']}"
        )
    except Exception as e:
        return f"获取 Issue 详情失败: {e}"


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})
async def create_issue(
    owner: str, repo: str, title: str,
    body: Optional[str] = None, labels: Optional[str] = None,
) -> str:
    """创建 Issue（需要 GitHub Token）

    Args:
        owner: 仓库所有者
        repo: 仓库名称
        title: 标题（不能为空）
        body: 内容（Markdown 格式）
        labels: 标签（逗号分隔）
    """
    try:
        label_list = [l.strip() for l in labels.split(",")] if labels else None
        issue = await github.create_issue(owner, repo, title, body, label_list)
        return f"Issue 创建成功! **#{issue['number']}** - {issue['title']}\nURL: {issue['html_url']}"
    except Exception as e:
        return f"创建 Issue 失败: {e}"
"""Issue 管理工具"""

from typing import Optional
from fastmcp import FastMCP
from ..client import GitHubClient

mcp = FastMCP("github-server")
github = GitHubClient()


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def list_issues(owner: str, repo: str, state: str = "open", limit: int = 10) -> str:
    """列出 Issues

    Args:
        owner: 仓库所有者
        repo: 仓库名称
        state: 状态 (open/closed/all)
        limit: 返回结果数
    """
    try:
        issues = await github.list_issues(owner, repo, state, limit)
        lines = [f"**{owner}/{repo}** Issues ({state}):\n"]
        for issue in issues[:limit]:
            labels = ", ".join(l["name"] for l in issue.get("labels", []))
            lines.append(
                f"- **#{issue['number']}** {issue['title']}\n"
                f"  状态: {issue['state']} | 评论: {issue['comments']}\n"
                f"  标签: {labels or '无'}\n"
                f"  URL: {issue['html_url']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"列出 Issues 失败: {e}"


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
async def get_issue_detail(owner: str, repo: str, issue_number: int) -> str:
    """获取 Issue 详情

    Args:
        owner: 仓库所有者
        repo: 仓库名称
        issue_number: Issue 编号
    """
    try:
        issue = await github.get_issue(owner, repo, issue_number)
        labels = ", ".join(l["name"] for l in issue.get("labels", []))
        return (
            f"# #{issue['number']} - {issue['title']}\n\n"
            f"**状态**: {issue['state']} | **创建者**: {issue['user']['login']}\n"
            f"**标签**: {labels or '无'}\n\n"
            f"{issue.get('body', 'No description')}\n\n"
            f"URL: {issue['html_url']}"
        )
    except Exception as e:
        return f"获取 Issue 详情失败: {e}"


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})
async def create_issue(
    owner: str, repo: str, title: str,
    body: Optional[str] = None, labels: Optional[str] = None,
) -> str:
    """创建 Issue

    Args:
        owner: 仓库所有者
        repo: 仓库名称
        title: 标题
        body: 内容
        labels: 标签（逗号分隔）
    """
    try:
        label_list = [l.strip() for l in labels.split(",")] if labels else None
        issue = await github.create_issue(owner, repo, title, body, label_list)
        return f"Issue 创建成功! **#{issue['number']}** - {issue['title']}\nURL: {issue['html_url']}"
    except Exception as e:
        return f"创建 Issue 失败: {e}"
