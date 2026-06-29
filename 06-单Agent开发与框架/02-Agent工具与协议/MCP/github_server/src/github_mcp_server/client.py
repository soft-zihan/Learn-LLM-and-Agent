"""
GitHub API 客户端 - 封装 GitHub REST API 调用

所有工具模块共享这个客户端实例，避免重复创建 HTTP 连接。

使用方式：
    client = GitHubClient()
    repo = await client.get_repo("owner", "repo")

环境变量：
    GITHUB_PERSONAL_ACCESS_TOKEN - GitHub 个人访问令牌（可选，无 Token 受 API 限流）
"""

import os
from typing import Optional

import httpx


class GitHubClient:
    """
    GitHub REST API 客户端
    
    封装了常用的 GitHub API 调用，包括：
    - 仓库查询（get_repo, search_repos）
    - Issue 管理（list_issues, get_issue, create_issue）
    - PR 管理（list_pulls, get_pull）
    
    所有方法都是异步的，使用 httpx 进行 HTTP 请求。
    """
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self, token: Optional[str] = None):
        """
        初始化 GitHub 客户端
        
        Args:
            token: GitHub 个人访问令牌。如果不提供，从环境变量读取。
                   无 Token 时受 GitHub API 限流（60次/小时）。
        """
        self.token = token or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        
        # 构建请求头
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "github-mcp-server",
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"
    
    def _get_client(self) -> httpx.AsyncClient:
        """创建带认证信息的 HTTP 客户端"""
        return httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers=self.headers,
            timeout=30.0,
        )
    
    # ── 仓库相关 ──────────────────────────────────────────────
    
    async def get_repo(self, owner: str, repo: str) -> dict:
        """
        获取仓库信息
        
        Args:
            owner: 仓库所有者（用户名或组织名）
            repo: 仓库名称
        
        Returns:
            仓库信息字典，包含 full_name, description, stargazers_count 等
        
        Raises:
            httpx.HTTPStatusError: API 请求失败时抛出
        """
        async with self._get_client() as client:
            response = await client.get(f"/repos/{owner}/{repo}")
            response.raise_for_status()
            return response.json()
    
    async def search_repos(self, query: str, sort: str = "stars", limit: int = 10) -> dict:
        """
        搜索仓库
        
        Args:
            query: 搜索关键词
            sort: 排序方式（stars/forks/help-wanted-issues/recent/updated）
            limit: 返回结果数量（1-100）
        
        Returns:
            搜索结果字典，包含 total_count 和 items 列表
        """
        # 参数校验
        limit = max(1, min(100, limit))
        valid_sorts = {"stars", "forks", "help-wanted-issues", "recent", "updated"}
        if sort not in valid_sorts:
            sort = "stars"
        
        async with self._get_client() as client:
            response = await client.get(
                "/search/repositories",
                params={"q": query, "sort": sort, "per_page": limit},
            )
            response.raise_for_status()
            return response.json()
    
    # ── Issue 相关 ──────────────────────────────────────────────
    
    async def list_issues(
        self, owner: str, repo: str,
        state: str = "open", limit: int = 10,
    ) -> list:
        """
        列出仓库的 Issues
        
        Args:
            owner: 仓库所有者
            repo: 仓库名称
            state: 过滤状态（open/closed/all）
            limit: 返回数量（1-100）
        
        Returns:
            Issue 列表
        """
        # 参数校验
        limit = max(1, min(100, limit))
        valid_states = {"open", "closed", "all"}
        if state not in valid_states:
            state = "open"
        
        async with self._get_client() as client:
            response = await client.get(
                f"/repos/{owner}/{repo}/issues",
                params={"state": state, "per_page": limit},
            )
            response.raise_for_status()
            return response.json()
    
    async def get_issue(self, owner: str, repo: str, issue_number: int) -> dict:
        """
        获取单个 Issue 详情
        
        Args:
            owner: 仓库所有者
            repo: 仓库名称
            issue_number: Issue 编号（必须 > 0）
        
        Returns:
            Issue 详情字典
        """
        if issue_number <= 0:
            raise ValueError(f"Issue 编号必须大于 0，收到: {issue_number}")
        
        async with self._get_client() as client:
            response = await client.get(f"/repos/{owner}/{repo}/issues/{issue_number}")
            response.raise_for_status()
            return response.json()
    
    async def create_issue(
        self, owner: str, repo: str,
        title: str, body: Optional[str] = None,
        labels: Optional[list] = None,
    ) -> dict:
        """
        创建 Issue（需要 Token 认证）
        
        Args:
            owner: 仓库所有者
            repo: 仓库名称
            title: Issue 标题
            body: Issue 内容（Markdown 格式）
            labels: 标签列表
        
        Returns:
            创建的 Issue 详情
        """
        if not self.token:
            raise PermissionError("创建 Issue 需要 GitHub Token，请设置 GITHUB_PERSONAL_ACCESS_TOKEN")
        
        if not title or not title.strip():
            raise ValueError("Issue 标题不能为空")
        
        payload = {"title": title}
        if body:
            payload["body"] = body
        if labels:
            payload["labels"] = labels
        
        async with self._get_client() as client:
            response = await client.post(
                f"/repos/{owner}/{repo}/issues",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
    
    # ── PR 相关 ──────────────────────────────────────────────
    
    async def list_pulls(
        self, owner: str, repo: str,
        state: str = "open", limit: int = 10,
    ) -> list:
        """
        列出 Pull Requests
        
        Args:
            owner: 仓库所有者
            repo: 仓库名称
            state: 过滤状态（open/closed/all）
            limit: 返回数量（1-100）
        
        Returns:
            PR 列表
        """
        limit = max(1, min(100, limit))
        valid_states = {"open", "closed", "all"}
        if state not in valid_states:
            state = "open"
        
        async with self._get_client() as client:
            response = await client.get(
                f"/repos/{owner}/{repo}/pulls",
                params={"state": state, "per_page": limit},
            )
            response.raise_for_status()
            return response.json()
    
    async def get_pull(self, owner: str, repo: str, pr_number: int) -> dict:
        """
        获取单个 PR 详情
        
        Args:
            owner: 仓库所有者
            repo: 仓库名称
            pr_number: PR 编号（必须 > 0）
        
        Returns:
            PR 详情字典
        """
        if pr_number <= 0:
            raise ValueError(f"PR 编号必须大于 0，收到: {pr_number}")
        
        async with self._get_client() as client:
            response = await client.get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
            response.raise_for_status()
            return response.json()
