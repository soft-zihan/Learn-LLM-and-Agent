 # MCP Server 开发实战

> 📅 **更新时间**: 2026-06-17  
> 📌 **规范基准**: MCP 2026-07-28 Release Candidate（当前最新）  
> 🏷️ **标记说明**:
> - `📌 MCP规范` = MCP 2025-11-25 及以前版本已确立的稳定规范
> - `🆕 2026-07-28` = 2026-07-28 RC 新增特性（尚未正式启用）
> - `⚠️ 已弃用` = 不推荐继续使用

---

## 目录

- [1. MCP Server 开发（Python）](#1-mcp-server-开发python)
- [2. MCP Server 开发（TypeScript）](#2-mcp-server-开发typescript)
- [3. MCP Server 开发（Go）](#3-mcp-server-开发go)
- [4. MCP Client 集成](#4-mcp-client-集成)
- [5. MCP 2026-07-28 新特性速查](#5-mcp-2026-07-28-新特性速查)

---

## 1. MCP Server 开发（Python）

### 1.1 环境搭建

#### 安装 MCP SDK

```bash
# 使用 pip
pip install mcp>=1.10.0

# 使用 uv（推荐,更快）
pip install uv
uv pip install mcp>=1.10.0

# 使用 poetry
poetry add "mcp>=1.10.0"

# 使用 pdm
pdm add mcp>=1.10.0
```

> **注意**: MCP SDK 1.10.0+ 支持 2026-07-28 规范。本文档所有代码示例均以 **2026-07-28** 规范为基准，新增特性用 `🆕` 标注。

#### 安装 FastMCP（推荐的高级开发框架）

`fastmcp` 是 MCP 协议的**高级封装库**，提供装饰器、自动 Schema 生成等特性，大幅简化开发：

```bash
# 安装 FastMCP（独立包）
pip install fastmcp

# 或使用 uv
uv pip install fastmcp
```

**`mcp` vs `fastmcp` 的区别**：

| 包 | 层级 | 用途 |
|------|------|------|
| `mcp` | 底层协议实现 | 手动操作 TextContent/Tool 类型、处理 JSON-RPC 消息 |
| `fastmcp` | 高级开发框架 | 装饰器注册、自动 Schema 生成、返回值自动封装 |

**推荐**：新项目直接使用 `fastmcp`，无需手动处理底层协议细节。

#### 验证安装

```python
# 测试安装
python -c "import mcp; print(mcp.__version__)"

# 应该输出版本号
```

#### 项目结构

```
my-mcp-server/
├── pyproject.toml              # 项目配置
├── src/
│   └── my_mcp_server/
│       ├── __init__.py
│       ├── server.py           # Server 主文件
│       ├── tools/              # 工具模块
│       │   ├── __init__.py
│       │   ├── search.py
│       │   └── analyze.py
│       ├── resources/          # 资源模块
│       │   ├── __init__.py
│       │   └── config.py
│       └── prompts/            # 提示模块
│           ├── __init__.py
│           └── review.py
├── tests/                      # 测试
│   ├── test_tools.py
│   └── test_resources.py
└── README.md
```

**pyproject.toml**：

```toml
[project]
name = "my-mcp-server"
version = "0.1.0"
description = "My MCP Server"
requires-python = ">=3.10"
dependencies = [
    "mcp>=1.0.0",
]

[project.scripts]
my-server = "my_mcp_server.server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 1.2 第一个 MCP Server

#### Hello World Server

```python
"""
第一个 MCP Server - Hello World
"""
# 📌 MCP规范: 使用 FastMCP 高级 API 简化开发
# 从 fastmcp 包导入（推荐）
from fastmcp import FastMCP
# 旧版也可从 mcp.server.fastmcp 导入，但新版已拆分
# from mcp.server.fastmcp import FastMCP

# 📌 MCP规范: 创建 Server 实例
mcp = FastMCP(
    "hello-world",                    # Server 名称
    instructions="这是一个示例 MCP Server",  # 使用说明
    # 🆕 2026-07-28: 显式声明协议版本
    protocol_version="2026-07-28"
)

# 📌 MCP规范: 定义工具
@mcp.tool(
    # 🆕 2026-07-28: Tool Annotations — 描述工具行为特征
    annotations={
        "readOnlyHint": True,      # 不修改环境
        "openWorldHint": False,    # 不与外部实体交互
        "idempotentHint": True,    # 幂等：多次调用结果相同
    }
)
def greet(name: str) -> str:
    """
    向某人问好
    
    Args:
        name: 要问候的人名
    """
    return f"Hello, {name}! 👋"

# 定义另一个工具
@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    }
)
def add(a: int, b: int) -> int:
    """
    两个数相加
    
    Args:
        a: 第一个数
        b: 第二个数
    """
    return a + b

# 📌 MCP规范: 定义资源
@mcp.resource("info://about")
def get_about() -> str:
    """关于信息"""
    return """
    # Hello World MCP Server
    
    这是一个简单的示例服务器，提供：
    - greet 工具：向某人问好
    - add 工具：计算两个数的和
    - info://about 资源：关于信息
    """

# 📌 MCP规范: 运行服务器
if __name__ == "__main__":
    # Stdio 传输（默认）
    mcp.run()
    
    # 或者显式指定
    # mcp.run(transport="stdio")
```

#### FastMCP API 参考

以下是 `FastMCP` 类的完整 API 说明：

**FastMCP 构造函数**：

```python
class FastMCP:
    def __init__(
        self,
        name: str,                          # 必须：Server 标识符
        instructions: str | None = None,    # 可选：向客户端描述自身用途
        protocol_version: str | None = None,# 可选：显式声明协议版本（2.0+ 才支持）
        extensions: list[str] | None = None # 可选：启用的扩展列表
    ) -> None:
        ...
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | 必须 | Server 标识符，用于客户端识别 |
| `instructions` | `str \| None` | `None` | 向客户端描述 Server 用途，帮助 LLM 理解何时调用 |
| `protocol_version` | `str \| None` | `None` | 显式声明协议版本（如 `"2026-07-28"`），当前稳定版暂不支持 |
| `extensions` | `list[str] \| None` | `None` | 启用的扩展列表，如 `["io.modelcontextprotocol/extensions.tasks"]` |

**@mcp.tool() 装饰器**：

```python
@mcp.tool(
    name: str | None = None,           # 工具名称（默认用函数名）
    annotations: dict | None = None,   # 工具行为注解
    timeout: float | None = None,      # 超时时间（秒）
    output_schema: dict | None = None, # 输出 JSON Schema
    input_schema: dict | None = None,  # 手动指定输入 Schema（覆盖自动生成）
)
def tool_function(...) -> Any:
    ...
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str \| None` | 函数名 | 工具名称，用于客户端调用 |
| `annotations` | `dict \| None` | `None` | 工具行为注解（见下表） |
| `timeout` | `float \| None` | `None` | 超时时间（秒），超时后自动中止 |
| `output_schema` | `dict \| None` | `None` | 输出 JSON Schema，启用 `structuredContent` |
| `input_schema` | `dict \| None` | `None` | 手动指定输入 Schema，覆盖从函数签名自动生成 |

**Tool Annotations 字段**：

```python
annotations = {
    "title": str,            # 可读标题
    "readOnlyHint": bool,    # True = 不修改环境（只读操作）
    "destructiveHint": bool, # True = 可能执行破坏性操作
    "idempotentHint": bool,  # True = 幂等，多次调用结果相同
    "openWorldHint": bool,   # True = 与外部实体交互（网络 API/数据库）
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `title` | `str` | 可读标题，用于 UI 展示 |
| `readOnlyHint` | `bool` | `True` 表示工具不修改环境 |
| `destructiveHint` | `bool` | `True` 表示可能执行破坏性操作 |
| `idempotentHint` | `bool` | `True` 表示幂等，多次调用结果相同 |
| `openWorldHint` | `bool` | `True` 表示与外部实体交互 |

**返回值规则**：

| 返回类型 | 封装结果 | 说明 |
|---------|---------|------|
| `str` | `TextContent(type="text", text=...)` | 纯文本输出 |
| `dict` | `structuredContent` + `TextContent` | 结构化输出（需配合 `output_schema`） |
| `list` | `TextContent`（每个元素） | 多个内容块 |
| 抛异常 | `isError: true` | 自动设置错误标志，不应手动返回错误字符串 |

**重要**：不应手动构造 `[TextContent(...)]` — 这是低级 SDK 用法，FastMCP 会自动封装。

**@mcp.resource() 装饰器**：

```python
@mcp.resource(
    uri: str,                          # 资源 URI（支持模板）
    name: str | None = None,           # 资源名称
    description: str | None = None,    # 资源描述
    mime_type: str | None = None,      # MIME 类型
)
def resource_function(...) -> str:
    ...
```

URI 模板语法：
- 静态 URI：`"config://app"` — 固定资源
- 动态 URI：`"file://{path}"` — 参数化资源，从 URI 提取 `path` 参数
- 带查询参数：`"api://users/{user_id}?fields={fields}"`

**@mcp.prompt() 装饰器**：

```python
@mcp.prompt(
    name: str | None = None,           # 提示名称
)
def prompt_function(...) -> list[dict]:
    # 返回消息列表
    return [
        {"role": "system", "content": {"type": "text", "text": "..."}}
    ]
```

**mcp.run() 方法**：

```python
mcp.run(
    transport: str = "stdio",          # 传输方式：stdio / sse / streamable-http
    host: str = "0.0.0.0",             # HTTP 监听地址
    port: int = 8000,                  # HTTP 监听端口
    ...                                # 其他 uvicorn 参数
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `transport` | `str` | `"stdio"` | 传输方式：`stdio`（标准输入/输出）、`sse`（Server-Sent Events，已弃用）、`streamable-http`（推荐） |
| `host` | `str` | `"0.0.0.0"` | HTTP 监听地址（仅 HTTP 传输） |
| `port` | `int` | `8000` | HTTP 监听端口（仅 HTTP 传输） |

#### 测试 Server

**使用 MCP Inspector**：

```bash
# 安装 MCP Inspector
npm install -g @modelcontextprotocol/inspector

# 运行 Inspector
mcp-inspector

# 在浏览器中打开 http://localhost:5173
# 连接到你的 Server 并测试
```

**使用 Claude Desktop**：

```json
// 配置 Claude Desktop
// macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
// Windows: %APPDATA%\Claude\claude_desktop_config.json

{
  "mcpServers": {
    "hello-world": {
      "command": "python",
      "args": ["/path/to/your/server.py"],
      "env": {}
    }
  }
}
```

**编程测试**：

```python
"""
测试 MCP Server
"""
import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

async def test_server():
    # 配置 Server
    server_params = StdioServerParameters(
        command="python",
        args=["/path/to/server.py"]
    )
    
    # 连接并测试
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化
            await session.initialize()
            
            # 获取工具列表
            tools = await session.list_tools()
            print("可用工具:", [t.name for t in tools.tools])
            
            # 调用 greet 工具
            result = await session.call_tool("greet", {"name": "World"})
            print("greet 结果:", result.content[0].text)
            
            # 调用 add 工具
            result = await session.call_tool("add", {"a": 3, "b": 4})
            print("add 结果:", result.content[0].text)
            
            # 读取资源
            resource = await session.read_resource("info://about")
            print("资源内容:", resource.contents[0].text)

# 运行测试
asyncio.run(test_server())
```

### 1.2.1 🆕 2026-07-28 规范关键特性速览

> 以下特性均已集成到前面的示例中，此处集中说明以便快速掌握。

#### Tool Annotations — 工具行为注解

MCP 2026-07-28 规范要求工具声明行为特征，帮助客户端理解工具性质：

```python
@mcp.tool(
    annotations={
        "title": "搜索产品",               # 可读标题
        "readOnlyHint": True,               # 不修改环境
        "destructiveHint": False,         # 不执行破坏性操作
        "idempotentHint": True,             # 幂等
        "openWorldHint": True,              # 与外部实体交互（如数据库/API）
    }
)
def search_products(query: str) -> list[dict]:
    """搜索产品目录"""
    return db.search(query)
```

#### Output Schema — 输出结构化

工具可以声明输出 schema，返回 `structuredContent` 而非纯文本：

```python
@mcp.tool(
    output_schema={
        "type": "object",
        "properties": {
            "temperature": {"type": "number", "description": "温度（°C）"},
            "conditions": {"type": "string", "description": "天气状况"},
        },
        "required": ["temperature", "conditions"]
    }
)
def get_weather(city: str) -> dict:
    """获取天气信息"""
    # 🆕 返回 dict，FastMCP 自动生成 structuredContent + TextContent
    return {"temperature": 22.5, "conditions": "晴"}
```

#### isError 错误处理

工具执行失败时应使用 `isError: true`，而非返回普通文本：

```python
@mcp.tool()
def fetch_data(url: str) -> str:
    """获取远程数据"""
    try:
        response = httpx.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        # 🆕 抛异常，FastMCP 自动设置 isError: true
        raise ValueError(f"获取数据失败: {e}")
```

#### 无状态设计（2026-07-28 核心变化）

2026-07-28 规范移除了 `initialize` 握手和 `Mcp-Session-Id`，协议完全无状态。任何请求可落到任何服务器实例，水平扩展更简单。服务器不应依赖会话状态，所有必要信息通过工具参数显式传递。

#### 超时控制

```python
@mcp.tool(timeout=30.0)  # 🆕 工具级超时（秒）
def slow_operation(data: str) -> str:
    """耗时操作，30秒超时"""
    return process(data)
```

#### Resource Annotations

资源也可以声明注解，帮助客户端决定如何展示资源：

```python
from mcp.types import Resource

# 🆕 2026-07-28: Resource Annotations
Resource(
    uri="config://app",
    name="App Config",
    annotations={
        "audience": ["assistant"],  # 目标受众：assistant / user
        "priority": 0.8,           # 优先级（0-1）
    }
)
```

#### 其他新特性

| 特性 | 说明 | 详见 |
|------|------|------|
| Tasks 扩展 | 长时间运行操作的进度跟踪 | 5.2 节 |
| MCP Apps | 服务器发送交互式 HTML 界面 | 5.3 节 |
| JSON Schema 2020-12 | 完整的 oneOf/$ref 支持 | 5.4 节 |
| 缓存策略 | ttlMs + cacheScope | 5.5 节 |
| InputRequiredResult | 多轮交互，要求客户端补充输入 | 5.6 节 |
| W3C Trace Context | 分布式追踪传播 | 5.7 节 |

---

### 1.3 资源定义

#### 基础资源

```python
from mcp.server.fastmcp import FastMCP
import json

mcp = FastMCP("resource-server")

# 静态资源
@mcp.resource("config://app")
def get_app_config() -> str:
    """应用配置"""
    config = {
        "appName": "My App",
        "version": "1.0.0",
        "features": ["search", "analyze", "export"],
        "limits": {
            "maxResults": 100,
            "timeout": 30
        }
    }
    return json.dumps(config, indent=2)

# 动态资源 - 文件读取
@mcp.resource("file://{path}")
def read_file(path: str) -> str:
    """
    读取文件内容
    
    Args:
        path: 文件路径
    """
    # 安全检查
    if path.startswith("/etc") or path.startswith("/root"):
        raise ValueError("无权限访问该路径")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f"文件不存在: {path}")

# 带参数的资源
@mcp.resource("api://users/{user_id}?fields={fields}")
def get_user(user_id: str, fields: str = "*") -> str:
    """
    获取用户信息
    
    Args:
        user_id: 用户 ID
        fields: 字段列表（逗号分隔）
    """
    # 模拟数据库查询
    user = {
        "id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com",
        "role": "admin" if user_id == "1" else "user"
    }
    
    # 字段过滤
    if fields != "*":
        field_list = [f.strip() for f in fields.split(",")]
        user = {k: v for k, v in user.items() if k in field_list}
    
    return json.dumps(user, indent=2)
```

#### 复杂资源

```python
from datetime import datetime
from typing import Optional

@mcp.resource("logs://app?level={level}&limit={limit}")
def get_app_logs(level: str = "INFO", limit: str = "100") -> str:
    """
    获取应用日志
    
    Args:
        level: 日志级别（DEBUG/INFO/WARNING/ERROR）
        limit: 返回条数
    """
    # 模拟日志数据
    logs = [
        {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "message": "Application started"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "level": "WARNING",
            "message": "High memory usage detected"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "message": "Failed to connect to database"
        }
    ]
    
    # 过滤和限制
    level_order = ["DEBUG", "INFO", "WARNING", "ERROR"]
    min_level_idx = level_order.index(level) if level in level_order else 0
    
    filtered_logs = [
        log for log in logs
        if level_order.index(log["level"]) >= min_level_idx
    ]
    
    return json.dumps(filtered_logs[:int(limit)], indent=2)

@mcp.resource("metrics://system")
def get_system_metrics() -> str:
    """系统指标"""
    import psutil
    
    metrics = {
        "cpu": {
            "percent": psutil.cpu_percent(interval=1),
            "count": psutil.cpu_count()
        },
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent
        },
        "disk": {
            "total": psutil.disk_usage('/').total,
            "used": psutil.disk_usage('/').used,
            "free": psutil.disk_usage('/').free,
            "percent": psutil.disk_usage('/').percent
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return json.dumps(metrics, indent=2)
```

#### 资源列表

```python
# 显式定义资源列表（高级用法）
from mcp.server.fastmcp import FastMCP
from mcp.types import Resource

mcp = FastMCP("advanced-server")

@mcp.list_resources()
async def list_resources() -> list[Resource]:
    """列出所有资源"""
    return [
        Resource(
            uri="config://app",
            name="App Config",
            description="Application configuration",
            mimeType="application/json",
            # 🆕 2026-07-28: Resource Annotations
            annotations={
                "audience": ["assistant"],   # 目标受众：assistant / user
                "priority": 0.8,             # 优先级（0-1）
            }
        ),
        Resource(
            uri="file://{path}",
            name="File Reader",
            description="Read file contents",
            mimeType="text/plain",
            annotations={"audience": ["assistant"], "priority": 0.5}
        ),
        Resource(
            uri="metrics://system",
            name="System Metrics",
            description="Real-time system metrics",
            mimeType="application/json",
            annotations={"audience": ["assistant", "user"], "priority": 0.9}
        )
    ]
```

### 1.4 完整示例：GitHub MCP Server

这是一个**完整的 GitHub MCP Server**，支持仓库查询、Issue 管理、PR 操作。

#### 项目结构

```
github-mcp-server/
├── pyproject.toml
├── src/
│   └── github_mcp_server/
│       ├── __init__.py
│       ├── server.py
│       ├── client.py          # GitHub API 客户端
│       ├── tools/
│       │   ├── repos.py       # 仓库工具
│       │   ├── issues.py      # Issue 工具
│       │   └── pulls.py       # PR 工具
│       └── resources/
│           └── repo_info.py   # 仓库资源
└── README.md
```

#### GitHub API 客户端

```python
"""
GitHub API 客户端
"""
import httpx
from typing import Optional
from urllib.parse import urljoin

class GitHubClient:
    """GitHub REST API 客户端"""
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self, token: Optional[str] = None):
        """
        初始化客户端
        
        Args:
            token: GitHub Personal Access Token
        """
        self.token = token
        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers=self._get_headers()
        )
    
    def _get_headers(self) -> dict:
        """获取请求头"""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    async def get_repo(self, owner: str, repo: str) -> dict:
        """获取仓库信息"""
        response = await self.client.get(f"/repos/{owner}/{repo}")
        response.raise_for_status()
        return response.json()
    
    async def list_issues(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        limit: int = 10
    ) -> list[dict]:
        """列出 Issues"""
        response = await self.client.get(
            f"/repos/{owner}/{repo}/issues",
            params={
                "state": state,
                "per_page": limit
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def get_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int
    ) -> dict:
        """获取 Issue 详情"""
        response = await self.client.get(
            f"/repos/{owner}/{repo}/issues/{issue_number}"
        )
        response.raise_for_status()
        return response.json()
    
    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: Optional[str] = None,
        labels: Optional[list[str]] = None
    ) -> dict:
        """创建 Issue"""
        data = {"title": title}
        if body:
            data["body"] = body
        if labels:
            data["labels"] = labels
        
        response = await self.client.post(
            f"/repos/{owner}/{repo}/issues",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    async def list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "open"
    ) -> list[dict]:
        """列出 PRs"""
        response = await self.client.get(
            f"/repos/{owner}/{repo}/pulls",
            params={"state": state}
        )
        response.raise_for_status()
        return response.json()
    
    async def get_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int
    ) -> dict:
        """获取 PR 详情"""
        response = await self.client.get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}"
        )
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """关闭客户端"""
        await self.client.aclose()
```

#### 仓库工具

```python
"""
仓库相关工具
"""
from mcp.server.fastmcp import FastMCP
import json
from .client import GitHubClient

mcp = FastMCP("github-server")

# 创建全局客户端
github = GitHubClient(token=None)  # 从环境变量读取

@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,  # 与 GitHub API 交互
    }
)
async def get_repository(owner: str, repo: str) -> str:
    """
    获取仓库信息
    
    Args:
        owner: 仓库所有者
        repo: 仓库名称
    """
    try:
        repo_data = await github.get_repo(owner, repo)
        
        # 格式化输出
        output = f"""# {repo_data['full_name']}

{repo_data.get('description', 'No description')}

⭐ Stars: {repo_data['stargazers_count']}
🍴 Forks: {repo_data['forks_count']}
👀 Watchers: {repo_data['watchers_count']}

语言：{repo_data.get('language', 'Unknown')}
许可证：{repo_data.get('license', {}).get('spdx_id', 'Unknown') if repo_data.get('license') else 'Unknown'}
创建时间：{repo_data['created_at']}
更新时间：{repo_data['updated_at']}

URL: {repo_data['html_url']}
"""
        return output
    except Exception as e:
        # 🆕 2026-07-28: 异常会被 FastMCP 转换为 isError: true
        return f"获取仓库信息失败：{str(e)}"

@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
    }
)
async def search_repositories(
    query: str,
    sort: str = "stars",
    limit: int = 10
) -> str:
    """
    搜索仓库
    
    Args:
        query: 搜索关键词
        sort: 排序方式（stars/forks/help）
        limit: 返回结果数
    """
    try:
        response = await github.client.get(
            "/search/repositories",
            params={
                "q": query,
                "sort": sort,
                "order": "desc",
                "per_page": limit
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # 格式化输出
        output = f"找到 {data['total_count']} 个仓库\n\n"
        for i, repo in enumerate(data['items'][:limit], 1):
            output += f"""{i}. **{repo['full_name']}**
   ⭐ {repo['stargazers_count']} | 语言：{repo.get('language', 'N/A')}
   {repo.get('description', 'No description')}
   URL: {repo['html_url']}

"""
        
        return output
    except Exception as e:
        return f"搜索仓库失败：{str(e)}"
```

#### Issue 工具

```python
"""
Issue 管理工具
"""
from mcp.server.fastmcp import FastMCP
from typing import Optional

mcp = FastMCP("github-server")

@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
    }
)
async def list_issues(
    owner: str,
    repo: str,
    state: str = "open",
    limit: int = 10
) -> str:
    """
    列出 Issues
    
    Args:
        owner: 仓库所有者
        repo: 仓库名称
        state: 状态（open/closed/all）
        limit: 返回结果数
    """
    try:
        issues = await github.list_issues(owner, repo, state, limit)
        
        output = f"**{owner}/{repo}** 的 Issues ({state}):\n\n"
        for issue in issues[:limit]:
            labels = ", ".join([l['name'] for l in issue.get('labels', [])])
            output += f"""- **#{issue['number']}** {issue['title']}
  状态：{issue['state']} | 评论：{issue['comments']}
  标签：{labels if labels else '无'}
  创建者：{issue['user']['login']}
  URL: {issue['html_url']}

"""
        
        return output
    except Exception as e:
        return f"列出 Issues 失败：{str(e)}"

@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
    }
)
async def get_issue_detail(
    owner: str,
    repo: str,
    issue_number: int
) -> str:
    """
    获取 Issue 详情
    
    Args:
        owner: 仓库所有者
        repo: 仓库名称
        issue_number: Issue 编号
    """
    try:
        issue = await github.get_issue(owner, repo, issue_number)
        
        labels = ", ".join([l['name'] for l in issue.get('labels', [])])
        
        output = f"""# #{issue['number']} - {issue['title']}

**状态**: {issue['state']}
**创建者**: {issue['user']['login']}
**创建时间**: {issue['created_at']}
**评论数**: {issue['comments']}
**标签**: {labels if labels else '无'}


{issue.get('body', 'No description')}

URL: {issue['html_url']}
"""
        return output
    except Exception as e:
        return f"获取 Issue 详情失败：{str(e)}"

@mcp.tool(
    annotations={
        "readOnlyHint": False,     # 创建 Issue 会修改环境
        "destructiveHint": False,  # 非破坏性
        "openWorldHint": True,
    }
)
async def create_issue(
    owner: str,
    repo: str,
    title: str,
    body: Optional[str] = None,
    labels: Optional[str] = None  # 🆕 建议用 list[str] 替代逗号分隔字符串
) -> str:
    """
    创建 Issue
    
    Args:
        owner: 仓库所有者
        repo: 仓库名称
        title: 标题
        body: 内容
        labels: 标签（逗号分隔）
    """
    try:
        label_list = [l.strip() for l in labels.split(",")] if labels else None
        
        issue = await github.create_issue(
            owner,
            repo,
            title,
            body,
            label_list
        )
        
        return f"""Issue 创建成功！

**#{issue['number']}** - {issue['title']}
URL: {issue['html_url']}
"""
    except Exception as e:
        return f"创建 Issue 失败：{str(e)}"
```

#### PR 工具

```python
"""
Pull Request 工具
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("github-server")

@mcp.tool(
    annotations={"readOnlyHint": True, "openWorldHint": True}
)
async def list_pull_requests(
    owner: str,
    repo: str,
    state: str = "open"
) -> str:
    """
    列出 Pull Requests
    
    Args:
        owner: 仓库所有者
        repo: 仓库名称
        state: 状态（open/closed/all）
    """
    try:
        prs = await github.list_pull_requests(owner, repo, state)
        
        output = f"**{owner}/{repo}** 的 Pull Requests ({state}):\n\n"
        for pr in prs[:10]:
            output += f"""- **#{pr['number']}** {pr['title']}
  状态：{pr['state']} | 评论：{pr['comments']}
  作者：{pr['user']['login']}
  分支：{pr['head']['ref']} → {pr['base']['ref']}
  URL: {pr['html_url']}

"""
        
        return output
    except Exception as e:
        return f"列出 PRs 失败：{str(e)}"

@mcp.tool(
    annotations={"readOnlyHint": True, "openWorldHint": True}
)
async def get_pull_request_detail(
    owner: str,
    repo: str,
    pr_number: int
) -> str:
    """
    获取 PR 详情
    
    Args:
        owner: 仓库所有者
        repo: 仓库名称
        pr_number: PR 编号
    """
    try:
        pr = await github.get_pull_request(owner, repo, pr_number)
        
        output = f"""# #{pr['number']} - {pr['title']}

**状态**: {pr['state']}
**作者**: {pr['user']['login']}
**创建时间**: {pr['created_at']}
**评论数**: {pr['comments']}


`{pr['head']['ref']}` → `{pr['base']['ref']}`


{pr.get('body', 'No description')}


+{pr['additions']} -{pr['deletions']}

URL: {pr['html_url']}
"""
        return output
    except Exception as e:
        return f"获取 PR 详情失败：{str(e)}"
```

#### 主入口

```python
"""
GitHub MCP Server - 主入口
"""
import os
import asyncio
from .client import GitHubClient

# 从环境变量读取 token
GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

# 初始化全局客户端
github = GitHubClient(token=GITHUB_TOKEN)

# 导入所有工具模块
from .tools import repos, issues, pulls

def main():
    """主函数"""
    print("Starting GitHub MCP Server...")
    print(f"Token configured: {'Yes' if GITHUB_TOKEN else 'No'}")
    
    # 🆕 2026-07-28: 显式声明协议版本
    mcp.run(transport="stdio", protocol_version="2026-07-28")

if __name__ == "__main__":
    main()
```

#### 配置和使用

**安装**：

```bash
# 克隆项目
git clone https://github.com/your-org/github-mcp-server
cd github-mcp-server

# 安装依赖
pip install -e .

# 或者使用 uv
uv pip install -e .
```

**Claude Desktop 配置**：

```json
{
  "mcpServers": {
    "github": {
      "command": "python",
      "args": ["-m", "github_mcp_server.server"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token-here"
      }
    }
  }
}```

**测试**：

```python
"""
测试 GitHub MCP Server
"""
import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
import os

async def test_github_server():
    # 配置
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "github_mcp_server.server"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 测试获取仓库
            result = await session.call_tool(
                "get_repository",
                {"owner": "modelcontextprotocol", "repo": "python-sdk"}
            )
            print(result.content[0].text)
            
            # 测试搜索
            result = await session.call_tool(
                "search_repositories",
                {"query": "MCP protocol", "limit": 5}
            )
            print(result.content[0].text)
            
            # 测试列出 Issues
            result = await session.call_tool(
                "list_issues",
                {
                    "owner": "modelcontextprotocol",
                    "repo": "python-sdk",
                    "limit": 5
                }
            )
            print(result.content[0].text)

asyncio.run(test_github_server())
```

---

## 2. MCP Server 开发（TypeScript）

### 2.1 环境搭建

#### 安装 SDK

```bash
# 创建项目
mkdir my-mcp-server
cd my-mcp-server
npm init -y

# 安装 MCP SDK
npm install @modelcontextprotocol/sdk

# 安装 Zod（用于参数验证）
npm install zod

# 安装 TypeScript
npm install -D typescript @types/node

# 初始化 TypeScript 配置
npx tsc --init
```

**tsconfig.json**：

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "Node16",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

**package.json**：

```json
{
  "name": "my-mcp-server",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "build": "tsc",
    "start": "node dist/index.js",
    "dev": "tsx src/index.ts"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "^1.0.0",
    "zod": "^3.22.0"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "@types/node": "^20.0.0",
    "tsx": "^4.7.0"
  }
}
```

### 2.2 Server 实现

#### Hello World Server

```typescript
/**
 * Hello World MCP Server - TypeScript
 * 基于 MCP 2026-07-28 规范
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

// 创建 Server
const server = new McpServer({
  name: "hello-world",
  version: "1.0.0"
});

// 定义工具
server.tool(
  "greet",
  "向某人问好",
  {
    name: z.string().describe("要问候的人名")
  },
  async ({ name }) => ({
    content: [
      {
        type: "text",
        text: `Hello, ${name}! 👋`
      }
    ]
  }),
  // 🆕 2026-07-28: Tool Annotations
  {
    annotations: {
      readOnlyHint: true,
      openWorldHint: false,
      idempotentHint: true,
    }
  }
);

// 定义另一个工具
server.tool(
  "add",
  "两个数相加",
  {
    a: z.number().describe("第一个数"),
    b: z.number().describe("第二个数")
  },
  async ({ a, b }) => ({
    content: [
      {
        type: "text",
        text: String(a + b)
      }
    ],
    // 🆕 2026-07-28: structuredContent 支持结构化输出
    structuredContent: { result: a + b }
  }),
  {
    annotations: {
      readOnlyHint: true,
      idempotentHint: true,
    }
  }
);

// 定义资源
server.resource(
  "about",
  "info://about",
  async (uri) => ({
    contents: [
      {
        uri: uri.href,
        mimeType: "text/markdown",
        text: `# Hello World MCP Server

这是一个简单的示例服务器，提供：
- greet 工具：向某人问好
- add 工具：计算两个数的和
- info://about 资源：关于信息
`
      }
    ]
  })
);

// 定义提示
server.prompt(
  "code-review",
  {
    language: z.string().default("typescript")
  },
  ({ language }) => ({
    messages: [
      {
        role: "system",
        content: {
          type: "text",
          text: `你是一位资深的 ${language} 代码审查专家。

请审查以下代码，关注：
1. 代码质量和最佳实践
2. 类型安全性
3. 性能优化
4. 错误处理

请提供具体的改进建议。`
        }
      }
    ]
  })
);

// 运行服务器
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Hello World MCP Server running on stdio");
}

main().catch(console.error);
```

#### 复杂工具示例

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

const server = new McpServer({
  name: "advanced-server",
  version: "1.0.0"
});

// 带可选参数的工具
server.tool(
  "search",
  "搜索信息",
  {
    query: z.string().min(1).describe("搜索关键词"),
    source: z.enum(["web", "docs", "code"])
      .default("web")
      .describe("数据源"),
    limit: z.number()
      .min(1)
      .max(50)
      .default(10)
      .describe("返回结果数"),
    language: z.string()
      .optional()
      .describe("语言过滤器")
  },
  async ({ query, source, limit, language }) => {
    try {
      // 模拟搜索
      const results = await performSearch({
        query,
        source,
        limit,
        language
      });
      
      return {
        content: [
          {
            type: "text",
            text: formatResults(results)
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `搜索失败：${error.message}`
          }
        ],
        isError: true
      };
    }
  }
);

// 异步工具
server.tool(
  "fetch-weather",
  "获取天气信息",
  {
    city: z.string().describe("城市名称"),
    country: z.string().default("CN").describe("国家代码")
  },
  async ({ city, country }) => {
    const response = await fetch(
      `https://api.weather.com/v3/weather?city=${city}&country=${country}`
    );
    
    if (!response.ok) {
      return {
        content: [
          {
            type: "text",
            text: "获取天气信息失败"
          }
        ],
        isError: true
      };
    }
    
    const data = await response.json();
    
    return {
      content: [
        {
          type: "text",
          text: `${city}, ${country} 天气：
温度：${data.temperature}°C
湿度：${data.humidity}%
天气：${data.condition}
风速：${data.windSpeed} km/h`
        }
      ]
    };
  }
);

// 带进度报告的工具
server.tool(
  "process-data",
  "处理大量数据",
  {
    totalItems: z.number().min(1).max(1000)
  },
  async ({ totalItems }, { reportProgress }) => {
    const results = [];
    
    for (let i = 0; i < totalItems; i++) {
      // 处理数据
      const result = await processData(i);
      results.push(result);
      
      // 报告进度
      if (reportProgress) {
        await reportProgress({
          progress: i + 1,
          total: totalItems
        });
      }
    }
    
    return {
      content: [
        {
          type: "text",
          text: `处理完成，共 ${results.length} 个项目`
        }
      ]
    };
  }
);
```

### 2.3 传输层配置

#### Stdio 传输

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new McpServer({
  name: "my-server",
  version: "1.0.0"
});

// ... 定义工具 ...

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Server running on stdio");
}

main();
```

#### ⚠️ HTTP/SSE 传输（已弃用，建议用 Streamable HTTP）

> ⚠️ SSE 传输在 2026-07-28 规范中已标记为弃用，新项目请使用下方的 Streamable HTTP。

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import express from "express";

const server = new McpServer({
  name: "my-server",
  version: "1.0.0"
});

// ... 定义工具 ...

const app = express();

let transport: SSEServerTransport;

// SSE 端点
app.get("/sse", async (req, res) => {
  transport = new SSEServerTransport("/messages", res);
  await server.connect(transport);
  
  console.log("Client connected via SSE");
  
  // 连接关闭时清理
  res.on("close", () => {
    console.log("Client disconnected");
  });
});

// 消息端点
app.post("/messages", async (req, res) => {
  if (transport) {
    await transport.handlePostMessage(req, res);
  } else {
    res.status(503).send("No active connection");
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
```

#### 🆕 Streamable HTTP 传输（2026-07-28 推荐）

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import express from "express";

const server = new McpServer({
  name: "my-server",
  version: "1.0.0"
});

// ... 定义工具 ...

const app = express();
app.use(express.json());

// 🆕 单个端点处理所有通信（替代 SSE + POST 双端点）
app.all("/mcp", async (req, res) => {
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: undefined,  // 无状态模式
  });
  await server.connect(transport);
  await transport.handleRequest(req, res);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}/mcp`);
});
```

优势：
- 单个端点处理所有通信（无需独立的 SSE + POST 端点）
- 原生支持无状态模式（无需粘性会话）
- 更好的负载均衡和 CDN 兼容性

#### 认证配置

```typescript
import express from "express";

const app = express();

// API Key 认证
const API_KEY = process.env.API_KEY;

app.use("/sse", (req, res, next) => {
  const key = req.headers["x-api-key"];
  if (!key || key !== API_KEY) {
    res.status(401).send("Unauthorized");
    return;
  }
  next();
});

// JWT 认证
import jwt from "jsonwebtoken";

app.use("/sse", (req, res, next) => {
  const token = req.headers["authorization"]?.replace("Bearer ", "");
  if (!token) {
    res.status(401).send("No token provided");
    return;
  }
  
  try {
    jwt.verify(token, process.env.JWT_SECRET);
    next();
  } catch (error) {
    res.status(403).send("Invalid token");
  }
});
```

#### CORS 设置

```typescript
import cors from "cors";

app.use(cors({
  origin: ["http://localhost:5173", "https://myapp.com"],
  methods: ["GET", "POST"],
  allowedHeaders: ["Content-Type", "Authorization", "X-API-Key"]
}));
```

---

## 3. MCP Server 开发（Go）

### 3.1 环境搭建

#### 安装 SDK

```bash
# 创建项目
mkdir my-mcp-server
cd my-mcp-server
go mod init my-mcp-server

# 安装 MCP SDK
go get github.com/metoro-io/mcp-golang

# 或使用官方 SDK（如果有）
# go get github.com/modelcontextprotocol/go-sdk
```

**go.mod**：

```go
module my-mcp-server

go 1.21

require (
    github.com/metoro-io/mcp-golang v0.1.0
)
```

### 3.2 Server 实现

#### Hello World Server

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    
    mcp "github.com/metoro-io/mcp-golang"
)

func main() {
    // 创建 Server
    server := mcp.NewServer(&mcp.ServerInfo{
        Name:    "hello-world",
        Version: "1.0.0",
    })
    
    // 注册工具
    // 🆕 2026-07-28: 支持 annotations
    server.RegisterTool("greet", "向某人问好", func(args map[string]interface{}) (interface{}, error) {
        name, ok := args["name"].(string)
        if !ok {
            return nil, fmt.Errorf("name is required and must be a string")
        }
        
        result := fmt.Sprintf("Hello, %s! 👋", name)
        
        return map[string]interface{}{
            "content": []map[string]interface{}{
                {
                    "type": "text",
                    "text": result,
                },
            },
        }, nil
    })
    
    // 注册另一个工具
    // 🆕 2026-07-28: 建议用 RegisterToolWithSchema 明确定义参数和注解
    server.RegisterTool("add", "两个数相加", func(args map[string]interface{}) (interface{}, error) {
        a, ok := args["a"].(float64)
        if !ok {
            return nil, fmt.Errorf("a is required and must be a number")
        }
        
        b, ok := args["b"].(float64)
        if !ok {
            return nil, fmt.Errorf("b is required and must be a number")
        }
        
        result := a + b
        
        return map[string]interface{}{
            "content": []map[string]interface{}{
                {
                    "type": "text",
                    "text": fmt.Sprintf("%.0f", result),
                },
            },
        }, nil
    })
    
    // 注册资源
    server.RegisterResource("info://about", "About", "About this server", func() (interface{}, error) {
        content := `# Hello World MCP Server

这是一个简单的示例服务器。
`
        return map[string]interface{}{
            "contents": []map[string]interface{}{
                {
                    "uri":      "info://about",
                    "mimeType": "text/markdown",
                    "text":     content,
                },
            },
        }, nil
    })
    
    // 运行服务器（Stdio 传输）
    transport := mcp.NewStdioTransport()
    if err := server.Run(transport); err != nil {
        log.Fatal(err)
    }
}
```

#### 使用 JSON Schema 定义参数

```go
package main

import (
    "encoding/json"
    "fmt"
    
    mcp "github.com/metoro-io/mcp-golang"
)

// 定义参数结构
type SearchArgs struct {
    Query    string  `json:"query"`
    Source   string  `json:"source,omitempty"`
    Limit    int     `json:"limit,omitempty"`
    Language *string `json:"language,omitempty"`
}

func main() {
    server := mcp.NewServer(&mcp.ServerInfo{
        Name:    "search-server",
        Version: "1.0.0",
    })
    
    // 使用 JSON Schema 定义参数
    searchSchema := map[string]interface{}{
        "type": "object",
        "properties": map[string]interface{}{
            "query": map[string]interface{}{
                "type":        "string",
                "description": "搜索关键词",
            },
            "source": map[string]interface{}{
                "type":        "string",
                "enum":        []string{"web", "docs", "code"},
                "default":     "web",
                "description": "数据源",
            },
            "limit": map[string]interface{}{
                "type":        "integer",
                "minimum":     1,
                "maximum":     50,
                "default":     10,
                "description": "返回结果数",
            },
            "language": map[string]interface{}{
                "type":        "string",
                "description": "语言过滤器",
            },
        },
        "required": []string{"query"},
    }
    
    server.RegisterToolWithSchema(
        "search",
        "搜索信息",
        searchSchema,
        func(args map[string]interface{}) (interface{}, error) {
            // 解析参数
            var searchArgs SearchArgs
            jsonData, _ := json.Marshal(args)
            json.Unmarshal(jsonData, &searchArgs)
            
            // 验证参数
            if searchArgs.Query == "" {
                return nil, fmt.Errorf("query is required")
            }
            
            if searchArgs.Limit < 1 || searchArgs.Limit > 50 {
                return nil, fmt.Errorf("limit must be between 1 and 50")
            }
            
            // 执行搜索
            results := performSearch(searchArgs)
            
            return map[string]interface{}{
                "content": []map[string]interface{}{
                    {
                        "type": "text",
                        "text": formatResults(results),
                    },
                },
            }, nil
        },
    )
    
    // 运行服务器
    transport := mcp.NewStdioTransport()
    server.Run(transport)
}

func performSearch(args SearchArgs) []map[string]interface{} {
    // 模拟搜索逻辑
    return []map[string]interface{}{
        {
            "title": "Result 1",
            "url":   "https://example.com/1",
        },
    }
}

func formatResults(results []map[string]interface{}) string {
    output := "搜索结果：\n\n"
    for i, result := range results {
        output += fmt.Sprintf("%d. %s\n   %s\n\n", i+1, result["title"], result["url"])
    }
    return output
}
```

#### 错误处理

```go
// 错误处理示例
server.RegisterTool("fetch-data", "获取数据", func(args map[string]interface{}) (interface{}, error) {
    url, ok := args["url"].(string)
    if !ok {
        return nil, fmt.Errorf("url is required")
    }
    
    // 执行 HTTP 请求
    resp, err := http.Get(url)
    if err != nil {
        // 返回工具错误（不是 protocol 错误）
        return map[string]interface{}{
            "content": []map[string]interface{}{
                {
                    "type": "text",
                    "text": fmt.Sprintf("获取数据失败：%v", err),
                },
            },
            "isError": true,
        }, nil
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != 200 {
        return map[string]interface{}{
            "content": []map[string]interface{}{
                {
                    "type": "text",
                    "text": fmt.Sprintf("HTTP 错误：%d", resp.StatusCode),
                },
            },
            "isError": true,
        }, nil
    }
    
    // 处理成功响应
    // ...
})
```

#### 日志记录

```go
import (
    "log"
    "os"
)

// 设置日志
var logger = log.New(os.Stderr, "[MCP] ", log.LstdFlags|log.Lshortfile)

server.RegisterTool("process", "处理数据", func(args map[string]interface{}) (interface{}, error) {
    logger.Println("开始处理数据")
    
    // 处理逻辑
    result := processData()
    
    logger.Printf("处理完成，结果：%v", result)
    
    return map[string]interface{}{
        "content": []map[string]interface{}{
            {
                "type": "text",
                "text": result,
            },
        },
    }, nil
})
```

---

## 4. MCP Client 集成

### 4.1 Claude Desktop 配置

#### 配置文件位置

```
macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
Windows: %APPDATA%\Claude\claude_desktop_config.json
Linux: ~/.config/Claude/claude_desktop_config.json
```

#### 基础配置

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/username/Documents",
        "/Users/username/Projects"
      ]
    },
    "github": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-github"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token-here"
      }
    },
    "database": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://localhost:5432/mydb"
      ]
    }
  }
}
```

#### 自定义 Server 配置

```json
{
  "mcpServers": {
    "my-python-server": {
      "command": "python",
      "args": ["/path/to/server.py"],
      "env": {
        "DEBUG": "true",
        "API_KEY": "your-api-key"
      },
      "disabled": false
    },
    "my-node-server": {
      "command": "node",
      "args": ["/path/to/server.js"],
      "env": {}
    },
    "my-go-server": {
      "command": "/path/to/my-server-binary",
      "args": [],
      "env": {}
    }
  }
}
```

#### 远程 Server 配置

```json
{
  "mcpServers": {
    "remote-server": {
      "url": "http://localhost:3000/sse",
      "headers": {
        "Authorization": "Bearer your-token"
      }
    }
  }
}
```

#### 🆕 Streamable HTTP 配置（2026-07-28 推荐）

> ⚠️ SSE 传输已标记为弃用，Streamable HTTP 是新一代 HTTP 传输。

```json
{
  "mcpServers": {
    "remote-server": {
      "url": "http://localhost:3000/mcp",
      "headers": {
        "Authorization": "Bearer your-token"
      }
    }
  }
}
```

Streamable HTTP 传输相比 SSE 的优势：
- 单个端点处理所有通信（无需独立的 SSE + POST 端点）
- 原生支持无状态模式（无需粘性会话）
- 更好的负载均衡和 CDN 兼容性

### 4.2 Cursor 集成

#### 配置方式

Cursor 支持与 MCP 的集成，配置方式类似：

```json
// .cursor/mcp.json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

#### 使用方式

1. 配置 MCP Servers
2. 重启 Cursor
3. 在 AI 对话中使用工具
4. Cursor 会自动发现可用的 MCP Servers

### 4.3 自定义 Client 开发

#### Python Client

```python
"""
自定义 MCP Client
"""
import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

class MCPClient:
    """MCP Client 封装"""
    
    def __init__(self, command: str, args: list[str] = None, env: dict = None):
        """
        初始化 Client
        
        Args:
            command: Server 命令
            args: 命令参数
            env: 环境变量
        """
        self.server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env or {}
        )
        self.session: ClientSession | None = None
        self._running = False
    
    async def connect(self):
        """连接到 Server"""
        self._stdio_context = stdio_client(self.server_params)
        read, write = await self._stdio_context.__aenter__()
        
        self._session_context = ClientSession(read, write)
        self.session = await self._session_context.__aenter__()
        
        # 初始化
        await self.session.initialize()
        self._running = True
        
        print(f"已连接到 Server")
    
    async def disconnect(self):
        """断开连接"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._stdio_context:
            await self._stdio_context.__aexit__(None, None, None)
        self._running = False
        print("已断开连接")
    
    async def list_tools(self) -> list:
        """获取工具列表"""
        if not self.session:
            raise RuntimeError("Not connected")
        
        result = await self.session.list_tools()
        return result.tools
    
    async def call_tool(self, name: str, args: dict = None) -> str:
        """调用工具"""
        if not self.session:
            raise RuntimeError("Not connected")
        
        result = await self.session.call_tool(name, args or {})
        
        # 提取文本结果
        texts = []
        for content in result.content:
            if content.type == "text":
                texts.append(content.text)
        
        return "\n".join(texts)
    
    async def list_resources(self) -> list:
        """获取资源列表"""
        if not self.session:
            raise RuntimeError("Not connected")
        
        result = await self.session.list_resources()
        return result.resources
    
    async def read_resource(self, uri: str) -> str:
        """读取资源"""
        if not self.session:
            raise RuntimeError("Not connected")
        
        result = await self.session.read_resource(uri)
        return result.contents[0].text

# 使用示例
async def main():
    client = MCPClient(
        command="python",
        args=["server.py"],
        env={"DEBUG": "true"}
    )
    
    try:
        await client.connect()
        
        # 获取工具列表
        tools = await client.list_tools()
        print("可用工具:", [t.name for t in tools])
        
        # 调用工具
        result = await client.call_tool("greet", {"name": "World"})
        print("结果:", result)
        
        # 读取资源
        resource = await client.read_resource("info://about")
        print("资源:", resource)
    
    finally:
        await client.disconnect()

asyncio.run(main())
```

#### TypeScript Client

```typescript
/**
 * 自定义 MCP Client - TypeScript
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

class MCPClient {
  private client: Client;
  private transport: StdioClientTransport | null = null;
  
  constructor(command: string, args: string[] = [], env: Record<string, string> = {}) {
    this.client = new Client(
      {
        name: "my-client",
        version: "1.0.0"
      },
      {
        capabilities: {
          tools: {},
          resources: { subscribe: true }
        }
      }
    );
    
    this.transport = new StdioClientTransport({
      command,
      args,
      env
    });
  }
  
  async connect(): Promise<void> {
    if (!this.transport) {
      throw new Error("Transport not initialized");
    }
    
    await this.client.connect(this.transport);
    console.log("已连接到 Server");
  }
  
  async disconnect(): Promise<void> {
    await this.client.close();
    console.log("已断开连接");
  }
  
  async listTools(): Promise<any[]> {
    const result = await this.client.listTools();
    return result.tools;
  }
  
  async callTool(name: string, args: Record<string, any> = {}): Promise<string> {
    const result = await this.client.callTool({
      name,
      arguments: args
    });
    
    // 提取文本结果
    const texts = result.content
      .filter((c: any) => c.type === "text")
      .map((c: any) => c.text);
    
    return texts.join("\n");
  }
  
  async listResources(): Promise<any[]> {
    const result = await this.client.listResources();
    return result.resources;
  }
  
  async readResource(uri: string): Promise<string> {
    const result = await this.client.readResource({ uri });
    return result.contents[0].text;
  }
}

// 使用示例
async function main() {
  const client = new MCPClient("python", ["server.py"], { DEBUG: "true" });
  
  try {
    await client.connect();
    
    const tools = await client.listTools();
    console.log("可用工具:", tools.map((t: any) => t.name));
    
    const result = await client.callTool("greet", { name: "World" });
    console.log("结果:", result);
  } finally {
    await client.disconnect();
  }
}

main().catch(console.error);
```

#### 错误处理和重试

```python
"""
带错误处理和重试的 Client
"""
import asyncio
from typing import Optional
from mcp.client.session import ClientSession
from mcp.shared.exceptions import McpError

class RobustMCPClient:
    """健壮的 MCP Client"""
    
    def __init__(
        self,
        command: str,
        args: list[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.server_params = StdioServerParameters(
            command=command,
            args=args or []
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session: Optional[ClientSession] = None
    
    async def call_with_retry(
        self,
        func,
        *args,
        **kwargs
    ) -> any:
        """带重试的调用"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except McpError as e:
                last_error = e
                print(f"尝试 {attempt + 1} 失败: {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
            
            except Exception as e:
                last_error = e
                print(f"未知错误: {e}")
                break
        
        raise last_error
    
    async def call_tool_safe(
        self,
        name: str,
        args: dict = None
    ) -> tuple[str, bool]:
        """
        安全调用工具
        
        Returns:
            (结果, 是否成功)
        """
        try:
            result = await self.call_with_retry(
                self.session.call_tool,
                name,
                args or {}
            )
            
            texts = [c.text for c in result.content if c.type == "text"]
            return "\n".join(texts), True
        
        except Exception as e:
            return f"调用失败: {e}", False
```

---

## 5. MCP 2026-07-28 新特性速查

> 基础特性（annotations、output schema、isError、无状态设计）已集成到第 1-4 节的代码示例中，见 1.2.1 节。
> 本节保留**高级特性**的完整示例。

### 5.1 基础特性对照表

| 特性 | 说明 | 已集成位置 |
|------|------|----------|
| `protocol_version` | 显式声明协议版本 | 1.2 Hello World + 1.4 主入口 |
| Tool Annotations | 工具行为注解 | 1.2 + 1.4 所有工具 |
| Resource Annotations | 资源受众/优先级注解 | 1.2.1 + 1.3 资源列表 |
| Output Schema | 结构化输出 | 1.2.1 |
| isError | 工具错误标志 | 1.2.1 |
| 无状态设计 | 移除会话管理 | 1.2.1 |
| 工具超时 | `timeout` 参数 | 1.2.1 |
| Streamable HTTP (Python) | 新一代 HTTP 传输 | 4.1 远程配置 |
| Streamable HTTP (TS) | 新一代 HTTP 传输 | 2.3 传输层配置 |

### 5.2 Tasks 扩展（长时间运行操作）

Tasks 扩展用于处理长时间运行的操作。

#### 实现 Tasks 扩展

```python
from mcp.server.fastmcp import FastMCP
import asyncio

mcp = FastMCP(
    "my-server",
    protocol_version="2026-07-28",
    extensions=["io.modelcontextprotocol/extensions.tasks"]
)

# 存储运行中的任务
running_tasks = {}

@mcp.tool()
async def train_model(dataset: str, epochs: int) -> dict:
    """
    训练模型（长时间运行）
    
    返回 task handle,由客户端驱动
    """
    task_id = generate_task_id()
    
    # 创建任务
    task = asyncio.create_task(
        run_training(task_id, dataset, epochs)
    )
    running_tasks[task_id] = {
        "task": task,
        "status": "running",
        "progress": 0,
        "result": None
    }
    
    # 返回 task handle
    return {
        "content": [
            {
                "type": "text",
                "text": f"Training started for dataset '{dataset}' with {epochs} epochs"
            }
        ],
        "task": {
            "taskId": task_id,
            "status": "running"
        }
    }

async def run_training(task_id: str, dataset: str, epochs: int):
    """实际训练逻辑"""
    try:
        for epoch in range(epochs):
            # 更新进度
            running_tasks[task_id]["progress"] = (epoch + 1) / epochs * 100
            
            # 执行训练步骤
            await train_one_epoch(dataset, epoch)
            
            # 通知客户端更新
            await mcp.notify_task_update(
                task_id,
                {
                    "status": "running",
                    "progress": running_tasks[task_id]["progress"]
                }
            )
        
        # 任务完成
        running_tasks[task_id]["status"] = "completed"
        running_tasks[task_id]["result"] = {"accuracy": 0.95}
        
    except Exception as e:
        running_tasks[task_id]["status"] = "failed"
        running_tasks[task_id]["error"] = str(e)

# Tasks 扩展会自动提供以下方法:
# - tasks/get: 查询任务状态
# - tasks/update: 订阅任务更新
# - tasks/cancel: 取消任务
```

#### 任务状态管理

```python
@mcp.tool()
async def cancel_task(task_id: str, reason: str = "") -> dict:
    """取消任务"""
    if task_id not in running_tasks:
        return {"error": f"Task {task_id} not found"}
    
    task_info = running_tasks[task_id]
    task_info["task"].cancel()
    task_info["status"] = "cancelled"
    
    return {"message": f"Task {task_id} cancelled"}

@mcp.tool()
async def get_task_status(task_id: str) -> dict:
    """获取任务状态"""
    if task_id not in running_tasks:
        return {"error": f"Task {task_id} not found"}
    
    return running_tasks[task_id]
```

**重要变化**:
- `tasks/list` 已移除（无法在无会话情况下安全作用域）
- 任务创建是服务器导向的（服务器决定何时返回 task handle）
- 客户端使用 `tasks/get`、`tasks/update`、`tasks/cancel` 驱动任务

### 5.3 MCP Apps（服务器发送交互式 UI）

MCP Apps 允许服务器发送交互式 HTML 界面，在客户端沙盒 iframe 中渲染。

```python
mcp = FastMCP(
    "my-server",
    protocol_version="2026-07-28",
    extensions=["io.modelcontextprotocol/extensions.apps"]
)

@mcp.app_template("search-results-table")
def search_results_template(data: dict) -> dict:
    """搜索结果表格模板"""
    return {
        "html": "<table>...</table>",  # Jinja2 模板
        "data": data,
        "styles": "table { width: 100%; }"
    }

@mcp.tool()
def search_database(query: str) -> dict:
    results = execute_search(query)
    return {
        "content": [{"type": "text", "text": f"Found {len(results)} results"}],
        "app": {"template": "search-results-table", "data": {"results": results}}
    }
```

安全要求：App 在客户端沙盒 iframe 中运行，所有 UI 操作经过审计和同意路径。

### 5.4 JSON Schema 2020-12 支持

工具 schema 现在支持完整的 JSON Schema 2020-12。

#### 使用 oneOf

```python
from mcp.server.fastmcp import FastMCP
from typing import Union

mcp = FastMCP("my-server", protocol_version="2026-07-28")

@mcp.tool()
def process_input(
    data: Union[str, list[str], dict],
    mode: str = "auto"
) -> str:
    """
    处理多种类型的输入
    
    使用 oneOf 支持多种输入类型
    """
    if isinstance(data, str):
        return process_text(data)
    elif isinstance(data, list):
        return process_list(data)
    elif isinstance(data, dict):
        return process_dict(data)
```

**生成的 Schema**:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "data": {
      "oneOf": [
        {"type": "string"},
        {"type": "array", "items": {"type": "string"}},
        {"type": "object"}
      ]
    },
    "mode": {
      "type": "string",
      "enum": ["auto", "text", "list", "dict"],
      "default": "auto"
    }
  },
  "required": ["data"]
}
```

#### 使用 $ref 和 $defs

```python
@mcp.tool()
def create_user(user: dict, settings: Optional[dict] = None) -> dict:
    """
    创建用户
    
    使用 $ref 引用定义的类型
    """
    # 验证和创建
    return {"id": generate_id(), "user": user, "settings": settings}
```

**手动定义复杂 Schema**:

```python
from mcp.server.fastmcp import FastMCP
from mcp.types import Tool

mcp = FastMCP("my-server")

# 手动定义工具以使用完整 JSON Schema
@mcp.tool(
    input_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "user": {
                "$ref": "#/$defs/User"
            },
            "settings": {
                "$ref": "#/$defs/Settings"
            }
        },
        "required": ["user"],
        "$defs": {
            "User": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                    "age": {"type": "integer", "minimum": 0}
                },
                "required": ["name", "email"]
            },
            "Settings": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string", "enum": ["light", "dark"]},
                    "notifications": {"type": "boolean"}
                }
            }
        }
    }
)
def create_user(user: dict, settings: Optional[dict] = None) -> dict:
    """创建用户,使用完整 JSON Schema 验证"""
    return {"id": generate_id(), "user": user, "settings": settings}
```

#### 输出 Schema

```python
@mcp.tool(
    output_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "user": {"$ref": "#/$defs/User"},
            "created_at": {"type": "string", "format": "date-time"}
        },
        "$defs": {
            "User": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"}
                }
            }
        }
    }
)
def create_user(user: dict) -> dict:
    """创建用户并返回结构化输出"""
    return {
        "id": generate_id(),
        "user": user,
        "created_at": now_iso()
    }
```

**新特性**:
- `structuredContent` 现在可以是任何 JSON 值（不仅限于对象）
- 支持条件（`if`/`then`/`else`）
- 支持组合（`oneOf`、`anyOf`、`allOf`）

### 5.5 缓存策略

服务器可在响应中指定缓存策略：

```python
@mcp.tool()
def list_products(category: str | None = None) -> dict:
    products = get_products(category)
    return {
        "content": [{"type": "text", "text": f"Found {len(products)} products"}],
        "ttlMs": 300000,        # 缓存 5 分钟
        "cacheScope": "user"    # 可跨用户共享
    }
```

| cacheScope | 说明 |
|------------|------|
| `"user"` | 可跨用户共享 |
| `"session"` | 仅当前会话 |
| `"request"` | 不缓存 |

### 5.6 多轮请求（InputRequiredResult）

服务器可在处理请求时要求客户端提供额外输入（如删除确认）：

```python
from mcp.server.fastmcp import FastMCP
from mcp.types import InputRequiredResult

mcp = FastMCP("my-server", protocol_version="2026-07-28")

@mcp.tool()
def delete_files(files: list[str]) -> dict:
    """
    删除文件
    
    需要用户确认
    """
    if len(files) > 1:
        # 返回 InputRequiredResult 要求确认
        return InputRequiredResult(
            resultType="inputRequired",
            inputRequests={
                "confirm": {
                    "type": "elicitation",
                    "message": f"Delete {len(files)} files?",
                    "schema": {"type": "boolean"}
                }
            },
            requestState="eyJzdGVwIjoxLCJmaWxlcyI6WyJhIiwiYiIsImMiXX0="
        )
    
    # 直接删除
    perform_delete(files)
    return {"content": [{"type": "text", "text": "Files deleted"}]}

# 客户端会重新发起请求,携带 inputResponses
@mcp.tool()
def delete_files_with_confirm(
    files: list[str],
    confirm: Optional[bool] = None,
    requestState: Optional[str] = None
) -> dict:
    """删除文件,支持确认"""
    if confirm is None:
        # 首次调用,要求确认
        return InputRequiredResult(
            resultType="inputRequired",
            inputRequests={
                "confirm": {
                    "type": "elicitation",
                    "message": f"Delete {len(files)} files?",
                    "schema": {"type": "boolean"}
                }
            },
            requestState=requestState or encode_state({"files": files})
        )
    
    if not confirm:
        return {"content": [{"type": "text", "text": "Cancelled"}]}
    
    # 用户确认,执行删除
    files = decode_state(requestState)["files"]
    perform_delete(files)
    return {"content": [{"type": "text", "text": "Files deleted"}]}
```

工作流程：服务器返回 `InputRequiredResult` → 客户端收集用户输入 → 客户端重新发起请求（携带 `inputResponses` + `requestState`） → 任何服务器实例可处理重试。

### 5.7 W3C Trace Context 传播

服务器应传播追踪上下文以支持分布式追踪：

```python
@mcp.tool()
async def search_database(query: str, context: RequestContext) -> dict:
    traceparent = context.meta.get("traceparent")
    # 使用追踪上下文调用下游服务
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://search-service/search",
            params={"q": query},
            headers={"traceparent": traceparent}
        )
    return {"content": [{"type": "text", "text": response.text}]}
```

### 5.8 生产部署

无状态服务器可轻松水平扩展，使用普通轮询负载均衡即可（无需 sticky sessions）。Nginx 配置只需在 proxy_pass 后加 `proxy_set_header MCP-Protocol-Version 2026-07-28`。

---

