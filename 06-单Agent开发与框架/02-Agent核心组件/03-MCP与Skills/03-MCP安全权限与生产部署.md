# MCP 安全权限与生产部署

> 📅 **更新时间**: 2026-06-17  

---

## 目录

- [1. 实战项目](#1-实战项目)
- [2. 生态工具](#2-生态工具)
- [3. 前沿技术](#3-前沿技术)
- [4. MCP 2026-07-28 安全与生产部署更新](#4-mcp-2026-07-28-安全与生产部署更新)

---

## 1. 实战项目

### 1.1 数据库查询 MCP Server

完整的数据库查询 MCP Server，支持 SQL 执行、结果格式化、权限控制。

```python
"""
数据库查询 MCP Server
"""
from mcp.server.fastmcp import FastMCP
from databases import Database
from typing import Optional
import json
import re

mcp = FastMCP("database-server")

# 数据库连接
database = Database("postgresql://user:password@localhost/mydb")

# 安全配置
ALLOWED_OPERATIONS = ["SELECT"]
MAX_ROWS = 1000

def validate_sql(sql: str) -> tuple[bool, str]:
    """验证 SQL 语句"""
    # 检查操作类型
    operation = sql.strip().split()[0].upper()
    if operation not in ALLOWED_OPERATIONS:
        return False, f"不允许的操作: {operation}，仅允许: {ALLOWED_OPERATIONS}"
    
    # 检查危险关键字
    dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER", "INSERT", "UPDATE"]
    for keyword in dangerous_keywords:
        if re.search(r'\b' + keyword + r'\b', sql, re.IGNORECASE):
            return False, f"SQL 包含危险关键字: {keyword}"
    
    return True, "OK"

@mcp.tool()
async def execute_query(sql: str, params: Optional[dict] = None) -> str:
    """
    执行只读 SQL 查询
    
    Args:
        sql: SQL 语句（仅支持 SELECT）
        params: 查询参数
    """
    try:
        # 验证 SQL
        valid, message = validate_sql(sql)
        if not valid:
            return f"SQL 验证失败: {message}"
        
        # 执行查询
        async with database:
            if params:
                results = await database.fetch_all(sql, values=params)
            else:
                results = await database.fetch_all(sql)
        
        # 限制行数
        if len(results) > MAX_ROWS:
            results = results[:MAX_ROWS]
        
        # 格式化结果
        output = f"查询成功，返回 {len(results)} 行\n\n"
        output += json.dumps(
            [dict(row) for row in results],
            indent=2,
            default=str
        )
        
        return output
    
    except Exception as e:
        return f"查询失败: {e}"

@mcp.tool()
async def get_table_schema(table_name: str) -> str:
    """
    获取表结构
    
    Args:
        table_name: 表名
    """
    try:
        sql = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = $1
        ORDER BY ordinal_position
        """
        
        async with database:
            columns = await database.fetch_all(sql, values=[table_name])
        
        output = f"表 {table_name} 的结构:\n\n"
        output += "| 列名 | 数据类型 | 可空 |\n"
        output += "|------|----------|------|\n"
        
        for col in columns:
            output += f"| {col['column_name']} | {col['data_type']} | {col['is_nullable']} |\n"
        
        return output
    
    except Exception as e:
        return f"获取表结构失败: {e}"

@mcp.tool()
async def list_tables() -> str:
    """列出所有表"""
    try:
        sql = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
        """
        
        async with database:
            tables = await database.fetch_all(sql)
        
        output = "数据库中的表:\n\n"
        for table in tables:
            output += f"- {table['table_name']}\n"
        
        return output
    
    except Exception as e:
        return f"获取表列表失败: {e}"

@mcp.resource("db://schema/{table_name}")
def get_table_schema_resource(table_name: str) -> str:
    """表结构资源"""
    # 同步版本，用于资源
    return f"表 {table_name} 的结构信息"

# 启动
if __name__ == "__main__":
    mcp.run()
```

### 1.2 文件管理 MCP Server

文件管理 MCP Server，支持文件读写、目录操作、搜索。

```python
"""
文件管理 MCP Server
"""
from mcp.server.fastmcp import FastMCP
from pathlib import Path
from typing import Optional
import os
import json

mcp = FastMCP("file-manager")

# 配置
BASE_DIR = Path("/app/data")

def safe_path(file_path: str) -> Path:
    """安全的路径解析"""
    path = Path(file_path).resolve()
    
    # 确保在允许的目录内
    if not path.startswith(BASE_DIR):
        raise ValueError(f"路径超出允许范围: {BASE_DIR}")
    
    return path

@mcp.tool()
async def read_file(file_path: str) -> str:
    """
    读取文件内容
    
    Args:
        file_path: 文件路径（相对于 BASE_DIR）
    """
    try:
        path = safe_path(file_path)
        
        if not path.exists():
            return f"文件不存在: {file_path}"
        
        if not path.is_file():
            return f"不是文件: {file_path}"
        
        # 读取文件
        content = path.read_text(encoding='utf-8')
        
        return f"文件: {file_path}\n\n{content}"
    
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"读取失败: {e}"

@mcp.tool()
async def write_file(file_path: str, content: str, append: bool = False) -> str:
    """
    写入文件
    
    Args:
        file_path: 文件路径
        content: 内容
        append: 是否追加
    """
    try:
        path = safe_path(file_path)
        
        # 确保父目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入文件
        mode = 'a' if append else 'w'
        with open(path, mode, encoding='utf-8') as f:
            f.write(content)
        
        return f"文件写入成功: {file_path}"
    
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"写入失败: {e}"

@mcp.tool()
async def list_directory(dir_path: str = ".") -> str:
    """
    列出目录内容
    
    Args:
        dir_path: 目录路径
    """
    try:
        path = safe_path(dir_path)
        
        if not path.exists():
            return f"目录不存在: {dir_path}"
        
        if not path.is_dir():
            return f"不是目录: {dir_path}"
        
        # 列出内容
        output = f"目录: {dir_path}\n\n"
        
        for item in sorted(path.iterdir()):
            if item.is_dir():
                output += f"📁 {item.name}/\n"
            else:
                size = item.stat().st_size
                output += f"📄 {item.name} ({size:,} bytes)\n"
        
        return output
    
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"列出失败: {e}"

@mcp.tool()
async def search_files(
    dir_path: str,
    pattern: str,
    file_type: Optional[str] = None
) -> str:
    """
    搜索文件
    
    Args:
        dir_path: 搜索目录
        pattern: 文件名模式（支持通配符）
        file_type: 文件类型过滤
    """
    try:
        path = safe_path(dir_path)
        
        # 搜索
        results = []
        for file_path in path.rglob(pattern):
            if file_type and file_path.suffix != file_type:
                continue
            
            results.append(file_path)
        
        # 格式化结果
        output = f"搜索结果: {pattern}\n\n"
        output += f"找到 {len(results)} 个文件\n\n"
        
        for file_path in results[:50]:  # 限制显示数量
            rel_path = file_path.relative_to(path)
            if file_path.is_dir():
                output += f"📁 {rel_path}/\n"
            else:
                size = file_path.stat().st_size
                output += f"📄 {rel_path} ({size:,} bytes)\n"
        
        if len(results) > 50:
            output += f"\n... 还有 {len(results) - 50} 个文件"
        
        return output
    
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"搜索失败: {e}"

@mcp.resource("file://{path}")
def read_file_resource(path: str) -> str:
    """文件资源"""
    try:
        file_path = safe_path(path)
        return file_path.read_text(encoding='utf-8')
    except Exception as e:
        return f"读取失败: {e}"

# 启动
if __name__ == "__main__":
    mcp.run()
```

### 1.3 API 集成 MCP Server

API 集成 MCP Server，封装 REST API、GraphQL 支持、认证管理。

```python
"""
API 集成 MCP Server
"""
from mcp.server.fastmcp import FastMCP
import httpx
from typing import Optional
import json
from datetime import datetime

mcp = FastMCP("api-integration")

# HTTP 客户端
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    timeout=30.0,
    headers={
        "Content-Type": "application/json",
        "User-Agent": "MCP-API-Client/1.0"
    }
)

# 认证
API_KEY = "your-api-key"

@mcp.tool()
async def get_users(
    page: int = 1,
    limit: int = 10,
    status: Optional[str] = None
) -> str:
    """
    获取用户列表
    
    Args:
        page: 页码
        limit: 每页数量
        status: 状态过滤
    """
    try:
        params = {
            "page": page,
            "limit": limit
        }
        if status:
            params["status"] = status
        
        response = await client.get(
            "/users",
            params=params,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        response.raise_for_status()
        
        data = response.json()
        
        output = f"用户列表 (第 {page} 页)\n\n"
        output += f"总计: {data['total']}\n\n"
        
        for user in data['users']:
            output += f"- {user['name']} ({user['email']})\n"
            output += f"  状态: {user['status']}\n"
            output += f"  创建时间: {user['created_at']}\n\n"
        
        return output
    
    except httpx.HTTPError as e:
        return f"API 请求失败: {e}"
    except Exception as e:
        return f"获取用户失败: {e}"

@mcp.tool()
async def create_user(
    name: str,
    email: str,
    role: str = "user"
) -> str:
    """
    创建用户
    
    Args:
        name: 用户名
        email: 邮箱
        role: 角色
    """
    try:
        user_data = {
            "name": name,
            "email": email,
            "role": role
        }
        
        response = await client.post(
            "/users",
            json=user_data,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        response.raise_for_status()
        
        user = response.json()
        
        return f"用户创建成功:\n\nID: {user['id']}\n名称: {user['name']}\n邮箱: {user['email']}"
    
    except httpx.HTTPError as e:
        return f"API 请求失败: {e}"
    except Exception as e:
        return f"创建用户失败: {e}"

@mcp.tool()
async def graphql_query(query: str, variables: Optional[dict] = None) -> str:
    """
    执行 GraphQL 查询
    
    Args:
        query: GraphQL 查询语句
        variables: 查询变量
    """
    try:
        graphql_data = {
            "query": query
        }
        if variables:
            graphql_data["variables"] = variables
        
        response = await client.post(
            "/graphql",
            json=graphql_data,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        response.raise_for_status()
        
        result = response.json()
        
        if "errors" in result:
            return f"GraphQL 错误:\n\n{json.dumps(result['errors'], indent=2)}"
        
        return json.dumps(result['data'], indent=2)
    
    except httpx.HTTPError as e:
        return f"API 请求失败: {e}"
    except Exception as e:
        return f"GraphQL 查询失败: {e}"

@mcp.resource("api://users/{user_id}")
def get_user_resource(user_id: str) -> str:
    """用户信息资源"""
    import asyncio
    
    async def fetch():
        response = await client.get(
            f"/users/{user_id}",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        response.raise_for_status()
        return response.json()
    
    try:
        user = asyncio.get_event_loop().run_until_complete(fetch())
        return json.dumps(user, indent=2)
    except Exception as e:
        return f"获取用户失败: {e}"

# 清理
@mcp.on_shutdown()
async def cleanup():
    """关闭 HTTP 客户端"""
    await client.aclose()

# 启动
if __name__ == "__main__":
    mcp.run()
```

---

## 2. 生态工具

### 2.1 官方 Servers

Model Context Protocol 官方维护的 Servers：

#### GitHub Server

```bash
# 安装
npx -y @modelcontextprotocol/server-github

# 配置
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token"
      }
    }
  }
}

# 功能
- 仓库查询
- Issue 管理
- Pull Request 操作
- 文件管理
```

#### Filesystem Server

```bash
# 安装
npx -y @modelcontextprotocol/server-filesystem /path/to/allow

# 配置
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
    }
  }
}

# 功能
- 文件读写
- 目录操作
- 文件搜索
- 权限控制
```

#### PostgreSQL Server

```bash
# 安装
npx -y @modelcontextprotocol/server-postgres postgresql://localhost/mydb

# 配置
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://user:password@localhost:5432/mydb"
      ]
    }
  }
}

# 功能
- SQL 查询（只读）
- 表结构查看
- 数据导出
```

#### Playwright Server

```bash
# 安装
npx -y @modelcontextprotocol/server-playwright

# 配置
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-playwright"]
    }
  }
}

# 功能
- 网页截图
- 表单填充
- 元素交互
- 自动化测试
```

### 2.2 社区 Servers

#### 搜索工具

```bash
# Brave Search
npx -y @modelcontextprotocol/server-brave-search

# Google Search
npm install @anthropic/mcp-server-google-search

# DuckDuckGo
npm install @anthropic/mcp-server-duckduckgo
```

#### 开发工具

```bash
# Git
npx -y @modelcontextprotocol/server-git

# SQLite
npx -y @modelcontextprotocol/server-sqlite

# Slack
npx -y @modelcontextprotocol/server-slack

# Google Maps
npx -y @modelcontextprotocol/server-google-maps
```

#### 生产力工具

```bash
# Notion
npm install @notionhq/mcp-server

# Linear
npm install @linear/mcp-server

# Figma
npm install @figma/mcp-server
```

### 2.3 Skills 市场

#### Skills 发现

```bash
# MCP Skills Registry
# https://skills.modelcontextprotocol.io

# 搜索 Skills
mcp-skills search "code-review"
mcp-skills search "testing"
mcp-skills search "documentation"
```

#### Skills 安装

```bash
# 安装 Skill
mcp-skills install code-review-skill

# 查看已安装
mcp-skills list

# 更新 Skill
mcp-skills update code-review-skill
```

---

## 3. 前沿技术

### 3.1 MCP 协议演进

#### 版本历史

```
v1.0 (2024-11-05)
- 初始版本
- 基础 JSON-RPC 协议
- Tools、Resources、Prompts 支持
- Stdio 和 HTTP/SSE 传输

v1.1 (2025-01-15)
- 改进错误处理
- 添加批处理支持
- 增强安全特性

v2.0 (2025-03-26)
- 重大更新
- 改进的能力协商
- 新增采样（Sampling）功能
- 改进的认证框架
- 更好的类型安全
```

#### 新特性路线图

```
2025 Q2:
- 多模态支持（图像、音频、视频）
- 改进的流式响应
- 更好的错误恢复

2025 Q3:
- 联邦 MCP（跨服务器协作）
- 智能路由
- 自动发现协议

2025 Q4:
- MCP 2.1
- 区块链集成
- 去中心化身份
```

### 3.2 多模态 MCP

#### 图像工具

```python
@mcp.tool()
async def generate_image(prompt: str, size: str = "1024x1024") -> dict:
    """
    生成图像
    
    Args:
        prompt: 图像描述
        size: 图像尺寸
    """
    # 调用图像生成 API
    image_data = await call_image_api(prompt, size)
    
    return {
        "content": [
            {
                "type": "image",
                "data": image_data["base64"],
                "mimeType": "image/png"
            }
        ]
    }

@mcp.tool()
async def analyze_image(image_url: str) -> str:
    """
    分析图像内容
    
    Args:
        image_url: 图像 URL
    """
    # 下载图像
    image_data = await download_image(image_url)
    
    # 使用视觉模型分析
    analysis = await analyze_with_vision_model(image_data)
    
    return analysis
```

#### 音频工具

```python
@mcp.tool()
async def text_to_speech(text: str, voice: str = "default") -> dict:
    """
    文本转语音
    
    Args:
        text: 文本内容
        voice: 语音类型
    """
    # 调用 TTS API
    audio_data = await call_tts_api(text, voice)
    
    return {
        "content": [
            {
                "type": "audio",
                "data": audio_data["base64"],
                "mimeType": "audio/mpeg"
            }
        ]
    }

@mcp.tool()
async def transcribe_audio(audio_url: str) -> str:
    """
    转录音频
    
    Args:
        audio_url: 音频 URL
    """
    # 下载音频
    audio_data = await download_audio(audio_url)
    
    # 转录
    transcript = await transcribe(audio_data)
    
    return transcript
```

### 3.3 未来方向

#### 自动化 Server 生成

```python
# 从 OpenAPI 规范自动生成 MCP Server
from openapi_to_mcp import generate_server

# 读取 OpenAPI 规范
with open("openapi.json") as f:
    spec = json.load(f)

# 生成 MCP Server
server_code = generate_server(spec)

# 输出
with open("generated_server.py", "w") as f:
    f.write(server_code)
```

#### 智能路由

```python
# 基于语义的智能路由
class SmartRouter:
    """智能路由 MCP Server"""
    
    def __init__(self):
        self.servers = {}
        self.semantic_index = {}
    
    async def route_request(self, request: str) -> str:
        """智能路由请求"""
        # 分析请求意图
        intent = await analyze_intent(request)
        
        # 匹配最合适的 Server
        best_server = self.semantic_index.match(intent)
        
        # 转发请求
        return await best_server.handle(request)
```

#### 联邦学习

```
联邦 MCP 架构：

┌─────────────────────────────────────────┐
│          Federation Layer                 │
│                                          │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Server A │  │ Server B │  │Server C│ │
│  │ (Local)  │  │ (Local)  │  │(Cloud) │ │
│  └────┬─────┘  └────┬─────┘  └───┬────┘ │
│       │              │            │       │
│       └──────────────┼────────────┘       │
│                      │                    │
│              Knowledge Sharing            │
└──────────────────────────────────────────┘
```

#### 去中心化架构

```
去中心化 MCP：

基于区块链的 Server 注册和发现
智能合约管理权限和计费
IPFS 存储资源和配置
DID（去中心化身份）认证

优势：
- 抗审查
- 高可用
- 用户主权
- 透明计费
```

---


### 官方文档

- **MCP 官方文档**: https://modelcontextprotocol.io
- **MCP 规范**: https://github.com/modelcontextprotocol/specification
- **Python SDK**: https://github.com/modelcontextprotocol/python-sdk
- **TypeScript SDK**: https://github.com/modelcontextprotocol/typescript-sdk

### 学习资源

- **Microsoft MCP for Beginners**: https://github.com/microsoft/mcp-for-beginners
- **MCP 中文教程**: https://github.com/chenmingyong0423/mcp-tutorials
- **MCP 资源集合**: https://github.com/cyanheads/model-context-protocol-resources

### 官方 Servers

- **GitHub Server**: https://github.com/modelcontextprotocol/servers/tree/main/src/github
- **Filesystem Server**: https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem
- **PostgreSQL Server**: https://github.com/modelcontextprotocol/servers/tree/main/src/postgres
- **Playwright Server**: https://github.com/modelcontextprotocol/servers/tree/main/src/playwright

### 社区资源

- **MCP Discord**: https://discord.gg/mcp
- **MCP GitHub Discussions**: https://github.com/modelcontextprotocol/specification/discussions
- **Awesome MCP**: https://github.com/punkpeye/awesome-mcp-servers

### 相关技术

- **JSON-RPC 2.0**: https://www.jsonrpc.org/specification
- **Server-Sent Events**: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- **OpenAPI Specification**: https://swagger.io/specification/

---

## 4. MCP 2026-07-28 安全与生产部署更新

### 4.1 授权强化（Authorization Hardening）

MCP 2026-07-28 规范通过六个 SEP 强化授权,使其更符合 OAuth 2.0 和 OpenID Connect 的实际部署。

#### RFC 9207 iss 参数验证

**问题**: 在 MCP 的单客户端多服务器部署模式中,mix-up 攻击更常见。

**解决方案**: 客户端必须验证授权响应中的 `iss` 参数。

```python
from oauthlib.oauth2 import WebApplicationClient
import httpx

async def handle_authorization_response(authorization_response: dict):
    """
    处理授权响应,验证 iss 参数
    """
    # 提取 iss 参数
    iss = authorization_response.get("iss")
    
    if not iss:
        raise ValueError("授权响应缺少 iss 参数")
    
    # 验证 iss 与预期的授权服务器匹配
    expected_iss = "https://auth.example.com"
    if iss != expected_iss:
        raise ValueError(
            f"iss 不匹配: 期望 {expected_iss}, 实际 {iss}"
        )
    
    # 继续处理授权码
    code = authorization_response.get("code")
    if not code:
        raise ValueError("授权响应缺少 code 参数")
    
    # 交换授权码获取令牌
    token_response = await exchange_code_for_token(code, iss)
    return token_response

async def exchange_code_for_token(code: str, issuer: str) -> dict:
    """使用授权码交换令牌"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{issuer}/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": "my-client-id",
                "client_secret": "my-client-secret",
                "redirect_uri": "http://localhost:3000/callback"
            }
        )
        return response.json()
```

**迁移路径**:
- 当前版本: 客户端应验证 `iss`,但不强制要求
- 未来版本: 客户端必须拒绝缺少 `iss` 的响应
- **建议**: 授权服务器现在就应该开始提供 `iss` 参数

#### OpenID Connect application_type 声明

**问题**: 授权服务器经常默认桌面或 CLI 客户端为 "web" 类型,拒绝 localhost 重定向 URI。

**解决方案**: 客户端在动态客户端注册时声明应用类型。

```python
import httpx

async def register_client(authorization_server: str):
    """
    动态客户端注册,声明 application_type
    """
    registration_endpoint = f"{authorization_server}/register"
    
    registration_data = {
        "client_name": "My MCP Client",
        "redirect_uris": [
            "http://localhost:3000/callback",
            "http://localhost:8080/callback"
        ],
        "application_type": "native",  # 声明为原生应用（桌面/CLI）
        "grant_types": [
            "authorization_code",
            "refresh_token"
        ],
        "response_types": ["code"],
        "token_endpoint_auth_method": "client_secret_post",
        "scope": "openid profile email mcp:tools mcp:resources"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            registration_endpoint,
            json=registration_data
        )
        
        if response.status_code == 201:
            client_credentials = response.json()
            print(f"Client ID: {client_credentials['client_id']}")
            print(f"Client Secret: {client_credentials['client_secret']}")
            return client_credentials
        else:
            raise Exception(f"注册失败: {response.text}")
```

**application_type 选项**:
- `"web"`: Web 应用（服务器端）
- `"native"`: 原生应用（桌面/CLI/移动）
- 避免授权服务器错误拒绝 localhost 重定向 URI

#### 凭证绑定与迁移

**问题**: 资源在授权服务器间迁移时,已注册的凭证失效。

**解决方案**: 客户端将凭证绑定到授权服务器的 issuer,迁移时重新注册。

```python
class MCPClientAuth:
    def __init__(self):
        self.credentials_store = {}
    
    def register_credentials(
        self,
        issuer: str,
        client_id: str,
        client_secret: str
    ):
        """注册凭证,绑定到 issuer"""
        self.credentials_store[issuer] = {
            "client_id": client_id,
            "client_secret": client_secret,
            "registered_at": datetime.now()
        }
    
    def get_credentials(self, issuer: str) -> dict:
        """获取凭证"""
        if issuer not in self.credentials_store:
            raise ValueError(f"未找到 issuer {issuer} 的凭证")
        return self.credentials_store[issuer]
    
    async def migrate_resource(
        self,
        resource_url: str,
        old_issuer: str,
        new_issuer: str
    ):
        """
        资源迁移到新的授权服务器
        
        需要重新注册凭证
        """
        # 1. 从旧授权服务器注销
        await self.deregister_from_issuer(old_issuer)
        
        # 2. 在新授权服务器重新注册
        new_credentials = await self.register_with_issuer(new_issuer)
        
        # 3. 更新资源配置
        await self.update_resource_config(
            resource_url,
            new_issuer,
            new_credentials
        )
        
        # 4. 存储新凭证
        self.register_credentials(
            new_issuer,
            new_credentials["client_id"],
            new_credentials["client_secret"]
        )
        
        print(f"资源 {resource_url} 已迁移到 {new_issuer}")
    
    async def deregister_from_issuer(self, issuer: str):
        """从授权服务器注销"""
        # 实现注销逻辑
        pass
    
    async def register_with_issuer(self, issuer: str) -> dict:
        """在授权服务器注册"""
        return await register_client(issuer)
    
    async def update_resource_config(
        self,
        resource_url: str,
        issuer: str,
        credentials: dict
    ):
        """更新资源配置"""
        # 实现更新逻辑
        pass
```

#### 刷新令牌支持

规范记录了如何从 OpenID Connect 风格的授权服务器请求刷新令牌。

```python
async def request_tokens_with_refresh(authorization_server: str, code: str):
    """
    请求访问令牌和刷新令牌
    """
    token_endpoint = f"{authorization_server}/token"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            token_endpoint,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": "my-client-id",
                "client_secret": "my-client-secret",
                "redirect_uri": "http://localhost:3000/callback",
                "scope": "openid profile email mcp:tools offline_access"
                # offline_access scope 请求刷新令牌
            }
        )
        
        token_response = response.json()
        
        return {
            "access_token": token_response["access_token"],
            "refresh_token": token_response.get("refresh_token"),
            "expires_in": token_response["expires_in"],
            "token_type": token_response["token_type"]
        }

async def refresh_access_token(authorization_server: str, refresh_token: str):
    """
    使用刷新令牌获取新的访问令牌
    """
    token_endpoint = f"{authorization_server}/token"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            token_endpoint,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": "my-client-id",
                "client_secret": "my-client-secret",
                "scope": "openid profile email mcp:tools"
            }
        )
        
        return response.json()
```

#### 范围累积（Scope Accumulation）

明确了 step-up 认证时的范围累积规则。

```python
class ScopeManager:
    def __init__(self):
        self.granted_scopes = set()
    
    def request_step_up(
        self,
        current_scopes: list[str],
        required_scopes: list[str]
    ) -> list[str]:
        """
        请求额外的 scope（step-up）
        
        范围累积:新请求的 scope 与已有的 scope 合并
        """
        # 计算需要的新 scope
        new_scopes = set(required_scopes) - set(current_scopes)
        
        if not new_scopes:
            return current_scopes
        
        # 累积范围
        accumulated_scopes = set(current_scopes) | set(required_scopes)
        
        return list(accumulated_scopes)
    
    async def perform_step_up_auth(
        self,
        authorization_server: str,
        current_token: str,
        required_scopes: list[str]
    ) -> dict:
        """
        执行 step-up 认证
        
        用户需要重新授权,获得额外的 scope
        """
        # 累积范围
        accumulated_scopes = self.request_step_up(
            self.granted_scopes,
            required_scopes
        )
        
        # 重定向到授权服务器,请求累积的 scope
        auth_url = self.build_authorization_url(
            authorization_server,
            accumulated_scopes,
            prompt="consent"  # 强制用户重新同意
        )
        
        # 用户授权后,获取新令牌
        # ... (授权流程)
        
        new_token = await self.exchange_code_for_token(code)
        self.granted_scopes = set(accumulated_scopes)
        
        return new_token
```

### 4.2 可观测性（Observability）

MCP 2026-07-28 规范正式弃用 Logging 特性,推荐使用 OpenTelemetry 进行结构化可观测性。

#### W3C Trace Context 传播

`_meta` 中的 W3C Trace Context 传播现已文档化。

```python
from mcp.server.fastmcp import FastMCP
from mcp.server.context import RequestContext
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

mcp = FastMCP("my-server", protocol_version="2026-07-28")
tracer = trace.get_tracer("mcp-server")

@mcp.tool()
async def process_request(data: str, context: RequestContext) -> dict:
    """
    处理请求,传播追踪上下文
    """
    # 从 _meta 提取 W3C Trace Context
    carrier = {
        "traceparent": context.meta.get("traceparent", ""),
        "tracestate": context.meta.get("tracestate", ""),
        "baggage": context.meta.get("baggage", {})
    }
    
    # 提取上下文
    ctx = TraceContextTextMapPropagator().extract(carrier=carrier)
    
    # 在当前 span 中继续追踪
    with tracer.start_as_current_span(
        "mcp.tool.process_request",
        context=ctx,
        attributes={
            "mcp.tool.name": "process_request",
            "mcp.tool.input_size": len(data)
        }
    ) as span:
        # 处理请求
        result = await do_processing(data)
        
        # 记录结果到 span
        span.set_attribute("mcp.tool.output_size", len(result))
        
        return {
            "content": [{"type": "text", "text": result}],
            "ttlMs": 60000,
            "cacheScope": "user"
        }
```

**追踪上下文格式**:

```python
# traceparent 格式
# {version}-{trace-id}-{parent-id}-{flags}
# 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01

traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
# version: 00
# trace-id: 4bf92f3577b34da6a3ce929d0e0e4736
# parent-id: 00f067aa0ba902b7
# flags: 01 (sampled)
```

#### OpenTelemetry 集成

使用 OpenTelemetry 进行结构化可观测性。

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

# 配置资源
resource = Resource.create({
    "service.name": "mcp-server",
    "service.version": "1.0.0",
    "deployment.environment": "production"
})

# 配置追踪
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint="http://otel-collector:4318/v1/traces"
        )
    )
)
trace.set_tracer_provider(tracer_provider)

# 配置指标
meter_provider = MeterProvider(resource=resource)
meter_provider.add_metric_reader(
    # 配置指标导出
)
metrics.set_meter_provider(meter_provider)

# 使用
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server", protocol_version="2026-07-28")
tracer = trace.get_tracer("mcp-server")
meter = metrics.get_meter("mcp-server")

# 创建指标
tool_calls_counter = meter.create_counter(
    "mcp.tool.calls",
    unit="1",
    description="Number of tool calls"
)

tool_duration_histogram = meter.create_histogram(
    "mcp.tool.duration",
    unit="ms",
    description="Tool call duration"
)

@mcp.tool()
async def search(query: str, context: RequestContext) -> dict:
    """搜索工具,记录指标"""
    start_time = time.time()
    
    with tracer.start_as_current_span(
        "mcp.tool.search",
        attributes={"mcp.tool.query": query}
    ) as span:
        try:
            # 执行搜索
            results = await do_search(query)
            
            # 记录成功指标
            duration_ms = (time.time() - start_time) * 1000
            tool_calls_counter.add(1, {"tool": "search", "status": "success"})
            tool_duration_histogram.record(duration_ms, {"tool": "search"})
            
            span.set_attribute("mcp.tool.result_count", len(results))
            
            return {
                "content": [{"type": "text", "text": results}],
                "ttlMs": 300000,
                "cacheScope": "user"
            }
        
        except Exception as e:
            # 记录失败指标
            tool_calls_counter.add(1, {"tool": "search", "status": "error"})
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise
```

#### 结构化日志

对于 stdio 传输,使用 stderr 输出结构化日志。

```python
import sys
import json
import logging

# 配置结构化日志
logger = logging.getLogger("mcp-server")
logger.setLevel(logging.INFO)

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(StructuredFormatter())
logger.addHandler(handler)

# 使用
@mcp.tool()
async def process_data(data: str) -> dict:
    """处理数据,记录结构化日志"""
    logger.info("Processing data", extra={"data_size": len(data)})
    
    try:
        result = await do_process(data)
        logger.info("Data processed successfully", extra={"result_size": len(result)})
        return {"content": [{"type": "text", "text": result}]}
    
    except Exception as e:
        logger.error("Failed to process data", exc_info=True)
        raise
```

### 4.3 生产部署最佳实践（2026-07-28）

#### 无状态水平扩展

MCP 2026-07-28 的无状态特性使水平扩展更简单。

```yaml
# docker-compose.yml - 无状态部署
version: '3.8'

services:
  mcp-server:
    image: my-mcp-server:latest
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/mydb
      - REDIS_URL=redis://redis:6379
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "8000:8000"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - mcp-server

  db:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  otel-collector:
    image: otel/opentelemetry-collector:latest
    volumes:
      - ./otel-config.yaml:/etc/otel/config.yaml
    command: ["--config", "/etc/otel/config.yaml"]

volumes:
  postgres_data:
  redis_data:
```

```nginx
# nginx.conf - 普通轮询负载均衡（无需 sticky sessions）
http {
    upstream mcp_backend {
        # 无状态服务器,使用普通轮询
        server mcp-server-1:8000;
        server mcp-server-2:8000;
        server mcp-server-3:8000;
        
        # 可选: 加权轮询
        # server mcp-server-1:8000 weight=3;
        # server mcp-server-2:8000 weight=2;
        # server mcp-server-3:8000 weight=1;
    }
    
    server {
        listen 8000;
        
        location /mcp {
            proxy_pass http://mcp_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            # MCP 协议头部
            proxy_set_header MCP-Protocol-Version 2026-07-28;
            
            # 超时配置
            proxy_connect_timeout 10s;
            proxy_read_timeout 300s;
            proxy_send_timeout 300s;
        }
        
        location /health {
            return 200 "OK";
            add_header Content-Type text/plain;
        }
    }
}
```

#### 基于 HTTP 头部的路由和限流

利用 `Mcp-Method` 和 `Mcp-Name` 头部进行精细控制。

```nginx
# nginx.conf - 基于头部的路由和限流
http {
    # 限流区域
    limit_req_zone $binary_remote_addr zone=mcp_general:10m rate=100r/s;
    limit_req_zone $binary_remote_addr zone=mcp_expensive:10m rate=10r/s;
    
    # 基于 Mcp-Method 的路由
    map $http_mcp_method $backend_pool {
        default mcp_backend;
        "tools/call" mcp_tools;
        "resources/read" mcp_resources;
    }
    
    upstream mcp_tools {
        server mcp-server-1:8000;
        server mcp-server-2:8000;
    }
    
    upstream mcp_resources {
        server mcp-server-3:8000;
    }
    
    server {
        listen 8000;
        
        location /mcp {
            # 基于 Mcp-Name 的限流
            set $limit_zone mcp_general;
            if ($http_mcp_name ~* "(train_model|process_large_file)") {
                set $limit_zone mcp_expensive;
            }
            
            limit_req zone=$limit_zone burst=20 nodelay;
            
            # 基于 Mcp-Method 的路由
            proxy_pass http://$backend_pool;
            
            # 验证头部
            if ($http_mcp_method = "") {
                return 400 "Missing Mcp-Method header";
            }
        }
    }
}
```

#### Kubernetes 部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
  labels:
    app: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      containers:
      - name: mcp-server
        image: my-mcp-server:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: database-url
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://otel-collector:4318"
        resources:
          requests:
            cpu: 500m
            memory: 256Mi
          limits:
            cpu: "1"
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server
spec:
  selector:
    app: mcp-server
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mcp-server
  annotations:
    nginx.ingress.kubernetes.io/load-balance: "round_robin"
    # 无需 sticky sessions
spec:
  rules:
  - host: mcp.example.com
    http:
      paths:
      - path: /mcp
        pathType: Prefix
        backend:
          service:
            name: mcp-server
            port:
              number: 8000
```

#### 缓存策略实施

利用 `ttlMs` 和 `cacheScope` 实施缓存。

```python
from mcp.server.fastmcp import FastMCP
import redis.asyncio as redis

mcp = FastMCP("my-server", protocol_version="2026-07-28")
redis_client = redis.Redis(host="redis", port=6379, db=0)

@mcp.tool()
async def get_product(product_id: str) -> dict:
    """
    获取产品信息
    
    实施缓存策略
    """
    # 检查缓存
    cache_key = f"product:{product_id}"
    cached = await redis_client.get(cache_key)
    
    if cached:
        return {
            "content": [{"type": "text", "text": cached}],
            "ttlMs": 300000,  # 5 分钟
            "cacheScope": "user"
        }
    
    # 从数据库获取
    product = await fetch_product(product_id)
    result_text = format_product(product)
    
    # 写入缓存
    await redis_client.setex(
        cache_key,
        300,  # 5 分钟 TTL
        result_text
    )
    
    return {
        "content": [{"type": "text", "text": result_text}],
        "ttlMs": 300000,
        "cacheScope": "user"
    }

@mcp.tool()
async def search_products(query: str) -> dict:
    """
    搜索产品
    
    不缓存搜索结果
    """
    results = await search_database(query)
    
    return {
        "content": [{"type": "text", "text": results}],
        "ttlMs": 0,  # 不缓存
        "cacheScope": "request"
    }
```

#### 安全加固

```python
from mcp.server.fastmcp import FastMCP
import jwt
from datetime import datetime, timedelta

mcp = FastMCP("my-server", protocol_version="2026-07-28")

# JWT 验证
async def verify_token(token: str) -> dict:
    """验证 JWT 令牌"""
    try:
        payload = jwt.decode(
            token,
            "your-secret-key",
            algorithms=["HS256"],
            audience="mcp-server",
            issuer="https://auth.example.com"
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")

# 中间件: 验证每个请求
@mcp.middleware("request")
async def auth_middleware(request, handler):
    """认证中间件"""
    # 从请求中提取令牌
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise ValueError("Missing or invalid Authorization header")
    
    token = auth_header.split(" ")[1]
    
    # 验证令牌
    payload = await verify_token(token)
    
    # 将用户信息添加到请求上下文
    request.context.user = payload
    
    # 继续处理
    return await handler(request)

# 基于角色的访问控制
def require_role(role: str):
    """角色装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            context = kwargs.get("context")
            if not context or not hasattr(context, "user"):
                raise ValueError("Authentication required")
            
            user_roles = context.user.get("roles", [])
            if role not in user_roles:
                raise ValueError(f"Role '{role}' required")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@mcp.tool()
@require_role("admin")
async def delete_all_data(context: RequestContext) -> dict:
    """仅管理员可调用"""
    # 执行删除操作
    pass
```

### 4.4 弃用特性迁移指南

根据 MCP 2026-07-28 的特性生命周期政策,以下特性已弃用:

#### Roots 弃用

**旧代码**:
```python
# 旧版本 - 使用 Roots
@mcp.root()
def project_root():
    return Root(
        uri="file:///home/user/project",
        name="Project Root"
    )
```

**新代码**:
```python
# 新版本 - 使用工具参数或资源 URI
@mcp.tool()
def analyze_project(project_path: str) -> str:
    """分析项目,路径通过参数传递"""
    return analyze(Path(project_path))

# 或使用资源
@mcp.resource("project://config")
def project_config():
    """项目配置作为资源"""
    return get_project_config()
```

#### Sampling 弃用

**旧代码**:
```python
# 旧版本 - 使用 MCP Sampling
@mcp.tool()
async def generate_text(prompt: str, context: RequestContext) -> str:
    # 请求 LLM 采样
    result = await context.sample_llm(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return result
```

**新代码**:
```python
# 新版本 - 直接集成 LLM 提供商 API
import openai

@mcp.tool()
async def generate_text(prompt: str) -> str:
    """生成文本,使用 OpenAI API"""
    client = openai.AsyncOpenAI(api_key="your-api-key")
    
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    
    return response.choices[0].message.content
```

#### Logging 弃用

**旧代码**:
```python
# 旧版本 - 使用 MCP Logging
@mcp.tool()
async def process_data(data: str, context: RequestContext) -> str:
    await context.log(
        level="info",
        message="Processing data",
        data={"size": len(data)}
    )
    return process(data)
```

**新代码**:
```python
# 新版本 - 使用 OpenTelemetry
from opentelemetry import trace

tracer = trace.get_tracer("mcp-server")

@mcp.tool()
async def process_data(data: str) -> str:
    """处理数据,使用 OpenTelemetry"""
    with tracer.start_as_current_span("process_data") as span:
        span.set_attribute("data.size", len(data))
        span.add_event("Processing started")
        
        result = process(data)
        
        span.add_event("Processing completed")
        return result
```

**弃用时间线**:
- 2026-07-28: 标记为弃用（annotation-only）
- 2027-07-28: 最早可能的移除时间（12个月后）
- 移除需要单独的 SEP

---

---


本笔记全面覆盖了 **MCP (Model Context Protocol)** 的核心概念和实战开发，包括最新的 2026-07-28 规范更新。

### 核心要点

1. **MCP 是什么**
   - 标准化的 LLM 工具集成协议
   - 基于 JSON-RPC 2.0 的消息规范
   - Client-Server 架构
   - **2026-07-28**: 无状态协议核心，移除会话管理

2. **三大核心功能**
   - **Resources**: 提供上下文数据
   - **Tools**: 执行操作和访问服务
   - **Prompts**: 模板化提示
   - **2026-07-28 新增**: MCP Apps（服务器渲染的 HTML 界面）

3. **多语言支持**
   - Python: `pip install mcp>=1.10.0`
   - TypeScript: `npm install @modelcontextprotocol/sdk`
   - Go: `go get github.com/metoro-io/mcp-golang`

4. **2026-07-28 重大更新**
   - **无状态协议**: 移除 initialize/initialized 握手
   - **扩展机制**: Tasks、MCP Apps 作为官方扩展
   - **JSON Schema 2020-12**: 完整支持 oneOf、anyOf、allOf、$ref
   - **缓存支持**: ttlMs 和 cacheScope
   - **授权强化**: OAuth 2.0 + OpenID Connect 完整支持
   - **可观测性**: W3C Trace Context + OpenTelemetry
   - **弃用通知**: Roots、Sampling、Logging 已弃用

5. **开发最佳实践**
   - 单一职责
   - 完善错误处理
   - 安全认证和授权（RFC 9207 iss 验证）
   - 性能优化和监控（OpenTelemetry）
   - 显式状态管理（替代会话状态）

6. **生产部署**
   - 无状态水平扩展（无需 sticky sessions）
   - 基于 HTTP 头部的路由和限流（Mcp-Method、Mcp-Name）
   - 容器化和云原生（Docker、Kubernetes）
   - 监控告警和日志聚合（OpenTelemetry）
   - 缓存策略实施（Redis + ttlMs）

### 下一步学习

- 实践官方 Servers（支持 2026-07-28 规范）
- 开发自己的 MCP Server（无状态架构）
- 参与社区贡献
- 关注协议演进（Extensions Track）
- 学习 MCP Apps 开发
- 实施 Tasks 扩展处理长时间任务

---

**笔记版本**: v1.0  
**创建时间**: 2025-06-12  
**最后更新**: 2025-06-12  
**字数**: ~15,000 字  
**代码行数**: 3,000+ 行