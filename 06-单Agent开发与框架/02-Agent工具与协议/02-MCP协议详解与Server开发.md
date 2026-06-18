# MCP 协议详解与 Server 开发

> 📅 **更新时间**: 2026-06-18  

---

## 目录

**第一部分：MCP 协议基础**
- [1. MCP 协议基础概念](#1-mcp-协议基础概念)
- [2. MCP 2026-07-28 规范重大更新](#2-mcp-2026-07-28-规范重大更新)
- [3. MCP 协议详解](#3-mcp-协议详解)

**第二部分：MCP Server 开发实战**
- [4. MCP Server 开发（Python）](#4-mcp-server-开发python)
- [5. MCP Server 开发（TypeScript）](#5-mcp-server-开发typescript)
- [6. MCP Server 开发（Go）](#6-mcp-server-开发go)
- [7. MCP Client 集成](#7-mcp-client-集成)
- [8. MCP 2026-07-28 新特性速查](#8-mcp-2026-07-28-新特性速查)
- [9. 更多实战场景](#9-更多实战场景)

**第三部分：MCP 安全权限与生产部署**
- [10. 生态工具](#10-生态工具)
- [11. 前沿技术](#11-前沿技术)
- [12. 参考资料](#12-参考资料)
- [13. MCP 2026-07-28 安全与生产部署更新](#13-mcp-2026-07-28-安全与生产部署更新)

---

## 1. MCP 协议基础概念

### 1.1 什么是 MCP

#### Model Context Protocol 定义

**MCP（Model Context Protocol）** 是一个开放的协议标准,由 Anthropic 于 2024 年底发起,旨在标准化 LLM（Large Language Model）应用程序与外部数据源和工具之间的集成方式。

> **🎯 最新版本**: MCP 规范 `2026-07-28`（2026年7月发布）
> - 无状态协议核心（Stateless Protocol）
> - 扩展机制（Extensions）：MCP Apps、Tasks
> - 强化授权模型（Authorization Hardening）
> - 完整 JSON Schema 2020-12 支持
> - 正式弃用 Roots、Sampling、Logging

从技术角度看,MCP 定义了一套基于 JSON-RPC 2.0 的消息规范,使得:

- **Hosts（主机）**：LLM 应用程序（如 Claude Desktop、Cursor、VS Code 等）可以发起连接
- **Clients（客户端）**：主机应用内的连接器，负责与服务器通信
- **Servers（服务器）**：提供上下文和能力的服务

通过这种标准化的方式，LLM 应用可以像使用本地功能一样，无缝访问外部资源、调用工具、使用提示模板。

**核心定义**：

```
MCP = 标准化的 LLM 工具集成协议
    = JSON-RPC 2.0 消息规范
    = Client-Server 架构
    = 资源 + 工具 + 提示的统一接口
```

#### 为什么需要 MCP

在 MCP 出现之前，LLM 应用集成外部工具面临以下问题：

1. **碎片化严重**
   - 每个 LLM 应用（ChatGPT、Claude、Gemini）都有自己的工具集成方式
   - 开发者需要为每个平台重复开发适配代码
   - 缺乏统一标准，维护成本极高

2. **集成复杂度高**
   ```
   传统方式（无 MCP）：
   LLM App A → 适配层 A → Tool X
   LLM App B → 适配层 B → Tool X
   LLM App C → 适配层 C → Tool X
   
   MCP 方式：
   LLM App A ↘
   LLM App B → MCP Client → MCP Server → Tool X
   LLM App C ↗
   ```

3. **上下文丢失**
   - 工具调用时缺少完整的上下文信息
   - 难以在多个工具间共享状态
   - 无法构建复杂的多步骤工作流

4. **安全风险**
   - 缺乏统一的权限控制机制
   - 用户难以理解工具的实际行为
   - 数据隐私难以保障

**MCP 的解决方案**：

```python
# MCP 统一接口示例
# 无论底层是什么工具，调用方式完全一致

# 调用工具
result = await client.call_tool("search", {"query": "MCP protocol"})

# 读取资源
config = await client.read_resource("config://app")

# 使用提示
prompt = await client.get_prompt("code-review", {"code": "..."})
```

#### MCP vs 其他协议

| 特性 | MCP | OpenAPI (Swagger) | LangChain Tools | OpenAI Plugins |
|------|-----|-------------------|-----------------|----------------|
| **设计目标** | LLM 工具集成 | REST API 描述 | 框架特定工具 | ChatGPT 插件 |
| **传输协议** | JSON-RPC 2.0 | HTTP/REST | 框架内部 | HTTP/REST |
| **标准化程度** | 开放标准（Linux Foundation） | 开放标准 | 框架绑定 | 平台特定 |
| **多语言支持** | Python/TS/Go/Java/Kotlin 等 | 所有语言 | Python/JS | 所有语言 |
| **实时通信** | ✅ 支持（SSE） | ❌ 请求-响应 | ✅ 支持 | ❌ 请求-响应 |
| **双向通信** | ✅ Client↔Server | ❌ 单向 | ✅ 支持 | ❌ 单向 |
| **上下文管理** | ✅ 内置 | ❌ 无 | ⚠️ 部分支持 | ❌ 无 |
| **权限控制** | ✅ 内置框架 | ⚠️ 需自行实现 | ⚠️ 部分支持 | ⚠️ 部分支持 |
| **资源访问** | ✅ Resources | ❌ 需定义端点 | ❌ 不支持 | ❌ 不支持 |
| **提示模板** | ✅ Prompts | ❌ 不支持 | ⚠️ 部分支持 | ❌ 不支持 |
| **流式响应** | ✅ 支持 | ⚠️ 需额外实现 | ✅ 支持 | ✅ 支持 |
| **错误处理** | ✅ JSON-RPC 标准 | HTTP 状态码 | 框架特定 | HTTP 状态码 |
| **工具发现** | ✅ 自动发现 | ✅ API 文档 | ⚠️ 需注册 | ✅ 插件市场 |

**深度对比分析**：

1. **MCP vs OpenAPI**
   ```
   OpenAPI: 描述 REST API 的规范
   - 优势：成熟、工具链完善、广泛支持
   - 劣势：不是为 LLM 设计，缺少上下文和状态管理
   
   MCP: 为 LLM 设计的工具集成协议
   - 优势：原生支持 LLM 场景、上下文感知、双向通信
   - 劣势：相对较新、生态仍在发展
   ```

2. **MCP vs LangChain Tools**
   ```
   LangChain Tools: 框架特定的工具抽象
   - 优势：与 LangChain 生态深度集成
   - 劣势：锁定在 LangChain 框架内
   
   MCP: 框架无关的开放协议
   - 优势：可在任何支持 MCP 的应用中使用
   - 劣势：需要额外集成工作
   ```

3. **MCP vs OpenAI Plugins**
   ```
   OpenAI Plugins: ChatGPT 专属插件系统
   - 优势：直接在 ChatGPT 中使用
   - 劣势：仅限 OpenAI 生态
   
   MCP: 跨平台开放标准
   - 优势：支持多个 LLM 应用
   - 劣势：需要各应用分别支持
   ```

#### 核心优势

1. **标准化与互操作性**
   ```
   一次开发，多处使用
   
   开发者：编写一个 MCP Server
   ↓
   可在 Claude Desktop、Cursor、VS Code 等任何支持 MCP 的应用中使用
   ↓
   无需为每个平台重复开发
   ```

2. **上下文感知**
   ```python
   # MCP Server 可以访问完整的上下文信息
   @mcp.tool()
   def analyze_code(context: RequestContext, file_path: str) -> str:
       # 获取当前打开的文件
       # 获取光标位置
       # 获取选中的代码
       # 基于完整上下文进行分析
       pass
   ```

3. **组合性**
   ```
   MCP Server 可以组合使用：
   - 多个 Resources 提供上下文
   - 多个 Tools 执行操作
   - 多个 Prompts 引导用户
   
   形成复杂的工作流
   ```

4. **安全可控**
   ```
   用户完全控制：
   - 哪些工具可以被调用
   - 哪些资源可以被访问
   - 何时使用提示模板
   - 所有操作都需要用户确认
   ```

5. **开发者友好**
   ```
   多种 SDK 支持：
   - Python: pip install mcp
   - TypeScript: npm install @modelcontextprotocol/sdk
   - Go: go get github.com/metoro-io/mcp-golang
   - Java/Kotlin/C#/Rust/Ruby/Swift/PHP: 持续增加中
   ```

### 1.2 MCP 架构

#### Client-Server 模型

MCP 采用 **Client-Server 架构**，但与传统的 C/S 架构有所不同：

```
┌─────────────────────────────────────────────────────────┐
│                         Host                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │              LLM Application                      │   │
│  │  (Claude Desktop / Cursor / VS Code / 自定义应用)  │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         │                               │
│                         ▼                               │
│  ┌──────────────────────────────────────────────────┐   │
│  │                  MCP Client                        │   │
│  │  - 发起连接                                        │   │
│  │  - 管理会话                                        │   │
│  │  - 调用工具                                        │   │
│  │  - 读取资源                                        │   │
│  └──────────────────────┬───────────────────────────┘   │
└─────────────────────────┼───────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │   Transport Layer     │
              │  (Stdio / Streamable  │
              │   HTTP (前身为 SSE))  │
              └───────────┬───────────┘
                          │
┌─────────────────────────┼───────────────────────────────┐
│                         ▼                               │
│  ┌──────────────────────────────────────────────────┐   │
│  │                  MCP Server                        │   │
│  │  - 提供 Resources                                  │   │
│  │  - 提供 Tools                                      │   │
│  │  - 提供 Prompts                                    │   │
│  │  - 处理请求                                        │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│                    External Services                     │
│              (APIs / Databases / Files)                  │
└─────────────────────────────────────────────────────────┘
```

**关键角色**：

1. **Host（主机）**
   - 最终的 LLM 应用程序
   - 用户直接交互的界面
   - 负责管理多个 MCP Client
   - 示例：Claude Desktop、Cursor、VS Code

2. **Client（客户端）**
   - 嵌入在 Host 中的连接器
   - 负责与 Server 建立和维护连接
   - 转发 Host 的请求到 Server
   - 处理 Server 的响应
   - 管理会话状态

3. **Server（服务器）**
   - 独立运行的服务进程
   - 提供 Resources、Tools、Prompts
   - 处理 Client 的请求
   - 访问外部数据源和 API

**通信流程**：

```
用户操作
  ↓
Host (LLM App)
  ↓ 决定需要调用工具
MCP Client
  ↓ 发送 JSON-RPC 请求
Transport Layer
  ↓ 传输消息
MCP Server
  ↓ 执行操作
External Service (API/DB/File)
  ↓ 返回结果
MCP Server
  ↓ 返回 JSON-RPC 响应
Transport Layer
  ↓ 传输响应
MCP Client
  ↓ 解析结果
Host (LLM App)
  ↓ 展示给用户
用户看到结果
```

#### 传输层（HTTP/SSE、Stdio）

MCP 支持两种主要的传输方式：

**1. Stdio 传输**

适用于本地进程间通信，简单高效：

```python
# Python - Stdio Server
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def hello(name: str) -> str:
    return f"Hello, {name}!"

# Stdio 传输通过标准输入输出通信
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

```typescript
// TypeScript - Stdio Server
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new McpServer({
  name: "my-server",
  version: "1.0.0"
});

server.tool("hello", { name: z.string() }, async ({ name }) => ({
  content: [{ type: "text", text: `Hello, ${name}!` }]
}));

// Stdio 传输
const transport = new StdioServerTransport();
await server.connect(transport);
```

**Stdio 传输特点**：
- ✅ 简单：无需配置网络
- ✅ 安全：进程间隔离，不暴露网络端口
- ✅ 高效：直接进程通信
- ❌ 局限：仅支持本地通信
- ❌ 局限：一对一通信

**典型场景**：
```json
// Claude Desktop 配置 - Stdio 传输
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allow"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token"
      }
    }
  }
}
```

### Streamable HTTP 传输层

> ⚠️ **重要更新**: 根据 MCP 2026-07-28 规范，**SSE (Server-Sent Events) 传输层已被弃用**，统一使用 **Streamable HTTP** 作为标准传输协议。

#### Streamable HTTP vs SSE 对比

| 特性 | SSE (已弃用) | Streamable HTTP (推荐) |
|------|-------------|----------------------|
| **连接方向** | 单向（Server→Client） | 双向（Client↔Server） |
| **协议** | HTTP + EventSource | HTTP/2 + Server Push |
| **状态** | 有状态 | 无状态 (Stateless) |
| **断线重连** | 需要手动实现 | 内置支持 |
| **扩展性** | 受限 | 优秀 |

#### Streamable HTTP Server 示例

```python
# 🚀 生产级可执行代码 - Streamable HTTP Server
from mcp.server.fastmcp import FastMCP
import asyncio

mcp = FastMCP(
    "StreamableHTTPServer",
    # 使用 Streamable HTTP 传输（默认）
    transport="streamable-http",
    host="0.0.0.0",
    port=8080
)

@mcp.tool()
def greet(name: str) -> str:
    """打招呼"""
    return f"Hello, {name}!"

@mcp.resource("file://{path}")
def read_file(path: str) -> str:
    """读取文件"""
    from pathlib import Path
    return Path(path).read_text()

if __name__ == "__main__":
    # 启动 Streamable HTTP Server
    mcp.run()
```

#### Streamable HTTP Client 示例

```python
# 🚀 生产级可执行代码 - Streamable HTTP Client
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import asyncio

async def main():
    # 连接到 Streamable HTTP Server
    async with streamablehttp_client("http://localhost:8080") as (
        read_stream,
        write_stream,
        _
    ):
        async with ClientSession(read_stream, write_stream) as session:
            # 初始化连接
            await session.initialize()
            
            # 调用工具
            result = await session.call_tool("greet", {"name": "World"})
            print(f"结果：{result}")
            
            # 读取资源
            content = await session.read_resource("file:///workspace/docs/readme.md")
            print(f"资源内容：{content}")

if __name__ == "__main__":
    asyncio.run(main())
```

#### 断线重连示例

```python
# 🚀 生产级可执行代码
from mcp.client.streamable_http import streamablehttp_client
import asyncio

async def connect_with_retry(url: str, max_retries: int = 3):
    """带重试的 Streamable HTTP 连接"""
    for attempt in range(max_retries):
        try:
            async with streamablehttp_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    print("连接成功！")
                    
                    # 保持连接
                    while True:
                        await asyncio.sleep(60)  # 心跳
                        
        except ConnectionError as e:
            print(f"连接失败（尝试 {attempt + 1}/{max_retries}）：{e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # 指数退避
            else:
                raise

if __name__ == "__main__":
    asyncio.run(connect_with_retry("http://localhost:8080"))
```

#### 从 SSE 迁移到 Streamable HTTP

如果你之前使用 SSE 传输，需要做以下修改：

```python
# ❌ 旧代码（SSE - 已弃用）
from mcp.client.sse import sse_client

async with sse_client("http://localhost:8080/sse") as streams:
    # ...

# ✅ 新代码（Streamable HTTP）
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client("http://localhost:8080") as (read, write, _):
    # ...
```

> 📌 **注意**: Streamable HTTP 不再需要单独的 `/sse` 端点，直接使用基础 URL 即可。

**2. HTTP/SSE 传输（已弃用，请使用 Streamable HTTP）**

适用于远程服务，支持跨网络通信：

```python
# Python - HTTP/SSE Server
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-remote-server")

@mcp.tool()
def search(query: str) -> str:
    # 调用外部 API
    return f"Search results for: {query}"

# HTTP/SSE 传输
if __name__ == "__main__":
    mcp.run(transport="sse", port=8000)
```

```typescript
// TypeScript - HTTP/SSE Server
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import express from "express";

const app = express();

let transport: SSEServerTransport;

app.get("/sse", async (req, res) => {
  transport = new SSEServerTransport("/messages", res);
  await server.connect(transport);
});

app.post("/messages", async (req, res) => {
  await transport.handlePostMessage(req, res);
});

app.listen(3000);
```

**HTTP/SSE 传输特点**：
- ✅ 远程：支持跨网络通信
- ✅ 可扩展：支持多个 Client
- ✅ 实时：SSE 支持服务端推送
- ❌ 复杂：需要配置网络和认证
- ❌ 安全：需要考虑网络安全

**典型场景**：
```
企业部署：
┌─────────────┐      HTTP/SSE      ┌──────────────┐
│  Claude      │ ←────────────────→ │  MCP Server  │
│  Desktop     │                    │  (内部服务)   │
└─────────────┘                    └──────────────┘
                                          ↓
                                    ┌──────────────┐
                                    │  企业内部     │
                                    │  API / DB     │
                                    └──────────────┘

云端部署：
┌─────────────┐      HTTPS         ┌──────────────┐
│  Cursor      │ ←────────────────→ │  MCP Server  │
│  (本地)      │                    │  (云端)       │
└─────────────┘                    └──────────────┘
                                          ↓
                                    ┌──────────────┐
                                    │  云服务       │
                                    │  (AWS/GCP)    │
                                    └──────────────┘
```

**传输层对比**：

| 特性 | Stdio | HTTP/SSE |
|------|-------|----------|
| **通信范围** | 本地进程 | 跨网络 |
| **配置复杂度** | 低 | 中-高 |
| **安全性** | 高（进程隔离） | 需额外配置 |
| **性能** | 高 | 中-高 |
| **可扩展性** | 低（1:1） | 高（1:N） |
| **适用场景** | 桌面应用 | Web/云端 |

#### 消息格式（JSON-RPC 2.0）

MCP 基于 **JSON-RPC 2.0** 规范，所有消息都遵循以下格式：

**1. 请求（Request）**

```typescript
{
  jsonrpc: "2.0";           // 协议版本，必须是 "2.0"
  id: string | number;      // 请求 ID，用于匹配响应
  method: string;           // 方法名
  params?: {                // 参数（可选）
    [key: string]: unknown;
  };
}
```

**请求示例**：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {
      "query": "MCP protocol"
    }
  }
}
```

**关键规则**：
- `id` 必须存在，且不能为 `null`（与标准 JSON-RPC 不同）
- `id` 在同一会话中必须唯一
- `method` 使用 `/` 分隔的命名空间（如 `tools/call`）

**2. 响应（Response）**

```typescript
{
  jsonrpc: "2.0";
  id: string | number;      // 必须与请求的 ID 相同
  result?: {                // 成功结果
    [key: string]: unknown;
  };
  error?: {                 // 错误信息
    code: number;           // 错误码
    message: string;        // 错误消息
    data?: unknown;         // 额外数据
  };
}
```

**成功响应示例**：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "MCP is a protocol for..."
      }
    ]
  }
}
```

**错误响应示例**：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32601,
    "message": "Tool not found: search",
    "data": {
      "availableTools": ["hello", "add"]
    }
  }
}
```

**标准错误码**：

| 错误码 | 含义 | 说明 |
|--------|------|------|
| -32700 | Parse error | JSON 解析错误 |
| -32600 | Invalid Request | 无效的请求 |
| -32601 | Method not found | 方法不存在 |
| -32602 | Invalid params | 参数错误 |
| -32603 | Internal error | 内部错误 |

**3. 通知（Notification）**

```typescript
{
  jsonrpc: "2.0";
  method: string;           // 方法名
  params?: {                // 参数（可选）
    [key: string]: unknown;
  };
}
```

**通知示例**：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/progress",
  "params": {
    "progressToken": "abc123",
    "progress": 50,
    "total": 100
  }
}
```

**通知特点**：
- 没有 `id` 字段（单向消息）
- 接收方不需要响应
- 用于事件通知、进度更新等

**4. 批量消息（Batching）**

```json
[
  {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
  {"jsonrpc": "2.0", "id": 2, "method": "resources/list"}
]
```

**批量响应**：

```json
[
  {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {"tools": [...]}
  },
  {
    "jsonrpc": "2.0",
    "id": 2,
    "result": {"resources": [...]}
  }
]
```

**完整消息流示例**：

```
Client → Server:
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "tools": {}
    },
    "clientInfo": {
      "name": "my-client",
      "version": "1.0.0"
    }
  }
}

Server → Client:
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "tools": {},
      "resources": {}
    },
    "serverInfo": {
      "name": "my-server",
      "version": "1.0.0"
    }
  }
}

Client → Server:
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}

Client → Server:
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {
      "query": "MCP"
    }
  }
}

Server → Client:
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "MCP (Model Context Protocol) is..."
      }
    ]
  }
}
```

#### 生命周期管理

> **⚠️ 重要更新**: MCP 2026-07-28 规范已**移除 initialize/initialized 握手**,协议变为无状态。以下内容分为旧版本（有状态）和新版本（无状态）两部分。

##### MCP 2026-07-28（无状态模式）

新版本中,MCP 连接不再需要握手阶段:

```
┌─────────────┐
│   建立连接   │  (Stdio / Streamable HTTP)
└──────┬──────┘
       ↓
┌─────────────┐
│   正常通信   │  每次请求携带 _meta 中的客户端信息
└──────┬──────┘
       ↓
┌─────────────┐
│   断开连接   │  清理资源
└─────────────┘
```

**无状态请求示例**:

```http
POST /mcp HTTP/1.1
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: search
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {"q": "MCP protocol"},
    "_meta": {
      "io.modelcontextprotocol/clientInfo": {
        "name": "my-client",
        "version": "1.0.0"
      }
    }
  }
}
```

**获取服务器能力**:

```json
// 使用 server/discover 方法（替代 initialize）
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "server/discover"
}

// 响应
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2026-07-28",
    "capabilities": {
      "tools": {},
      "resources": {},
      "extensions": {
        "io.modelcontextprotocol/extensions.tasks": {}
      }
    },
    "serverInfo": {
      "name": "my-server",
      "version": "1.0.0"
    }
  }
}
```

##### MCP 2025-11-25 及更早（有状态模式 - 已弃用）

旧版本中,MCP 连接遵循严格的**生命周期**:

```
┌─────────────┐
│   建立连接   │  (Stdio / HTTP)
└──────┬──────┘
       ↓
┌─────────────┐
│  Initialize  │  协商协议版本和能力
└──────┬──────┘
       ↓
┌─────────────┐
│  Initialized │  通知初始化完成
└──────┬──────┘
       ↓
┌─────────────┐
│   正常通信   │  调用工具、读取资源等
└──────┬──────┘
       ↓
┌─────────────┐
│   断开连接   │  清理资源
└─────────────┘
```

**详细流程**：

**阶段 1：建立连接**

```python
# Client 发起连接
# Stdio: 启动 Server 进程
process = subprocess.Popen(
    ["python", "server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)

# HTTP: 建立 SSE 连接
response = requests.get("http://localhost:8000/sse", stream=True)
```

**阶段 2：初始化（Initialize）**

```json
// Client → Server: initialize
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "tools": {},
      "resources": {}
    },
    "clientInfo": {
      "name": "my-client",
      "version": "1.0.0"
    }
  }
}

// Server → Client: 响应
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "tools": {
        "listChanged": true
      },
      "resources": {
        "subscribe": true,
        "listChanged": true
      }
    },
    "serverInfo": {
      "name": "my-server",
      "version": "1.0.0"
    }
  }
}
```

**能力协商**：

```typescript
// Client 声明支持的能力
{
  "capabilities": {
    "tools": {},           // 支持工具调用
    "resources": {},       // 支持资源访问
    "prompts": {},         // 支持提示模板
    "sampling": {}         // 支持 LLM 采样
  }
}

// Server 声明支持的能力
{
  "capabilities": {
    "tools": {
      "listChanged": true  // 支持工具列表变更通知
    },
    "resources": {
      "subscribe": true,   // 支持资源订阅
      "listChanged": true  // 支持资源列表变更通知
    },
    "prompts": {
      "listChanged": true  // 支持提示列表变更通知
    },
    "logging": {}          // 支持日志
  }
}
```

**阶段 3：初始化完成（Initialized）**

```json
// Client → Server: notifications/initialized
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

**这是一个通知，Server 不需要响应。**

**阶段 4：正常通信**

```json
// 工具调用
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {"query": "MCP"}
  }
}

// 资源读取
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "resources/read",
  "params": {
    "uri": "config://app"
  }
}

// 提示获取
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "prompts/get",
  "params": {
    "name": "code-review",
    "arguments": {"language": "python"}
  }
}
```

**阶段 5：断开连接**

```python
# Stdio: 关闭进程
process.stdin.close()
process.stdout.close()
process.terminate()

# HTTP: 关闭 SSE 连接
response.close()
```

**生命周期最佳实践**：

```python
# Python - 完整的生命周期管理
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

async def main():
    # 1. 建立连接
    async with stdio_client(
        StdioServerParameters(
            command="python",
            args=["server.py"]
        )
    ) as (read, write):
        # 2. 创建会话（自动处理 Initialize）
        async with ClientSession(read, write) as session:
            # 3. 初始化完成，开始通信
            await session.initialize()
            
            # 调用工具
            result = await session.call_tool("search", {"query": "MCP"})
            
            # 读取资源
            resource = await session.read_resource("config://app")
            
            # 4. 会话结束，自动清理
        # 连接自动关闭
```

### 1.3 MCP 核心概念

MCP 定义了三个核心功能：**Resources（资源）**、**Tools（工具）**、**Prompts（提示）**。

#### Resources（资源）

**定义**：

Resources 是 Server 向 Client 提供的**上下文信息和数据**。它们类似于 REST API 中的资源，但更灵活。

**特点**：
- 📖 **只读**：Client 只能读取，不能修改
- 🌐 **URI 标识**：每个资源有唯一的 URI
- 📊 **多种格式**：支持文本、二进制、JSON 等
- 🔄 **可订阅**：支持实时更新

**URI 模板**：

```
协议://路径?参数

示例：
config://app                     # 应用配置
file:///path/to/file.txt         # 文件
db://users/123                   # 数据库记录
api://github/repos/mcp/issues    # API 资源
weather://current?city=beijing   # 带参数的资源
```

**Resource 定义示例**：

```python
# Python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.resource("config://app")
def get_config() -> str:
    """获取应用配置"""
    return """
    {
      "appName": "My App",
      "version": "1.0.0",
      "features": ["tool1", "tool2"]
    }
    """

@mcp.resource("file://{path}")
def read_file(path: str) -> str:
    """读取文件内容"""
    with open(path, 'r') as f:
        return f.read()
```

```typescript
// TypeScript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";

server.resource(
  "config",
  "config://app",
  async (uri) => ({
    contents: [{
      uri: uri.href,
      text: JSON.stringify({ app: "My App", version: "1.0.0" })
    }]
  })
);
```

**Resource 读取流程**：

```
Client → Server: resources/list
Server → Client: 返回资源列表
  {
    "resources": [
      {
        "uri": "config://app",
        "name": "App Config",
        "description": "Application configuration",
        "mimeType": "application/json"
      }
    ]
  }

Client → Server: resources/read (uri: "config://app")
Server → Client: 返回资源内容
  {
    "contents": [
      {
        "uri": "config://app",
        "mimeType": "application/json",
        "text": "{...}"
      }
    ]
  }
```

**Resource 订阅**：

```python
# 订阅资源更新
await session.subscribe_to_resource("config://app")

# 当资源更新时，Server 发送通知
# Server → Client: notifications/resources/updated
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "uri": "config://app"
  }
}

# Client 可以重新读取
resource = await session.read_resource("config://app")
```

#### Tools（工具）

**定义**：

Tools 是 Server 向 Client 提供的**可执行函数**。LLM 可以调用这些工具来执行操作。

**特点**：
- ⚡ **可执行**：调用后执行具体操作
- 📝 **参数 Schema**：使用 JSON Schema 定义参数
- 🔄 **同步/异步**：支持同步和异步执行
- 📊 **结构化返回**：返回结构化的结果

**Tool 定义示例**：

```python
# Python
from mcp.server.fastmcp import FastMCP
from typing import Optional

mcp = FastMCP("my-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """两个数相加"""
    return a + b

@mcp.tool()
def search(
    query: str,
    limit: int = 10,
    source: Optional[str] = None
) -> str:
    """
    搜索信息
    
    Args:
        query: 搜索关键词
        limit: 返回结果数量
        source: 数据源（可选）
    """
    # 执行搜索逻辑
    results = perform_search(query, limit, source)
    return format_results(results)
```

```typescript
// TypeScript
import { z } from "zod";

server.tool(
  "add",
  "两个数相加",
  {
    a: z.number().describe("第一个数"),
    b: z.number().describe("第二个数")
  },
  async ({ a, b }) => ({
    content: [{
      type: "text",
      text: String(a + b)
    }]
  })
);

server.tool(
  "search",
  "搜索信息",
  {
    query: z.string().describe("搜索关键词"),
    limit: z.number().optional().describe("返回结果数量"),
    source: z.string().optional().describe("数据源")
  },
  async ({ query, limit = 10, source }) => {
    const results = await performSearch(query, limit, source);
    return {
      content: [{
        type: "text",
        text: formatResults(results)
      }]
    };
  }
);
```

**Tool 调用流程**：

```
1. Client 获取工具列表
Client → Server: tools/list
Server → Client: 
  {
    "tools": [
      {
        "name": "search",
        "description": "搜索信息",
        "inputSchema": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "搜索关键词"
            },
            "limit": {
              "type": "number",
              "description": "返回结果数量"
            }
          },
          "required": ["query"]
        }
      }
    ]
  }

2. Client 调用工具
Client → Server: tools/call
  {
    "name": "search",
    "arguments": {
      "query": "MCP protocol",
      "limit": 5
    }
  }

3. Server 执行并返回结果
Server → Client:
  {
    "content": [
      {
        "type": "text",
        "text": "MCP (Model Context Protocol) is..."
      }
    ],
    "isError": false
  }
```

**Tool 返回格式**：

```typescript
// 支持的 content 类型
{
  content: [
    { type: "text", text: "文本内容" },
    { type: "image", data: "base64...", mimeType: "image/png" },
    { type: "audio", data: "base64...", mimeType: "audio/mp3" },
    { type: "resource", resource: {...} }
  ],
  isError: false  // 是否错误
}
```

#### Prompts（提示）

**定义**：

Prompts 是 Server 向 Client 提供的**模板化提示**。它们可以包含动态参数，用于生成结构化的提示词。

**特点**：
- 📝 **模板化**：支持参数注入
- 🔄 **动态生成**：根据参数生成不同提示
- 🎯 **预定义工作流**：封装常见任务
- 🔗 **可组合**：多个提示可以组合使用

**Prompt 定义示例**：

```python
# Python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.prompt()
def code_review(language: str = "python") -> str:
    """
    代码审查提示模板
    
    Args:
        language: 编程语言
    """
    return f"""你是一位资深的 {language} 代码审查专家。

请审查以下代码，关注：
1. 代码质量和最佳实践
2. 潜在的性能问题
3. 安全性问题
4. 可维护性建议

请提供具体的改进建议。"""

@mcp.prompt()
def explain_concept(concept: str, level: str = "beginner") -> str:
    """
    解释概念
    
    Args:
        concept: 要解释的概念
        level: 难度级别（beginner/intermediate/advanced）
    """
    level_descriptions = {
        "beginner": "用简单易懂的语言，配合生动的例子",
        "intermediate": "包含一些技术细节，适合有一定基础的读者",
        "advanced": "深入技术细节，适合专业开发者"
    }
    
    return f"""请{level_descriptions[level]}解释"{concept}"这个概念。

要求：
- 定义清晰准确
- 举例说明
- 说明应用场景
- 指出常见误区"""
```

```typescript
// TypeScript
import { PromptTemplate } from "@modelcontextprotocol/sdk/types.js";

server.prompt(
  "code-review",
  { language: z.string().default("python") },
  ({ language }) => ({
    messages: [{
      role: "system",
      content: {
        type: "text",
        text: `你是一位资深的 ${language} 代码审查专家。...`
      }
    }]
  })
);
```

**Prompt 使用流程**：

```
1. Client 获取提示列表
Client → Server: prompts/list
Server → Client:
  {
    "prompts": [
      {
        "name": "code-review",
        "description": "代码审查提示模板",
        "arguments": [
          {
            "name": "language",
            "description": "编程语言",
            "required": false
          }
        ]
      }
    ]
  }

2. Client 获取提示内容
Client → Server: prompts/get
  {
    "name": "code-review",
    "arguments": {
      "language": "typescript"
    }
  }

3. Server 返回生成的提示
Server → Client:
  {
    "messages": [
      {
        "role": "system",
        "content": {
          "type": "text",
          "text": "你是一位资深的 typescript 代码审查专家..."
        }
      }
    ]
  }

4. Client 将提示发送给 LLM
LLM 执行代码审查任务
```

**Prompts 应用场景**：

```python
# 1. 代码审查
@mcp.prompt()
def code_review_prompt(language: str, focus_areas: list[str] = None) -> str:
    areas = focus_areas or ["代码质量", "性能", "安全性", "可维护性"]
    return f"""审查以下 {language} 代码，重点关注：
{chr(10).join(f'- {area}' for area in areas)}
"""

# 2. 文档生成
@mcp.prompt()
def doc_generator(style: str = "google") -> str:
    styles = {
        "google": "Google 风格",
        "sphinx": "Sphinx 风格",
        "javadoc": "Javadoc 风格"
    }
    return f"""为以下代码生成 {styles[style]} 文档注释。
包含：功能描述、参数说明、返回值、异常、示例用法。
"""

# 3. Bug 调试
@mcp.prompt()
def debug_assistant(error_message: str, language: str = "python") -> str:
    return f"""你是一位资深的 {language} 调试专家。

错误信息：{error_message}

请：
1. 分析错误原因
2. 提供解决方案
3. 给出预防措施
4. 解释背后的原理
"""

# 4. 学习辅导
@mcp.prompt()
def tutoring(topic: str, level: str = "beginner") -> str:
    return f"""请作为一名经验丰富的教师，讲解 {topic}。
难度级别：{level}

要求：
- 从基础概念开始
- 逐步深入
- 提供实际例子
- 设计练习题
- 总结关键点
"""
```

---

## 2. MCP 2026-07-28 规范重大更新

> **重要**: 2026年7月28日发布的 MCP 规范是该协议自发布以来最大规模的修订,引入了多项变革性特性。

### 无状态协议核心（Stateless Protocol）

**2026-07-28 规范的核心变革**: MCP 现在是**无状态协议**,不再需要握手和会话管理。

#### 旧版本（2025-11-25）的会话模式

在旧版本中,调用工具需要先建立会话:

```http
POST /mcp HTTP/1.1
Content-Type: application/json

{"jsonrpc":"2.0","id":1,"method":"initialize",
"params":{"protocolVersion":"2025-11-25","capabilities":{},
"clientInfo":{"name":"my-app","version":"1.0"}}}
```

服务器返回 `Mcp-Session-Id`,后续请求必须携带:

```http
POST /mcp HTTP/1.1
Mcp-Session-Id: 1868a90c-3a3f-4f5b
Content-Type: application/json

{"jsonrpc":"2.0","id":2,"method":"tools/call",
"params":{"name":"search","arguments":{"q":"otters"}}}
```

**问题**:
- 需要粘性会话（sticky sessions）
- 需要共享会话存储
- 水平扩展困难
- 负载均衡复杂

#### 新版本（2026-07-28）的无状态模式

新版本中,单个自包含请求可被任何服务器实例处理:

```http
POST /mcp HTTP/1.1
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: search
Content-Type: application/json

{"jsonrpc":"2.0","id":1,"method":"tools/call",
"params":{"name":"search","arguments":{"q":"otters"},
"_meta":{"io.modelcontextprotocol/clientInfo":{"name":"my-app","version":"1.0"}}}}
```

**关键变化**:

1. **移除 initialize/initialized 握手**
   - 协议版本、客户端信息、客户端能力现在在每个请求的 `_meta` 字段中传输
   - 新增 `server/discover` 方法让客户端按需获取服务器能力

2. **移除 Mcp-Session-Id 头部**
   - 任何 MCP 请求都可以落到任何服务器实例
   - 不再需要粘性路由和共享会话存储

3. **新增 HTTP 头部**
   - `MCP-Protocol-Version`: 协议版本
   - `Mcp-Method`: 方法名（如 `tools/call`）
   - `Mcp-Name`: 工具/资源名称
   - 负载均衡器可基于这些头部路由,无需检查请求体

#### 无状态协议,有状态应用

移除协议层会话**不意味着**应用必须是无状态的。服务器可以:

```python
# 服务器创建显式句柄,模型在后续调用中传递
@mcp.tool()
def create_browser_session() -> dict:
    """创建浏览器会话,返回 basket_id"""
    basket_id = generate_unique_id()
    store_session_state(basket_id, initial_state())
    return {"basket_id": basket_id, "status": "created"}

@mcp.tool()
def navigate_browser(basket_id: str, url: str) -> dict:
    """使用 basket_id 操作浏览器"""
    state = get_session_state(basket_id)
    # 执行导航操作
    return {"status": "navigated", "url": url}
```

**优势**:
- 模型可以跨工具组合句柄
- 模型可以推理状态
- 状态对模型可见,而非隐藏在传输元数据中

### 扩展机制（Extensions）

2026-07-28 规范正式引入**扩展机制**,使新功能可以独立于核心规范演进。

#### 扩展特性

- **标识**: 使用反向 DNS 标识符（如 `io.modelcontextprotocol/extensions.tasks`）
- **协商**: 通过客户端和服务器能力的 `extensions` 映射协商
- **版本**: 扩展独立于核心规范版本化
- **仓库**: 存储在 `ext-*` 仓库,由委托维护者管理

#### 官方扩展 1: MCP Apps

MCP Apps 允许服务器发送交互式 HTML 界面,主机在沙盒 iframe 中渲染:

```python
# 工具声明 UI 模板
@mcp.tool()
def search_database(query: str) -> dict:
    """搜索数据库并返回结果"""
    results = execute_search(query)
    return {
        "content": [
            {
                "type": "text",
                "text": f"Found {len(results)} results"
            },
            {
                "type": "app",
                "template": "search-results-table",
                "data": {"results": results}
            }
        ]
    }
```

**特性**:
- 服务器预先声明 UI 模板
- 主机可预取、缓存、安全审查
- UI 通过相同的 JSON-RPC 协议与主机通信
- 所有 UI 操作经过相同的审计和同意路径

#### 官方扩展 2: Tasks

Tasks 从实验性核心特性迁移为扩展,围绕无状态模型重新设计:

```python
# 服务器返回任务句柄
@mcp.tool()
def train_model(dataset: str, epochs: int) -> dict:
    """启动模型训练,返回任务句柄"""
    task_id = create_training_task(dataset, epochs)
    return {
        "content": [{"type": "text", "text": "Training started"}],
        "task": {
            "taskId": task_id,
            "status": "running"
        }
    }

# 客户端使用 tasks/get、tasks/update、tasks/cancel 驱动任务
```

**任务生命周期**:
```
tools/call → 返回 task handle
    ↓
tasks/get → 查询任务状态
    ↓
tasks/update → 接收任务更新
    ↓
任务完成或 tasks/cancel → 取消任务
```

**变化**:
- `tasks/list` 被移除（无法在无会话情况下安全作用域）
- 任务创建是服务器导向的
- 客户端决定何时将调用作为任务运行

### 授权强化（Authorization Hardening）

六个 SEP 强化授权规范,使其更符合 OAuth 2.0 和 OpenID Connect 的实际部署:

1. **RFC 9207 iss 参数验证**
   - 客户端必须验证授权响应中的 `iss` 参数
   - 防止 mix-up 攻击（在 MCP 的单客户端多服务器部署模式中更常见）

2. **OpenID Connect application_type 声明**
   - 客户端在动态客户端注册时声明应用类型
   - 避免授权服务器默认桌面/CLI 客户端为 "web" 类型

3. **凭证绑定**
   - 客户端将注册的凭证绑定到授权服务器的 issuer
   - 资源在授权服务器间迁移时需重新注册

4. **刷新令牌**
   - 规范记录了如何从 OpenID Connect 风格的授权服务器请求刷新令牌

5. **范围累积**
   - 明确了 step-up 认证时的范围累积规则

6. **.well-known 发现后缀**
   - 澄清了服务发现的 URL 路径

### 弃用特性

根据新的**特性生命周期政策**（SEP-2577）,以下特性被标记为弃用:

| 特性 | 替代方案 | 状态 |
|------|---------|------|
| **Roots** | 工具参数、资源 URI、服务器配置 | 弃用 |
| **Sampling** | 直接集成 LLM 提供商 API | 弃用 |
| **Logging** | stderr（stdio 传输）；OpenTelemetry（结构化可观测性） | 弃用 |

**重要**: 这些是仅注解的弃用。方法、类型和能力标志在此版本中继续工作,移除需要单独的 SEP。

### 完整 JSON Schema 2020-12 支持

工具的 `inputSchema` 和 `outputSchema` 提升为完整的 **JSON Schema 2020-12**:

**输入 Schema**:
- 保持 `type: "object"` 根约束
- 现在允许组合（`oneOf`、`anyOf`、`allOf`）
- 支持条件（`if`/`then`/`else`）
- 支持引用（`$ref`、`$defs`）

**输出 Schema**:
- 无限制
- `structuredContent` 现在可以是任何 JSON 值（不仅限于对象）

**示例**:

```python
@mcp.tool()
def process_data(
    data: Union[TextData, ImageData],  # oneOf 支持
    config: Optional[ProcessingConfig] = None  # 条件支持
) -> dict:
    """处理数据,支持多种输入类型"""
    pass
```

**安全要求**:
- 实现不得自动解引用外部 `$ref` URI
- 应限制 schema 深度和验证时间

### 缓存支持

列表和资源读取结果现在携带 `ttlMs` 和 `cacheScope`:

```json
{
  "result": {
    "tools": [...],
    "ttlMs": 300000,
    "cacheScope": "user"
  }
}
```

**特性**:
- 客户端确切知道 `tools/list` 响应的 freshness 时间
- `cacheScope` 指示是否可以跨用户共享
-  modeled on HTTP Cache-Control
- 不再需要长寿命 SSE 流来获知列表变化

### W3C Trace Context 传播

`_meta` 中的 W3C Trace Context 传播现已文档化:

- `traceparent`: 追踪父级
- `tracestate`: 追踪状态
- `baggage`: 附加数据

**优势**:
- 分布式追踪跨 SDK 和网关关联
- 与 OpenTelemetry 兼容后端集成
- 追踪从主机应用开始,经过客户端 SDK、MCP 服务器、下游调用

### 版本对比

| 特性 | 2025-11-25 | 2026-07-28 |
|------|-----------|-----------|
| **协议状态** | 有状态（会话） | 无状态 |
| **握手** | initialize/initialized | 移除,使用 `_meta` |
| **会话 ID** | `Mcp-Session-Id` | 移除 |
| **路由头部** | 无 | `Mcp-Method`、`Mcp-Name` |
| **扩展** | 非正式 | 正式机制 |
| **MCP Apps** | 不支持 | 官方扩展 |
| **Tasks** | 实验性核心 | 官方扩展 |
| **JSON Schema** | 子集 | 完整 2020-12 |
| **缓存** | 不支持 | `ttlMs` + `cacheScope` |
| **追踪** | 非标准 | W3C Trace Context |
| **授权** | 基础 | OAuth 2.0 + OIDC 强化 |

### 迁移指南

**从 2025-11-25 迁移到 2026-07-28**:

1. **移除初始化握手**
   ```python
   # 旧代码
   await client.initialize()
   await client.send_initialized()
   
   # 新代码 - 不需要,客户端信息在每次请求的 _meta 中
   ```

2. **移除会话管理**
   ```python
   # 旧代码
   session_id = response.headers["Mcp-Session-Id"]
   headers["Mcp-Session-Id"] = session_id
   
   # 新代码 - 添加路由头部
   headers["Mcp-Method"] = "tools/call"
   headers["Mcp-Name"] = tool_name
   ```

3. **更新工具 Schema**
   ```python
   # 旧代码 - 简单 schema
   input_schema = {
       "type": "object",
       "properties": {"query": {"type": "string"}}
   }
   
   # 新代码 - 完整 JSON Schema 2020-12
   input_schema = {
       "$schema": "https://json-schema.org/draft/2020-12/schema",
       "type": "object",
       "properties": {
           "query": {
               "oneOf": [
                   {"type": "string"},
                   {"type": "array", "items": {"type": "string"}}
               ]
           }
       },
       "$defs": {...}
   }
   ```

4. **迁移 Tasks API**（如果使用）
   ```python
   # 旧代码 - 2025-11-25
   task = await client.tasks_create(...)
   tasks = await client.tasks_list()
   
   # 新代码 - 2026-07-28 扩展
   # 通过 tools/call 返回任务句柄
   # 使用 tasks/get、tasks/update、tasks/cancel
   # tasks/list 已移除
   ```

---

## 3. MCP 协议详解

### 3.1 初始化流程

MCP 的初始化是一个**三阶段握手过程**，确保 Client 和 Server 在兼容的协议版本和能力下工作。

#### 完整的初始化序列

```
时序图：

Client                          Server
  │                               │
  │──── initialize (1) ──────────→│
  │                               │
  │←──── response (1) ────────────│
  │                               │
  │──── notifications/initialized →│
  │                               │
  │──── capabilities exchange ───→│
  │                               │
  │          正常通信开始           │
  │←──── tools/list ──────────────│
  │──── result ──────────────────→│
  │←──── resources/list ──────────│
  │──── result ──────────────────→│
```

#### 阶段 1：Initialize 请求

**Client → Server**：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "tools": {},
      "resources": {
        "subscribe": true
      },
      "prompts": {},
      "sampling": {
        "maxTokens": 4096
      }
    },
    "clientInfo": {
      "name": "claude-desktop",
      "version": "1.0.0"
    }
  }
}
```

**字段说明**：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `protocolVersion` | string | ✅ | Client 支持的协议版本 |
| `capabilities` | object | ✅ | Client 声明支持的能力 |
| `clientInfo` | object | ✅ | Client 的元信息 |
| `capabilities.tools` | object | ❌ | 支持工具调用 |
| `capabilities.resources` | object | ❌ | 支持资源访问 |
| `capabilities.prompts` | object | ❌ | 支持提示模板 |
| `capabilities.sampling` | object | ❌ | 支持 LLM 采样 |

**协议版本格式**：

```
YYYY-MM-DD

示例：
"2025-03-26"  # 2025 年 3 月 26 日版本
"2024-11-05"  # 2024 年 11 月 5 日版本
```

#### 阶段 2：Initialize 响应

**Server → Client**：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "tools": {
        "listChanged": true
      },
      "resources": {
        "subscribe": true,
        "listChanged": true
      },
      "prompts": {
        "listChanged": true
      },
      "logging": {}
    },
    "serverInfo": {
      "name": "my-mcp-server",
      "version": "1.0.0"
    },
    "instructions": "This server provides tools for searching and analyzing data."
  }
}
```

**字段说明**：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `protocolVersion` | string | ✅ | Server 选择的协议版本 |
| `capabilities` | object | ✅ | Server 声明支持的能力 |
| `serverInfo` | object | ✅ | Server 的元信息 |
| `instructions` | string | ❌ | 使用指南 |

**版本协商规则**：

```
规则 1：Server 必须响应 Client 支持的版本
规则 2：如果不支持，Server 可以拒绝连接
规则 3：Server 可以选择更早的版本（向后兼容）

示例：
Client 支持：["2025-03-26", "2024-11-05"]
Server 支持：["2025-03-26", "2025-01-15"]
协商结果："2025-03-26"（共同支持的最新版本）
```

#### 阶段 3：Initialized 通知

**Client → Server**：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

**这是一个单向通知，Server 不需要响应。**

**语义**：
- Client 已收到 Server 的能力声明
- Client 已准备好开始正常通信
- Server 现在可以提供完整的功能

#### 初始化最佳实践

**Python 实现**：

```python
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
import asyncio

async def initialize_example():
    # 配置 Server 参数
    server_params = StdioServerParameters(
        command="python",
        args=["my_server.py"],
        env={"DEBUG": "true"}  # 可选的环境变量
    )
    
    # 建立连接
    async with stdio_client(server_params) as (read, write):
        # 创建会话（自动处理初始化）
        async with ClientSession(read, write) as session:
            # 初始化（自动发送 initialize 和 initialized）
            init_result = await session.initialize()
            
            print(f"协议版本: {init_result.protocolVersion}")
            print(f"Server: {init_result.serverInfo.name}")
            print(f"Server 能力: {init_result.capabilities}")
            
            # 现在可以正常通信了
            tools = await session.list_tools()
            print(f"可用工具: {[t.name for t in tools.tools]}")

asyncio.run(initialize_example())
```

**TypeScript 实现**：

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

async function initializeExample() {
  // 创建传输层
  const transport = new StdioClientTransport({
    command: "node",
    args: ["my_server.js"],
    env: { DEBUG: "true" }
  });
  
  // 创建 Client
  const client = new Client({
    name: "my-client",
    version: "1.0.0"
  }, {
    capabilities: {
      tools: {},
      resources: { subscribe: true }
    }
  });
  
  // 连接（自动初始化）
  await client.connect(transport);
  
  console.log("已连接");
  
  // 获取工具列表
  const tools = await client.listTools();
  console.log("可用工具:", tools.tools.map(t => t.name));
}

initializeExample();
```

**错误处理**：

```python
# Python - 处理初始化错误
try:
    async with ClientSession(read, write) as session:
        init_result = await session.initialize()
except McpError as e:
    if e.code == -32602:
        print("参数错误")
    elif e.code == -32601:
        print("不支持的方法")
    elif e.code == -32603:
        print("内部错误")
    else:
        print(f"未知错误: {e}")
except Exception as e:
    print(f"连接失败: {e}")
```

### 3.2 资源管理

#### Resource 定义

Resource 是 MCP Server 提供的**只读数据源**，用于向 LLM 提供上下文信息。

**Resource 数据结构**：

```typescript
interface Resource {
  uri: string;              // 资源的唯一 URI
  name: string;             // 人类可读的名称
  description?: string;     // 描述
  mimeType?: string;        // MIME 类型
  size?: number;            // 大小（字节）
}
```

**ResourceContent 数据结构**：

```typescript
interface ResourceContent {
  uri: string;              // 资源的 URI
  mimeType?: string;        // MIME 类型
  text?: string;            // 文本内容
  blob?: string;            // 二进制内容（base64）
}
```

**支持的 MIME 类型**：

```
文本类型：
- text/plain          纯文本
- text/html           HTML
- text/markdown       Markdown
- application/json    JSON
- application/xml     XML
- text/csv            CSV

二进制类型：
- image/png           PNG 图像
- image/jpeg          JPEG 图像
- image/gif           GIF 图像
- application/pdf     PDF 文档
- audio/mpeg          MP3 音频
- video/mp4           MP4 视频
```

#### URI 模板

MCP 支持**动态 URI 模板**，允许参数化资源访问。

**URI 格式**：

```
协议://路径?参数#片段

示例：
静态 URI：
  config://app
  file:///etc/hosts
  
动态 URI（模板）：
  file://{path}
  db://users/{userId}
  api://github/repos/{owner}/{repo}
  weather://current?city={city}&country={country}
```

**Python 实现**：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

# 静态资源
@mcp.resource("config://app")
def get_app_config() -> str:
    """应用配置"""
    return """
    {
      "appName": "My Application",
      "version": "1.0.0",
      "features": ["search", "analyze"]
    }
    """

# 动态资源（带参数）
@mcp.resource("file://{path}")
def read_file(path: str) -> str:
    """
    读取文件内容
    
    Args:
        path: 文件路径
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f"文件不存在: {path}")
    except PermissionError:
        raise ValueError(f"无权限访问: {path}")

# 多个参数
@mcp.resource("weather://current?city={city}&country={country}")
def get_weather(city: str, country: str = "CN") -> str:
    """
    获取天气信息
    
    Args:
        city: 城市名称
        country: 国家代码（默认 CN）
    """
    # 调用天气 API
    weather_data = fetch_weather_api(city, country)
    return format_weather(weather_data)

# URI 带查询参数
@mcp.resource("db://users/{user_id}?include={fields}")
def get_user(user_id: str, fields: str = "*") -> str:
    """
    获取用户信息
    
    Args:
        user_id: 用户 ID
        fields: 字段列表（逗号分隔，默认全部）
    """
    field_list = fields.split(",") if fields != "*" else None
    user = fetch_user_from_db(user_id, field_list)
    return user.to_json()
```

**TypeScript 实现**：

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

const server = new McpServer({
  name: "my-server",
  version: "1.0.0"
});

// 静态资源
server.resource(
  "app-config",
  "config://app",
  async (uri) => ({
    contents: [{
      uri: uri.href,
      mimeType: "application/json",
      text: JSON.stringify({
        appName: "My Application",
        version: "1.0.0"
      })
    }]
  })
);

// 动态资源
server.resource(
  "file",
  "file://{path}",
  {
    path: z.string().describe("文件路径")
  },
  async (uri, { path }) => {
    const content = await readFile(path);
    return {
      contents: [{
        uri: uri.href,
        mimeType: "text/plain",
        text: content
      }]
    };
  }
);

// 多参数资源
server.resource(
  "weather",
  "weather://current?city={city}&country={country}",
  {
    city: z.string().describe("城市名称"),
    country: z.string().default("CN").describe("国家代码")
  },
  async (uri, { city, country }) => {
    const weather = await fetchWeather(city, country);
    return {
      contents: [{
        uri: uri.href,
        mimeType: "application/json",
        text: JSON.stringify(weather)
      }]
    };
  }
);
```

#### 读取机制

**资源读取流程**：

```
1. 获取资源列表
Client → Server: resources/list
Server → Client:
  {
    "resources": [
      {
        "uri": "config://app",
        "name": "App Config",
        "description": "Application configuration",
        "mimeType": "application/json"
      },
      {
        "uri": "file://{path}",
        "name": "File Reader",
        "description": "Read file contents",
        "mimeType": "text/plain"
      }
    ]
  }

2. 读取具体资源
Client → Server: resources/read
  {
    "uri": "config://app"
  }

Server → Client:
  {
    "contents": [
      {
        "uri": "config://app",
        "mimeType": "application/json",
        "text": "{...}"
      }
    ]
  }

3. 读取带参数的资源
Client → Server: resources/read
  {
    "uri": "file:///etc/hosts"
  }

Server → Client:
  {
    "contents": [
      {
        "uri": "file:///etc/hosts",
        "mimeType": "text/plain",
        "text": "127.0.0.1 localhost\n..."
      }
    ]
  }
```

**Python Client 读取示例**：

```python
from mcp.client.session import ClientSession

async def read_resources(session: ClientSession):
    # 获取资源列表
    resources = await session.list_resources()
    print("可用资源:")
    for r in resources.resources:
        print(f"  - {r.uri} ({r.name})")
    
    # 读取资源
    config = await session.read_resource("config://app")
    print("配置:", config.contents[0].text)
    
    # 读取文件
    file_content = await session.read_resource("file:///etc/hosts")
    print("文件内容:", file_content.contents[0].text)
```

**TypeScript Client 读取示例**：

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";

async function readResources(client: Client) {
  // 获取资源列表
  const resources = await client.listResources();
  console.log("可用资源:");
  resources.resources.forEach(r => {
    console.log(`  - ${r.uri} (${r.name})`);
  });
  
  // 读取资源
  const config = await client.readResource("config://app");
  console.log("配置:", config.contents[0].text);
}
```

#### 订阅更新

MCP 支持**资源订阅**，当资源发生变化时 Server 会主动通知 Client。

**订阅流程**：

```
1. Client 订阅资源
Client → Server: resources/subscribe
  {
    "uri": "config://app"
  }

2. 资源发生变化
Server 更新资源内容

3. Server 发送更新通知
Server → Client: notifications/resources/updated
  {
    "uri": "config://app"
  }

4. Client 重新读取资源
Client → Server: resources/read
  {
    "uri": "config://app"
  }

Server → Client: 返回最新内容
```

**Python 订阅示例**：

```python
from mcp.client.session import ClientSession
import asyncio

async def subscribe_example(session: ClientSession):
    # 订阅资源
    await session.subscribe_resource("config://app")
    print("已订阅 config://app")
    
    # 监听更新（在实际应用中）
    # 这需要实现消息处理循环
    async def handle_notifications():
        async for notification in session.incoming_messages:
            if notification.method == "notifications/resources/updated":
                uri = notification.params.uri
                print(f"资源已更新: {uri}")
                
                # 重新读取
                resource = await session.read_resource(uri)
                print(f"新内容: {resource.contents[0].text}")
    
    # 启动监听
    asyncio.create_task(handle_notifications())
    
    # 保持运行
    await asyncio.sleep(3600)  # 1 小时
```

**TypeScript 订阅示例**：

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";

async function subscribeExample(client: Client) {
  // 订阅资源
  await client.subscribe("config://app");
  console.log("已订阅 config://app");
  
  // 监听更新
  client.setNotificationHandler(
    "notifications/resources/updated",
    async (notification) => {
      const uri = notification.params.uri;
      console.log(`资源已更新: ${uri}`);
      
      // 重新读取
      const resource = await client.readResource(uri);
      console.log("新内容:", resource.contents[0].text);
    }
  );
}
```

**使用场景**：

```python
# 场景 1：配置文件监控
@mcp.resource("config://app")
@watch_file("config.yaml")  # 装饰器监听文件变化
def get_config() -> str:
    return load_config("config.yaml")

# 场景 2：实时数据流
@mcp.resource("sensor://temperature")
@poll(interval=5)  # 每 5 秒更新
def get_temperature() -> str:
    return read_sensor()

# 场景 3：数据库变更
@mcp.resource("db://users/{id}")
@on_table_change("users")  # 监听表变更
def get_user(id: str) -> str:
    return fetch_user(id)
```

### 3.3 工具调用

#### Tool 定义

Tool 是 MCP 的核心功能，允许 LLM **执行操作**和**访问外部服务**。

**Tool 数据结构**：

```typescript
interface Tool {
  name: string;                      // 工具名称
  description: string;               // 工具描述
  inputSchema: {                     // 参数 Schema（JSON Schema）
    type: "object";
    properties?: {
      [key: string]: {
        type: string;                // 参数类型
        description?: string;        // 参数描述
        enum?: any[];                // 枚举值
        default?: any;               // 默认值
      };
    };
    required?: string[];             // 必需参数
  };
}
```

**完整的 Tool 定义**：

```python
# Python - 使用装饰器
from mcp.server.fastmcp import FastMCP
from typing import Optional, Literal

mcp = FastMCP("my-server")

@mcp.tool()
def search(
    query: str,
    source: Literal["web", "docs", "code"] = "web",
    limit: int = 10,
    language: Optional[str] = None
) -> str:
    """
    搜索信息
    
    Args:
        query: 搜索关键词（必需）
        source: 数据源（web/docs/code，默认 web）
        limit: 返回结果数量（1-50，默认 10）
        language: 语言过滤器（可选）
    
    Returns:
        格式化的搜索结果
    """
    # 参数验证
    if not query.strip():
        raise ValueError("搜索关键词不能为空")
    
    if limit < 1 or limit > 50:
        raise ValueError("limit 必须在 1-50 之间")
    
    # 执行搜索
    results = perform_search(
        query=query,
        source=source,
        limit=limit,
        language=language
    )
    
    # 格式化结果
    return format_results(results)
```

```typescript
// TypeScript - 使用 Zod Schema
import { z } from "zod";

server.tool(
  "search",
  "搜索信息",
  {
    query: z.string()
      .min(1, "搜索关键词不能为空")
      .describe("搜索关键词"),
    
    source: z.enum(["web", "docs", "code"])
      .default("web")
      .describe("数据源"),
    
    limit: z.number()
      .min(1)
      .max(50)
      .default(10)
      .describe("返回结果数量"),
    
    language: z.string()
      .optional()
      .describe("语言过滤器")
  },
  async ({ query, source, limit, language }) => {
    // 执行搜索
    const results = await performSearch({
      query,
      source,
      limit,
      language
    });
    
    return {
      content: [{
        type: "text",
        text: formatResults(results)
      }]
    };
  }
);
```

#### 参数 Schema

MCP 使用 **JSON Schema** 定义工具参数，确保类型安全和验证。

**基本类型**：

```json
{
  "string": {
    "type": "string",
    "description": "字符串类型",
    "minLength": 1,
    "maxLength": 100,
    "pattern": "^[a-zA-Z]+$"
  },
  "number": {
    "type": "number",
    "description": "数字类型",
    "minimum": 0,
    "maximum": 100,
    "multipleOf": 0.5
  },
  "integer": {
    "type": "integer",
    "description": "整数类型",
    "minimum": 1,
    "maximum": 10
  },
  "boolean": {
    "type": "boolean",
    "description": "布尔类型"
  }
}
```

**复杂类型**：

```json
{
  "enum": {
    "type": "string",
    "enum": ["red", "green", "blue"],
    "description": "枚举类型"
  },
  "array": {
    "type": "array",
    "items": {
      "type": "string"
    },
    "minItems": 1,
    "maxItems": 10,
    "description": "字符串数组"
  },
  "object": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string"
      },
      "age": {
        "type": "integer"
      }
    },
    "required": ["name"],
    "description": "对象类型"
  },
  "nullable": {
    "type": ["string", "null"],
    "description": "可空类型"
  }
}
```

**Python 类型注解自动转换**：

```python
from typing import Optional, Literal, list

@mcp.tool()
def complex_params(
    name: str,                          # string
    age: int,                           # integer
    score: float,                       # number
    active: bool,                       # boolean
    color: Literal["red", "green"],    # enum
    tags: list[str],                    # array
    metadata: Optional[dict] = None     # object, nullable
) -> str:
    """复杂参数示例"""
    return f"{name}, {age}, {score}"
```

**生成的 JSON Schema**：

```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "name"
    },
    "age": {
      "type": "integer",
      "description": "age"
    },
    "score": {
      "type": "number",
      "description": "score"
    },
    "active": {
      "type": "boolean",
      "description": "active"
    },
    "color": {
      "type": "string",
      "enum": ["red", "green"],
      "description": "color"
    },
    "tags": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "tags"
    },
    "metadata": {
      "type": ["object", "null"],
      "description": "metadata"
    }
  },
  "required": ["name", "age", "score", "active", "color", "tags"]
}
```

#### 调用流程

**完整的工具调用序列**：

```
1. 获取工具列表
Client → Server: tools/list

Server → Client:
  {
    "tools": [
      {
        "name": "search",
        "description": "搜索信息",
        "inputSchema": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "搜索关键词"
            },
            "source": {
              "type": "string",
              "enum": ["web", "docs", "code"],
              "default": "web"
            },
            "limit": {
              "type": "integer",
              "default": 10
            }
          },
          "required": ["query"]
        }
      },
      {
        "name": "analyze",
        "description": "分析数据",
        "inputSchema": {
          "type": "object",
          "properties": {
            "data": {
              "type": "string",
              "description": "要分析的数据"
            },
            "type": {
              "type": "string",
              "enum": ["sentiment", "summary", "keywords"]
            }
          },
          "required": ["data", "type"]
        }
      }
    ]
  }

2. 调用工具
LLM 决定调用 search 工具

Client → Server: tools/call
  {
    "name": "search",
    "arguments": {
      "query": "MCP protocol",
      "source": "web",
      "limit": 5
    }
  }

3. Server 执行工具
Server:
  - 验证参数
  - 执行搜索逻辑
  - 格式化结果

4. 返回结果
Server → Client:
  {
    "content": [
      {
        "type": "text",
        "text": "搜索结果：\n1. MCP 是...\n2. MCP 提供...\n..."
      }
    ],
    "isError": false
  }

5. Client 将结果给 LLM
LLM 基于结果生成回复
```

**错误处理流程**：

```json
// 工具执行错误
Server → Client:
{
  "content": [
    {
      "type": "text",
      "text": "搜索失败：API 服务不可用"
    }
  ],
  "isError": true
}

// 参数错误
Server → Client:
{
  "error": {
    "code": -32602,
    "message": "Invalid params: query is required"
  }
}

// 工具不存在
Server → Client:
{
  "error": {
    "code": -32601,
    "message": "Tool not found: unknown_tool"
  }
}
```

#### 结果返回

**支持的内容类型**：

```python
# Python - 返回不同类型内容
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def get_text() -> str:
    """返回文本"""
    return "这是文本内容"

@mcp.tool()
def get_image() -> dict:
    """返回图像"""
    import base64
    with open("image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    return {
        "content": [
            {
                "type": "image",
                "data": image_data,
                "mimeType": "image/png"
            }
        ]
    }

@mcp.tool()
def get_multiple() -> dict:
    """返回多种类型内容"""
    return {
        "content": [
            {
                "type": "text",
                "text": "这是分析结果："
            },
            {
                "type": "image",
                "data": "base64...",
                "mimeType": "image/png"
            },
            {
                "type": "resource",
                "resource": {
                    "uri": "analysis://result",
                    "mimeType": "application/json",
                    "text": '{"score": 0.95}'
                }
            }
        ]
    }
```

```typescript
// TypeScript - 返回不同类型内容
server.tool(
  "get-text",
  "返回文本",
  {},
  async () => ({
    content: [{
      type: "text",
      text: "这是文本内容"
    }]
  })
);

server.tool(
  "get-image",
  "返回图像",
  {},
  async () => {
    const imageBuffer = await readFile("image.png");
    return {
      content: [{
        type: "image",
        data: imageBuffer.toString("base64"),
        mimeType: "image/png"
      }]
    };
  }
);

server.tool(
  "get-multiple",
  "返回多种类型",
  {},
  async () => ({
    content: [
      {
        type: "text",
        text: "这是分析结果："
      },
      {
        type: "image",
        data: "...",
        mimeType: "image/png"
      },
      {
        type: "resource",
        resource: {
          uri: "analysis://result",
          mimeType: "application/json",
          text: JSON.stringify({ score: 0.95 })
        }
      }
    ]
  })
);
```

**进度报告**：

```python
# Python - 长时间任务的进度报告
from mcp.server.fastmcp import FastMCP
from mcp.server.context import RequestContext

mcp = FastMCP("my-server")

@mcp.tool()
async def long_task(
    context: RequestContext,
    total_items: int
) -> str:
    """
    长时间运行的任务，带进度报告
    
    Args:
        total_items: 总项目数
    """
    results = []
    
    for i in range(total_items):
        # 处理项目
        result = await process_item(i)
        results.append(result)
        
        # 报告进度
        await context.report_progress(
            progress=i + 1,
            total=total_items
        )
    
    return f"处理完成，共 {len(results)} 个项目"
```

```typescript
// TypeScript - 长时间任务的进度报告
server.tool(
  "long-task",
  "长时间运行的任务",
  {
    totalItems: z.number()
  },
  async ({ totalItems }, { reportProgress }) => {
    const results = [];
    
    for (let i = 0; i < totalItems; i++) {
      // 处理项目
      const result = await processItem(i);
      results.push(result);
      
      // 报告进度
      await reportProgress({
        progress: i + 1,
        total: totalItems
      });
    }
    
    return {
      content: [{
        type: "text",
        text: `处理完成，共 ${results.length} 个项目`
      }]
    };
  }
);
```

### 3.4 提示系统

#### Prompt 模板

Prompt 是 MCP 的**模板化提示系统**，允许 Server 定义可复用的提示模板。

**Prompt 数据结构**：

```typescript
interface Prompt {
  name: string;                      // 提示名称
  description?: string;              // 描述
  arguments?: [                      // 参数列表
    {
      name: string;                  // 参数名
      description?: string;          // 描述
      required?: boolean;            // 是否必需
    }
  ];
}
```

**Prompt Message 数据结构**：

```typescript
interface PromptMessage {
  role: "user" | "assistant" | "system";  // 角色
  content: {
    type: "text" | "image";               // 内容类型
    text?: string;                        // 文本内容
    data?: string;                        // 图像数据（base64）
    mimeType?: string;                    // MIME 类型
  };
}
```

**完整的 Prompt 定义**：

```python
# Python - 完整的 Prompt 示例
from mcp.server.fastmcp import FastMCP
from typing import Literal

mcp = FastMCP("my-server")

@mcp.prompt()
def code_review(
    language: str = "python",
    focus_areas: Literal["quality", "performance", "security", "all"] = "all"
) -> str:
    """
    代码审查提示
    
    Args:
        language: 编程语言
        focus_areas: 关注领域
    """
    area_map = {
        "quality": "代码质量、命名规范、最佳实践",
        "performance": "性能优化、算法复杂度、内存使用",
        "security": "安全漏洞、输入验证、敏感信息处理",
        "all": "代码质量、性能、安全性、可维护性"
    }
    
    return f"""你是一位资深的 {language} 开发专家。

请审查以下代码，重点关注：
{area_map[focus_areas]}

要求：
1. 指出具体问题
2. 解释原因
3. 提供改进建议
4. 给出最佳实践示例

代码：
```{language}
{{code}}
```"""

@mcp.prompt()
def explain_code(
    language: str = "python",
    level: Literal["beginner", "intermediate", "advanced"] = "intermediate"
) -> list[dict]:
    """
    解释代码
    
    Args:
        language: 编程语言
        level: 解释深度
    """
    level_prompts = {
        "beginner": "用简单易懂的语言，避免技术术语",
        "intermediate": "适当使用技术术语，但提供解释",
        "advanced": "深入技术细节，讨论底层原理"
    }
    
    return [
        {
            "role": "system",
            "content": {
                "type": "text",
                "text": f"你是一位 {language} 教师。{level_prompts[level]}"
            }
        },
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": "请解释以下代码的工作原理：\n```{language}\n{code}\n```"
            }
        }
    ]

@mcp.prompt()
def write_tests(
    framework: str = "pytest",
    test_type: Literal["unit", "integration", "e2e"] = "unit"
) -> str:
    """
    编写测试用例
    
    Args:
        framework: 测试框架
        test_type: 测试类型
    """
    return f"""请使用 {framework} 为以下代码编写 {test_type} 测试。

要求：
1. 覆盖正常流程
2. 覆盖边界情况
3. 覆盖异常流程
4. 包含必要的 mock
5. 遵循测试最佳实践

代码：
```
{code}
```"""
```

```typescript
// TypeScript - 完整的 Prompt 示例
import { z } from "zod";

server.prompt(
  "code-review",
  {
    language: z.string().default("python"),
    focusAreas: z.enum(["quality", "performance", "security", "all"])
      .default("all")
  },
  ({ language, focusAreas }) => {
    const areaMap = {
      quality: "代码质量、命名规范、最佳实践",
      performance: "性能优化、算法复杂度、内存使用",
      security: "安全漏洞、输入验证、敏感信息处理",
      all: "代码质量、性能、安全性、可维护性"
    };
    
    return {
      messages: [
        {
          role: "system",
          content: {
            type: "text",
            text: `你是一位资深的 ${language} 开发专家。

请审查以下代码，重点关注：
${areaMap[focusAreas]}`
          }
        },
        {
          role: "user",
          content: {
            type: "text",
            text: "代码：\n```{language}\n{code}\n```"
          }
        }
      ]
    };
  }
);

server.prompt(
  "explain-code",
  {
    language: z.string().default("python"),
    level: z.enum(["beginner", "intermediate", "advanced"])
      .default("intermediate")
  },
  ({ language, level }) => {
    const levelPrompts = {
      beginner: "用简单易懂的语言，避免技术术语",
      intermediate: "适当使用技术术语，但提供解释",
      advanced: "深入技术细节，讨论底层原理"
    };
    
    return {
      messages: [
        {
          role: "system",
          content: {
            type: "text",
            text: `你是一位 ${language} 教师。${levelPrompts[level]}`
          }
        },
        {
          role: "user",
          content: {
            type: "text",
            text: `请解释以下代码的工作原理：\n\`\`\`${language}\n{code}\n\`\`\``
          }
        }
      ]
    };
  }
);
```

#### 参数注入

Prompt 模板支持**参数注入**，使用 `{parameter_name}` 语法。

**注入方式**：

```python
# 方式 1：函数返回值中的占位符
@mcp.prompt()
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}! 欢迎使用 MCP。"

# 调用时：
# prompts/get -> {name: "World", greeting: "Hi"}
# 返回："Hi, World! 欢迎使用 MCP。"

# 方式 2：保留占位符供后续填充
@mcp.prompt()
def code_template(language: str) -> str:
    return f"""请审查以下 {language} 代码：

```{language}
{{code}}
```"""

# 调用时：
# prompts/get -> {language: "python"}
# 返回："请审查以下 python 代码：\n\n```python\n{code}\n```"
# 然后 Client 可以将 {code} 替换为实际代码
```

**动态参数**：

```python
@mcp.prompt()
def dynamic_context(
    context_type: str,
    **kwargs
) -> str:
    """
    动态上下文提示
    
    Args:
        context_type: 上下文类型
        **kwargs: 额外的参数
    """
    if context_type == "file":
        return f"""当前文件：{kwargs.get('filename', 'unknown')}

内容：
```
{kwargs.get('content', '')}
```"""
    
    elif context_type == "selection":
        return f"""选中的代码：

```{kwargs.get('language', 'text')}
{kwargs.get('selection', '')}
```"""
    
    elif context_type == "cursor":
        return f"""当前光标位置：
文件：{kwargs.get('filename')}
行：{kwargs.get('line')}
列：{kwargs.get('column')}"""
    
    else:
        return f"不支持的上下文类型：{context_type}"
```

#### 动态生成

Prompt 可以**动态生成**，基于参数和上下文。

```python
from datetime import datetime

@mcp.prompt()
def daily_report(
    date: str = None,
    format_type: str = "markdown"
) -> str:
    """
    生成每日报告
    
    Args:
        date: 日期（YYYY-MM-DD）
        format_type: 格式类型
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # 动态查询数据
    stats = get_daily_stats(date)
    
    if format_type == "markdown":
        return f"""# 每日报告 - {date}

- 总提交：{stats['total_commits']}
- 活跃用户：{stats['active_users']}
- 错误数：{stats['error_count']}

{stats['details']}

请分析这些数据，提供：
1. 趋势分析
2. 异常检测
3. 改进建议
"""
    
    elif format_type == "json":
        return f"""以下是 {date} 的统计数据：

{stats.to_json()}

请分析这些数据。"""
    
    else:
        raise ValueError(f"不支持的格式：{format_type}")

@mcp.prompt()
def contextual_help(
    user_role: str,
    experience_level: str,
    topic: str
) -> str:
    """
    上下文感知的帮助
    
    Args:
        user_role: 用户角色
        experience_level: 经验级别
        topic: 主题
    """
    role_contexts = {
        "developer": "从开发者角度，关注实现细节",
        "designer": "从设计师角度，关注用户体验",
        "manager": "从管理者角度，关注项目规划"
    }
    
    level_adjustments = {
        "beginner": "使用简单语言，多举例子",
        "intermediate": "适当深入，关注最佳实践",
        "advanced": "深入技术细节，讨论权衡取舍"
    }
    
    return f"""{role_contexts.get(user_role, '')}

请解释"{topic}"。
{level_adjustments.get(experience_level, '')}

要求：
1. 提供准确的定义
2. 给出实际应用例子
3. 指出常见误区
4. 推荐学习资源"""
```

#### 组合使用

多个 Prompt 可以**组合使用**，形成复杂的工作流。

```python
# 组合示例：代码审查工作流

@mcp.prompt()
def review_step1(language: str) -> str:
    """第一步：识别问题"""
    return f"""作为一名 {language} 专家，请识别以下代码中的所有问题：

1. 语法错误
2. 逻辑错误
3. 性能问题
4. 安全隐患
5. 可读性问题

代码：
```{language}
{code}
```

请列出所有问题，编号说明。"""

@mcp.prompt()
def review_step2(issue_list: str) -> str:
    """第二步：分析问题"""
    return f"""以下是识别出的问题列表：

{issue_list}

请对每个问题：
1. 解释为什么这是问题
2. 说明可能的后果
3. 评估严重程度（高/中/低）"""

@mcp.prompt()
def review_step3(analysis: str) -> str:
    """第三步：提供解决方案"""
    return f"""基于以下分析：

{analysis}

请为每个问题提供具体的修复方案，包括：
1. 修复后的代码示例
2. 修复说明
3. 最佳实践建议"""

# 使用方式
# 1. 调用 review_step1 获取问题列表
# 2. 将结果传入 review_step2 获取分析
# 3. 将分析结果传入 review_step3 获取解决方案
```

---


---


<!-- 章节编号接续第一部分（§1-3），从此处起为 §4-9 -->

## 4. MCP Server 开发（Python）

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

## 6. 更多实战场景

> 本节包含更多 MCP Server 实战项目示例，展示不同场景下的开发实践。

### 6.1 数据库查询 MCP Server

完整的数据库查询 MCP Server，支持 SQL 执行、结果格式化、权限控制。

**核心功能**:
- 只读 SQL 查询（SELECT）
- SQL 安全验证（防止 DROP/DELETE 等危险操作）
- 表结构查看
- 表列表查询
- 结果行数限制（MAX_ROWS = 1000）

**关键代码片段**:

```python
from mcp.server.fastmcp import FastMCP
from databases import Database
import re

mcp = FastMCP("database-server")
database = Database("postgresql://user:password@localhost/mydb")

ALLOWED_OPERATIONS = ["SELECT"]

def validate_sql(sql: str) -> tuple[bool, str]:
    """验证 SQL 语句"""
    operation = sql.strip().split()[0].upper()
    if operation not in ALLOWED_OPERATIONS:
        return False, f"不允许的操作: {operation}"
    
    dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER", "INSERT", "UPDATE"]
    for keyword in dangerous_keywords:
        if re.search(r'\b' + keyword + r'\b', sql, re.IGNORECASE):
            return False, f"SQL 包含危险关键字: {keyword}"
    
    return True, "OK"

@mcp.tool()
async def execute_query(sql: str, params: dict = None) -> str:
    """执行只读 SQL 查询"""
    valid, message = validate_sql(sql)
    if not valid:
        return f"SQL 验证失败: {message}"
    
    async with database:
        results = await database.fetch_all(sql, values=params or {})
    
    return f"查询成功，返回 {len(results)} 行\n\n{results}"
```

### 6.2 文件管理 MCP Server

文件管理 MCP Server，支持文件读写、目录操作、搜索。

**核心功能**:
- 安全路径解析（防止路径遍历攻击）
- 文件读取/写入
- 目录列表
- 文件搜索（支持通配符）
- 基于 BASE_DIR 的访问控制

**关键代码片段**:

```python
from mcp.server.fastmcp import FastMCP
from pathlib import Path

mcp = FastMCP("file-manager")
BASE_DIR = Path("/app/data")

def safe_path(file_path: str) -> Path:
    """安全的路径解析"""
    path = Path(file_path).resolve()
    if not path.startswith(BASE_DIR):
        raise ValueError(f"路径超出允许范围: {BASE_DIR}")
    return path

@mcp.tool()
async def read_file(file_path: str) -> str:
    """读取文件内容"""
    path = safe_path(file_path)
    if not path.exists():
        return f"文件不存在: {file_path}"
    return path.read_text(encoding='utf-8')
```

### 6.3 API 集成 MCP Server

API 集成 MCP Server，封装 REST API、GraphQL 支持、认证管理。

**核心功能**:
- REST API 调用（GET/POST）
- GraphQL 查询
- Bearer Token 认证
- HTTP 客户端连接池
- 优雅关闭（on_shutdown）

**关键代码片段**:

```python
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("api-integration")

client = httpx.AsyncClient(
    base_url="https://api.example.com",
    timeout=30.0,
    headers={"Content-Type": "application/json"}
)

API_KEY = "your-api-key"

@mcp.tool()
async def get_users(page: int = 1, limit: int = 10) -> str:
    """获取用户列表"""
    response = await client.get(
        "/users",
        params={"page": page, "limit": limit},
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    response.raise_for_status()
    return str(response.json())

@mcp.on_shutdown()
async def cleanup():
    """关闭 HTTP 客户端"""
    await client.aclose()
```

> **完整代码**: 参考 `03-MCP安全权限与生产部署.md` 的第 1 节 "实战项目"（已迁移至此）



---


<!-- 章节编号接续第二部分（§4-9），从此处起为 §10-13 -->

## 10. 生态工具

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

## 2. 前沿技术

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

## 4. 参考资料

### 4.1 官方文档

- **MCP 官方文档**: https://modelcontextprotocol.io
- **MCP 规范**: https://github.com/modelcontextprotocol/specification
- **Python SDK**: https://github.com/modelcontextprotocol/python-sdk
- **TypeScript SDK**: https://github.com/modelcontextprotocol/typescript-sdk

### 4.2 学习资源

- **Microsoft MCP for Beginners**: https://github.com/microsoft/mcp-for-beginners
- **MCP 中文教程**: https://github.com/chenmingyong0423/mcp-tutorials
- **MCP 资源集合**: https://github.com/cyanheads/model-context-protocol-resources

### 4.3 官方 Servers

- **GitHub Server**: https://github.com/modelcontextprotocol/servers/tree/main/src/github
- **Filesystem Server**: https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem
- **PostgreSQL Server**: https://github.com/modelcontextprotocol/servers/tree/main/src/postgres
- **Playwright Server**: https://github.com/modelcontextprotocol/servers/tree/main/src/playwright

### 4.4 社区资源

- **MCP Discord**: https://discord.gg/mcp
- **MCP GitHub Discussions**: https://github.com/modelcontextprotocol/specification/discussions
- **Awesome MCP**: https://github.com/punkpeye/awesome-mcp-servers

### 4.5 相关技术

- **JSON-RPC 2.0**: https://www.jsonrpc.org/specification
- **Server-Sent Events**: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- **OpenAPI Specification**: https://swagger.io/specification/

---

## 3. MCP 2026-07-28 安全与生产部署更新

### 3.1 授权强化（Authorization Hardening）

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

### 3.2 可观测性（Observability）

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

### 3.3 生产部署最佳实践（2026-07-28）

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

### 3.4 弃用特性迁移指南

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