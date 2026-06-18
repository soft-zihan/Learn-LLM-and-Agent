# MCP 协议核心概念

> 📅 **更新时间**: 2026-06-17  

---

## 目录

- [1. MCP 协议基础概念](#1-mcp-协议基础概念)
- [2. MCP 2026-07-28 规范重大更新](#2-mcp-2026-07-28-规范重大更新)
- [3. MCP 协议详解](#3-mcp-协议详解)

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
              │  (Stdio / HTTP/SSE)   │
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

**2. HTTP/SSE 传输**

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

### 2.1 初始化流程

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

### 2.2 资源管理

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

### 2.3 工具调用

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

### 2.4 提示系统

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
