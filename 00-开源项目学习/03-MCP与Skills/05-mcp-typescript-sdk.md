# MCP TypeScript SDK - MCP 协议学习指南

## 项目概述

**一句话总结**：MCP 协议的官方 TypeScript 实现，支持 Node.js、Bun、Deno，让你用 JavaScript/TypeScript 构建 AI 工具服务。

**核心亮点**：
- **Split Packages 设计**：`@modelcontextprotocol/server` 和 `@modelcontextprotocol/client` 分离，按需引入
- **Standard Schema 支持**：Zod v4、Valibot、ArkType 等验证库都能用
- **中间件架构**：Express、Hono、Node.js HTTP 集成开箱即用
- **跨运行时**：Node.js、Bun、Deno 全面支持，一套代码多处运行

**适合谁学**：
- 前端/全栈开发者想给 AI 应用添加工具能力
- 需要理解 MCP 协议 TypeScript 实现原理
- 准备在 Web 环境部署 MCP 服务

## 核心架构解析

### MCP 协议整体架构

MCP 就像**应用商店系统**：
- **Server（服务端）** = 应用开发者，发布工具（App）
- **Client（客户端）** = 用户，安装和使用工具
- **Host（宿主）** = 应用商店平台，管理连接和权限

```
Client (用户) ←→ Host (平台) ←→ Server (开发者)
   使用工具         分发管理        提供工具
```

### Client-Host-Server 三角模型

**TypeScript 实现特点**：
```typescript
// 客户端包：@modelcontextprotocol/client
// 源码位置：packages/client/src/client/client.ts
// 负责：连接服务端、调用工具、读取资源

// 服务端包：@modelcontextprotocol/server
// 源码位置：packages/server/src/server/mcp.ts
// 负责：注册工具、处理请求、返回结果
```

**工作流程**（以天气查询为例）：
1. AI 助手（通过 Client）发现需要查天气
2. Client 向 Server 发送 `tools/call` 请求
3. Server 执行天气查询逻辑
4. Server 返回结构化结果
5. Client 将结果交给 AI 助手

### 三大原语（Tools、Resources、Prompts）

#### 1. **Tools（工具）** - AI 的"超能力"

让 AI 能执行实际操作：

```typescript
// 源码位置：packages/server/src/server/mcp.ts
import { McpServer } from '@modelcontextprotocol/server';
import { StdioServerTransport } from '@modelcontextprotocol/server/stdio';
import * as z from 'zod/v4';

const server = new McpServer({ name: 'weather-server', version: '1.0.0' });

server.registerTool(
    'get_weather',
    {
        description: '查询城市天气',
        inputSchema: z.object({ city: z.string() })
    },
    async ({ city }) => ({
        content: [{ type: 'text', text: `${city}今天晴天，25°C` }]
    })
);
```

#### 2. **Resources（资源）** - AI 的"知识库"

让 AI 能读取数据：

```typescript
// 资源就像 AI 可以访问的 URL
// 源码位置：packages/server/src/server/mcp.ts
server.registerResource(
    'config',
    'file:///config.json',
    { description: '应用配置文件' },
    async (uri) => ({
        contents: [{ uri: uri.href, text: '{ "theme": "dark" }' }]
    })
);
```

#### 3. **Prompts（提示词）** - AI 的"工作流模板"

让 AI 有标准化的操作流程：

```typescript
// 提示词就像 AI 的"快捷指令"
// 源码位置：packages/server/src/server/mcp.ts
server.registerPrompt(
    'code-review',
    {
        description: '代码审查模板',
        arguments: [
            { name: 'code', description: '要审查的代码', required: true }
        ]
    },
    async (args) => ({
        messages: [{
            role: 'user',
            content: { type: 'text', text: `请审查以下代码：\n${args.code}` }
        }]
    })
);
```

## 代码逻辑主线

### 核心代码流程

```
1. 创建 McpServer 实例
   ↓
2. 注册工具/资源/提示词（registerTool/registerResource/registerPrompt）
   ↓
3. 创建传输层（StdioServerTransport / StreamableHTTP）
   ↓
4. 调用 server.connect(transport)
   ↓
5. 等待请求 → 执行处理 → 返回响应
```

### 关键类与方法

| 类名 | 文件位置 | 作用 | 重要性 |
|------|---------|------|--------|
| `McpServer` | `packages/server/src/server/mcp.ts` | 高层服务器 API | ⭐⭐⭐⭐⭐ |
| `Server` | `packages/server/src/server/server.ts` | 底层服务器实现 | ⭐⭐⭐⭐ |
| `Client` | `packages/client/src/client/client.ts` | MCP 客户端 | ⭐⭐⭐⭐⭐ |
| `StdioServerTransport` | `packages/server/src/server/stdio.ts` | stdio 传输 | ⭐⭐⭐ |
| `StreamableHTTPServerTransport` | `packages/server/src/server/streamableHttp.ts` | HTTP 传输 | ⭐⭐⭐⭐⭐ |
| `StdioClientTransport` | `packages/client/src/client/stdio.ts` | 客户端 stdio 传输 | ⭐⭐⭐ |

### 通信协议实现

**JSON-RPC 2.0 消息格式**：

```typescript
// 请求消息（Client → Server）
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": { "city": "上海" }
  }
}

// 响应消息（Server → Client）
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [{ "type": "text", "text": "上海今天晴天，25°C" }]
  }
}
```

**关键源码**：
- 消息路由：`packages/server/src/server/server.ts` 的请求处理逻辑
- 传输层抽象：`packages/core` 中的 `Transport` 接口
- 客户端认证：`packages/client/src/client/auth.ts`（OAuth 2.0 实现）

### 中间件架构

TypeScript SDK 的独特设计：

```typescript
// 中间件包：packages/middleware/
// @modelcontextprotocol/express - Express 集成
// @modelcontextprotocol/hono - Hono 集成
// @modelcontextprotocol/node - Node.js HTTP 封装

// 使用示例（Express）
import { expressMiddleware } from '@modelcontextprotocol/express';
app.use('/mcp', expressMiddleware(server));
```

## 快速上手实践

### 环境配置步骤

```bash
# 1. 初始化项目
npm init -y
npm install typescript @types/node --save-dev

# 2. 安装 MCP SDK
npm install @modelcontextprotocol/server
npm install @modelcontextprotocol/client  # 如果需要客户端

# 3. 配置 TypeScript（tsconfig.json）
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "esModuleInterop": true,
    "strict": true
  }
}
```

### 运行第一个示例

创建文件 `hello-mcp.ts`：

```typescript
import { McpServer } from '@modelcontextprotocol/server';
import { StdioServerTransport } from '@modelcontextprotocol/server/stdio';
import * as z from 'zod/v4';

// 1. 创建服务器
const server = new McpServer({
    name: 'hello-server',
    version: '1.0.0'
});

// 2. 注册工具
server.registerTool(
    'greet',
    {
        description: '向某人打招呼',
        inputSchema: z.object({ name: z.string() })
    },
    async ({ name }) => ({
        content: [{ type: 'text', text: `你好，${name}！` }]
    })
);

// 3. 启动服务
async function main() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error('MCP 服务已启动');
}

main().catch(console.error);
```

编译并运行：
```bash
npx tsx hello-mcp.ts
```

### 预期输出与验证方法

服务会在后台运行。验证方法：

**方法一**：Claude Desktop 配置
```json
{
  "mcpServers": {
    "hello": {
      "command": "npx",
      "args": ["tsx", "/path/to/hello-mcp.ts"]
    }
  }
}
```

**方法二**：编写客户端测试
```typescript
import { Client } from '@modelcontextprotocol/client';
import { StdioClientTransport } from '@modelcontextprotocol/client/stdio';

async function test() {
    const transport = new StdioClientTransport({
        command: 'npx',
        args: ['tsx', 'hello-mcp.ts']
    });
    
    const client = new Client(transport);
    await client.connect();
    
    const tools = await client.listTools();
    console.log('可用工具:', tools.tools.map(t => t.name));
    
    const result = await client.callTool({
        name: 'greet',
        arguments: { name: '张三' }
    });
    
    console.log('调用结果:', result);
}

test().catch(console.error);
```

**预期输出**：
```
可用工具: ['greet']
调用结果: { content: [{ type: 'text', text: '你好，张三！' }] }
```

## 核心知识点总结

### 必须掌握的 8 个概念

1. **McpServer vs Server**
   - `McpServer` 是高层 API，使用 `registerTool()` 等方法
   - `Server` 是底层实现，手动设置请求处理器
   - **为什么重要**：选对 API 能提高开发效率

2. **Standard Schema**
   - 统一的 Schema 规范，支持 Zod、Valibot 等多种验证库
   - 使用 `zod/v4` 而非 `zod`（v4 版本）
   - **为什么重要**：TypeScript SDK 的核心设计，类型安全的基础

3. **Transport（传输层）**
   - stdio：进程间通信，适合桌面应用
   - Streamable HTTP：生产环境首选，支持现代 Web
   - **为什么重要**：决定服务的部署方式和网络能力

4. **Split Packages（分包设计）**
   - `@modelcontextprotocol/server`：服务端库
   - `@modelcontextprotocol/client`：客户端库
   - **为什么重要**：减小包体积，按需引入

5. **Middleware（中间件）**
   - Express、Hono、Node.js HTTP 的薄封装
   - 不引入新功能，只是适配不同框架
   - **为什么重要**：简化 Web 框架集成

6. **ToolAnnotations（工具注解）**
   - `readOnlyHint`、`destructiveHint` 等元数据
   - 帮助 AI 理解工具的行为特征
   - **为什么重要**：提升 AI 的工具选择准确性

7. **Structured Output（结构化输出）**
   - 工具可以返回 `structuredContent` 字段
   - 除了文本，还能返回 JSON 对象
   - **为什么重要**：支持程序化消费工具结果

8. **OAuth Authentication（OAuth 认证）**
   - 客户端和服务端的 OAuth 2.0 支持
   - 源码：`packages/client/src/client/auth.ts`
   - **为什么重要**：生产环境安全必备

## 常见疑问解答

### Q1: McpServer 和 Server 有什么区别？

**答**：
- **McpServer**：高层 API，一行代码注册工具，自动处理 Schema
- **Server**：底层 API，手动设置 `setRequestHandler()`，完全控制

**建议**：优先用 `McpServer`，只有在需要自定义协议行为（如发送通知、特殊错误处理）时才用 `Server`。

### Q2: 为什么用 `zod/v4` 而不是 `zod`？

**答**：
- MCP TypeScript SDK 使用 **Standard Schema** 规范
- `zod/v4` 实现了这个规范，Valibot、ArkType 也可以
- 这是为了让你能自由选择验证库，不锁定在 Zod

```typescript
// 使用 Zod v4
import * as z from 'zod/v4';

// 也可以用 Valibot（如果支持 Standard Schema）
import * as v from 'valibot';
```

### Q3: 如何在浏览器中使用 MCP 客户端？

**答**：
- 浏览器不能用 stdio（没有子进程），必须用 HTTP/SSE
- 使用 `StreamableHTTPClientTransport` 或 `SSEClientTransport`
- 源码：`packages/client/src/client/streamableHttp.ts`

```typescript
import { Client } from '@modelcontextprotocol/client';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/client/streamableHttp';

const transport = new StreamableHTTPClientTransport('http://localhost:3000/mcp');
const client = new Client(transport);
await client.connect();
```

### Q4: v1 和 v2 有什么区别？

**答**：
- 当前主分支是 **v2（开发中，pre-alpha）**
- **v1** 是稳定版，生产环境推荐使用
- v2 预计 2026 Q3 发布，配合更新的 MCP 规范

**建议**：
- 学习/实验：可以用 v2 了解最新特性
- 生产环境：使用 v1（`@modelcontextprotocol/server@latest`）

### Q5: 如何调试 MCP 服务？

**答**：
1. **日志输出**：使用 `console.error()`（不要用 `console.log()`，会干扰协议通信）
2. **客户端调试**：在客户端打印请求和响应
3. **查看源码**：
   - 服务端消息处理：`packages/server/src/server/server.ts`
   - 客户端请求发送：`packages/client/src/client/client.ts`
4. **使用示例代码**：`examples/` 目录有完整的 runnable examples

### Q6: TypeScript SDK 和 Python SDK 选哪个？

**答**：
- **选 TypeScript**：前端项目、Web 服务、Node.js 生态
- **选 Python**：数据科学、机器学习、后端服务
- **两者都支持**：MCP 协议是语言无关的，可以混合使用

**类比**：就像 REST API，可以用任何语言实现，只要能互相通信。
