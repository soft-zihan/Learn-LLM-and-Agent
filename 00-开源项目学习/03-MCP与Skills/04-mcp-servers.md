# MCP Servers - MCP 协议学习指南

## 项目概述

**一句话总结**：MCP 官方参考实现集合，包含 7 个即用型服务器，展示 MCP 协议的各种应用场景。

**核心亮点**：
- **7 个参考服务器**：Everything、Fetch、Filesystem、Git、Memory、Sequential Thinking、Time
- **教学价值**：每个服务器都是最佳实践示例，代码清晰易读
- **即插即用**：通过 `npx` 或 `uvx` 直接运行，零配置
- **多语言实现**：TypeScript 和 Python 混合展示不同 SDK 用法

**适合谁学**：
- 想快速了解 MCP 能做什么的初学者
- 需要参考实现来构建自己的 MCP 服务器
- 想测试 MCP 客户端功能的开发者

## 核心架构解析

### MCP 协议整体架构

把这个仓库想象成**APP 应用商店的示例应用**：
- 每个服务器 = 一个独立的 APP（提供特定功能）
- MCP 协议 = 操作系统接口（标准化通信）
- 客户端（如 Claude Desktop）= 手机系统（运行 APP）

```
Claude Desktop (手机系统)
    ↓ 安装和运行
┌─────────────────────────────────┐
│ Memory Server (备忘录 APP)       │
│ Filesystem Server (文件管理器)   │
│ Git Server (版本控制 APP)        │
│ ... 更多服务器                   │
└─────────────────────────────────┘
    ↑ 都通过 MCP 协议通信
```

### Client-Host-Server 三角模型

**在这个仓库中的体现**：

| 服务器 | 类型 | 语言 | 运行方式 |
|--------|------|------|----------|
| Everything | 全功能演示 | TypeScript | `npx` |
| Memory | 知识图谱 | TypeScript | `npx` |
| Filesystem | 文件操作 | TypeScript | `npx` |
| Fetch | 网页抓取 | TypeScript | `npx` |
| Git | Git 操作 | Python | `uvx` |
| Time | 时间查询 | Python | `uvx` |
| Sequential Thinking | 思维链 | TypeScript | `npx` |

### 三大原语实战展示

#### 1. **Tools（工具）** - 每个服务器都在用

**示例：Memory 服务器的工具**
```typescript
// 源码位置：src/memory/index.ts
// 创建实体工具
server.registerTool(
    "create_entities",
    {
        title: "Create Entities",
        description: "在知识图谱中创建新实体",
        inputSchema: {
            entities: z.array(EntitySchema)
        },
        outputSchema: {
            entities: z.array(EntitySchema)
        },
        annotations: {
            readOnlyHint: false,      // 会修改数据
            destructiveHint: false,   // 非破坏性
            idempotentHint: false,    // 非幂等
            openWorldHint: false      // 封闭世界
        }
    },
    async ({ entities }) => {
        const result = await knowledgeGraphManager.createEntities(entities);
        return {
            content: [{ type: "text", text: JSON.stringify(result) }],
            structuredContent: { entities: result }
        };
    }
);
```

**ToolAnnotations 详解**（帮助 AI 理解工具行为）：
- `readOnlyHint`: 是否只读（不修改数据）
- `destructiveHint`: 是否破坏性操作（删除、覆盖）
- `idempotentHint`: 是否幂等（多次执行结果相同）
- `openWorldHint`: 是否开放世界（影响推理逻辑）

#### 2. **Resources（资源）** - 提供数据访问

**示例：Everything 服务器的资源**
```typescript
// 源码位置：src/everything/
// 静态资源
server.registerResource(
    "static-resource",
    "file:///static/example.json",
    { description: "示例资源" },
    async (uri) => ({
        contents: [{
            uri: uri.href,
            mimeType: "application/json",
            text: JSON.stringify({ key: "value" })
        }]
    })
);

// 资源模板（动态资源）
server.registerResourceTemplate(
    "file:///{path}",
    { description: "读取文件内容" },
    async (uri, params) => {
        const content = await readFile(params.path);
        return { contents: [{ uri: uri.href, text: content }] };
    }
);
```

#### 3. **Prompts（提示词）** - 标准化工作流

**示例：Everything 服务器的提示词**
```typescript
// 源码位置：src/everything/prompts/
server.registerPrompt(
    "code-review",
    {
        description: "代码审查提示词模板",
        arguments: [
            { 
                name: "code", 
                description: "要审查的代码", 
                required: true 
            },
            {
                name: "language",
                description: "编程语言",
                required: false
            }
        ]
    },
    async (args) => ({
        messages: [{
            role: "user",
            content: {
                type: "text",
                text: `请作为资深开发者，审查以下 ${args.language || '代码'}：\n\n${args.code}`
            }
        }]
    })
);
```

## 代码逻辑主线

### Everything 服务器（全功能演示）

**为什么重要**：这是最完整的参考实现，展示了 MCP 的所有特性。

**核心代码流程**：
```
1. 解析命令行参数（选择传输方式）
   ↓ src/everything/index.ts
2. 加载对应传输模块（stdio / sse / streamableHttp）
   ↓ src/everything/transports/
3. 创建 McpServer 实例
   ↓
4. 注册所有工具、资源、提示词
   ↓
5. 连接传输层，启动服务
```

**关键文件**：
- `src/everything/index.ts`：入口文件，选择传输方式
- `src/everything/transports/stdio.ts`：stdio 传输实现
- `src/everything/transports/sse.ts`：SSE 传输实现
- `src/everything/transports/streamableHttp.ts`：HTTP 传输实现

### Memory 服务器（知识图谱）

**为什么重要**：展示了如何实现持久化存储和复杂数据管理。

**核心代码流程**：
```
1. 初始化知识图谱管理器（KnowledgeGraphManager）
   ↓ src/memory/index.ts
2. 加载 JSONL 文件（或创建空图谱）
   ↓
3. 注册 9 个工具：
   - create_entities（创建实体）
   - create_relations（创建关系）
   - add_observations（添加观察）
   - delete_entities（删除实体）
   - delete_observations（删除观察）
   - delete_relations（删除关系）
   - read_graph（读取整个图谱）
   - search_nodes（搜索节点）
   - open_nodes（打开指定节点）
   ↓
4. 启动 stdio 服务
```

**知识图谱数据结构**：
```typescript
// 实体（Entity）：节点
interface Entity {
    name: string;           // 实体名称
    entityType: string;     // 实体类型
    observations: string[]; // 观察内容
}

// 关系（Relation）：边
interface Relation {
    from: string;      // 起始实体
    to: string;        // 目标实体
    relationType: string; // 关系类型
}
```

**存储方式**：JSONL 格式（每行一个 JSON 对象）
```jsonl
{"type": "entity", "name": "张三", "entityType": "人", "observations": ["开发者"]}
{"type": "relation", "from": "张三", "to": "李四", "relationType": "同事"}
```

### Time 服务器（时间查询）

**为什么重要**：展示了 Python SDK 的简洁用法。

**核心代码**（Python 实现）：
```python
# 源码位置：src/time/（Python 包）
from mcp.server.fastmcp import FastMCP
from datetime import datetime
import pytz

mcp = FastMCP("time-server")

@mcp.tool()
def get_current_time(timezone: str = "UTC") -> str:
    """获取当前时间"""
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")

@mcp.tool()
def convert_time(time: str, from_tz: str, to_tz: str) -> str:
    """转换时区"""
    # 时区转换逻辑
    ...
```

## 快速上手实践

### 环境配置步骤

**TypeScript 服务器**（需要 Node.js）：
```bash
# 1. 安装 Node.js（推荐 v18+）
# 2. 直接使用 npx 运行，无需安装

# 验证安装
node --version  # 应显示 v18 或更高
```

**Python 服务器**（需要 Python 3.10+）：
```bash
# 1. 安装 uv（推荐的 Python 包管理器）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 或者使用 pip
pip install --upgrade pip
```

### 运行第一个示例

**示例 1：Time 服务器（最简单）**
```bash
# 使用 uvx 运行（推荐）
uvx mcp-server-time

# 或使用 pip
pip install mcp-server-time
python -m mcp_server_time
```

**示例 2：Memory 服务器（知识图谱）**
```bash
# 使用 npx 运行
npx -y @modelcontextprotocol/server-memory
```

**示例 3：Everything 服务器（全功能）**
```bash
# stdio 传输（默认）
npx -y @modelcontextprotocol/server-everything

# SSE 传输
npx -y @modelcontextprotocol/server-everything sse

# HTTP 传输
npx -y @modelcontextprotocol/server-everything streamableHttp
```

### 配置到 Claude Desktop

创建或编辑 Claude Desktop 配置文件：
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "time": {
      "command": "uvx",
      "args": ["mcp-server-time"]
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"]
    },
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git", "--repository", "/path/to/git/repo"]
    }
  }
}
```

### 预期输出与验证方法

**验证方法 1**：在 Claude Desktop 中测试
1. 重启 Claude Desktop
2. 在对话中输入："现在几点了？"（会调用 time 服务器）
3. Claude 应该能正确回答当前时间

**验证方法 2**：手动测试 stdio 通信
```bash
# 启动服务器
npx -y @modelcontextprotocol/server-memory

# 在另一个终端发送 JSON-RPC 请求
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | npx -y @modelcontextprotocol/server-memory
```

**预期响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {},
      "resources": {},
      "prompts": {}
    },
    "serverInfo": {
      "name": "memory-server",
      "version": "0.6.3"
    }
  }
}
```

## 核心知识点总结

### 必须掌握的 8 个概念

1. **参考实现 vs 生产代码**
   - 这些服务器是**教学示例**，不是生产就绪方案
   - 展示了 MCP 的最佳实践和 API 用法
   - **为什么重要**：避免直接用于生产环境，需要自己评估安全性

2. **npx 和 uvx 的用法**
   - `npx`：运行 TypeScript/Node.js 包（无需全局安装）
   - `uvx`：运行 Python 包（类似 npx 的 Python 版本）
   - **为什么重要**：零配置运行 MCP 服务器的关键

3. **Claude Desktop 配置**
   - JSON 格式配置，定义服务器命令和参数
   - `env` 字段传递环境变量（如 API Key）
   - **为什么重要**：实际使用 MCP 的主要方式

4. **ToolAnnotations（工具注解）**
   - 四个 Hint 帮助 AI 理解工具行为
   - 在 `src/memory/index.ts` 中有完整示例
   - **为什么重要**：提升 AI 工具调用的准确性

5. **JSONL 存储格式**
   - Memory 服务器使用 JSONL（每行一个 JSON）
   - 比 JSON 数组更适合增量写入
   - **为什么重要**：学习轻量级持久化方案

6. **传输方式选择**
   - stdio：本地进程通信（桌面应用）
   - SSE：服务端推送（实时数据）
   - HTTP：生产环境（跨网络）
   - **为什么重要**：决定部署架构

7. **Everything 服务器的价值**
   - 包含 MCP 所有特性的完整示例
   - 源码：`src/everything/`
   - **为什么重要**：学习 MCP 全貌的最佳起点

8. **多语言混合**
   - TypeScript 服务器用 `npx` 运行
   - Python 服务器用 `uvx` 运行
   - **为什么重要**：MCP 协议是语言无关的，可以混合使用

## 常见疑问解答

### Q1: 这些服务器能直接用于生产环境吗？

**答**：**不能**。这些是参考实现，目的是教学：
- 缺少生产级错误处理
- 缺少完整的访问控制
- 缺少监控和日志
- 缺少性能优化

**正确做法**：学习代码结构和 API 用法，然后根据自己需求重新实现。

### Q2: npx 和 npm install 有什么区别？

**答**：
- `npx`：临时运行包，不全局安装（类似"试用"）
- `npm install`：永久安装到项目（类似"购买"）

**MCP 服务器推荐用 npx**：
```bash
# 推荐：临时运行，用完即走
npx -y @modelcontextprotocol/server-memory

# 不推荐：全局安装（会污染全局环境）
npm install -g @modelcontextprotocol/server-memory
```

### Q3: 如何查看服务器实际发送和接收的消息？

**答**：
1. **stdio 模式**：消息通过标准输入输出，可以重定向到文件
```bash
# 记录通信内容
npx -y @modelcontextprotocol/server-memory > messages.log 2>&1
```

2. **HTTP 模式**：用浏览器开发者工具或 Charles/Postman
```bash
# 启动 HTTP 服务器
npx -y @modelcontextprotocol/server-everything streamableHttp

# 在浏览器访问 http://localhost:3000/mcp
```

3. **源码调试**：在 `registerTool` 的回调函数中添加 `console.error()`

### Q4: Everything 服务器的作用是什么？

**答**：Everything 是**MCP 功能的完整演示**：
- 展示了所有三大原语（Tools、Resources、Prompts）
- 包含所有传输方式（stdio、SSE、HTTP）
- 有工具补全、分页、日志等高级功能
- **类似"功能展示厅"**，让你快速了解 MCP 能做什么

**学习建议**：
1. 先运行 Everything 服务器体验所有功能
2. 阅读 `src/everything/` 源码理解实现
3. 参考代码结构构建自己的服务器

### Q5: 如何基于这些参考实现创建自己的服务器？

**答**：以 Memory 服务器为模板：

**步骤 1**：复制并修改
```bash
# 复制 memory 服务器代码
cp -r src/memory my-custom-server
cd my-custom-server
```

**步骤 2**：修改核心逻辑
```typescript
// 1. 修改服务器名称
const server = new McpServer({
    name: "my-custom-server",  // 改这里
    version: "1.0.0"
});

// 2. 修改工具注册
server.registerTool(
    "my-tool",  // 改工具名
    {
        description: "我的工具描述",
        inputSchema: z.object({ /* 定义参数 */ })
    },
    async (args) => {
        // 实现你的业务逻辑
        return {
            content: [{ type: "text", text: "结果" }]
        };
    }
);
```

**步骤 3**：发布和运行
```bash
# 发布到 npm（可选）
npm publish

# 或直接运行
npx tsx index.ts
```

### Q6: 为什么有些服务器用 TypeScript，有些用 Python？

**答**：
- **TypeScript 服务器**：展示 TypeScript SDK 用法，适合前端/全栈开发者
- **Python 服务器**：展示 Python SDK 用法，适合后端/数据科学家
- **MCP 协议是语言无关的**，客户端可以和任何语言的服务器通信

**学习建议**：
- 选你熟悉的语言学习
- 理解协议原理后，可以用任何语言实现
- 实际项目中，可以混合使用多种语言
