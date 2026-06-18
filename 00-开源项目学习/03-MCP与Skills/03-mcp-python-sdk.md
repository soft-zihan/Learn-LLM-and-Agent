# MCP Python SDK - MCP 协议学习指南

## 项目概述

**一句话总结**：这是 MCP（Model Context Protocol）协议的官方 Python 实现，让你能用 Python 快速构建 AI 工具服务和客户端。

**核心亮点**：
- **双层级 API**：高层 `MCPServer` 像搭积木一样简单，底层 `Server` 给你完全控制权
- **多种通信方式**：支持 stdio（命令行）、SSE（服务器推送）、Streamable HTTP（现代 Web）
- **内置认证系统**：OAuth 2.0 开箱即用，保护你的 AI 服务
- **类型安全**：基于 Pydantic，参数校验和自动补全一步到位

**适合谁学**：
- 想用 Python 给 AI 助手添加工具能力的开发者
- 需要理解 MCP 协议底层机制的技术人员
- 准备构建生产级 AI 服务的工程师

## 核心架构解析

### MCP 协议整体架构

把 MCP 想象成**餐厅系统**：
- **Client（客户端）** = 顾客，点菜（请求工具、资源、提示词）
- **Server（服务端）** = 厨房，做菜（提供工具、资源、提示词）
- **Host（宿主）** = 服务员，传递菜单和菜品（管理连接、转发消息）

```
Client (顾客) ←→ Host (服务员) ←→ Server (厨房)
     点菜              传递            做菜
```

### Client-Host-Server 三角模型

```python
# 源码位置：src/mcp/client/client.py
# 客户端负责发起请求，接收结果

# 源码位置：src/mcp/server/mcpserver/server.py
# 服务端负责注册工具、资源、提示词，处理请求
```

**工作流程比喻**：
1. 顾客（Client）看菜单，点"查天气"这道菜
2. 服务员（Host）把订单送到厨房
3. 厨房（Server）做好菜，返回结果
4. 服务员把菜端给顾客

### 三大原语（Tools、Resources、Prompts）

MCP 协议只有三种核心能力，就像手机的三个基础功能：

#### 1. **Tools（工具）** - 让 AI 能"动手"
就像给 AI 装了一双手，可以执行操作：
- 查询数据库
- 调用 API
- 处理文件

```python
# 源码位置：src/mcp/server/mcpserver/server.py
# 注册工具示例（简化版）
@server.tool()
def search_weather(city: str) -> str:
    """查询天气"""
    return f"{city}今天晴天"
```

#### 2. **Resources（资源）** - 让 AI 能"阅读"
就像给 AI 装了一双眼睛，可以读取数据：
- 文件内容
- 配置信息
- 文档资料

```python
# 资源就像 AI 可以访问的"书架"
# 源码位置：src/mcp/server/mcpserver/resources.py
```

#### 3. **Prompts（提示词）** - 让 AI 有"模板"
就像给 AI 准备了标准化话术：
- 代码审查模板
- 数据分析流程
- 报告生成指南

```python
# 提示词就像 AI 的"快捷回复"
# 源码位置：src/mcp/server/mcpserver/prompts.py
```

## 代码逻辑主线

### 核心代码流程

```
1. 创建服务器
   ↓
2. 注册工具/资源/提示词（装饰器模式）
   ↓
3. 选择传输方式（stdio / SSE / HTTP）
   ↓
4. 启动服务，等待请求
   ↓
5. 接收请求 → 执行逻辑 → 返回结果
```

### 关键类与方法

| 类名 | 文件位置 | 作用 | 重要性 |
|------|---------|------|--------|
| `MCPServer` | `src/mcp/server/mcpserver/server.py` | 高层服务器，简化 API | ⭐⭐⭐⭐⭐ |
| `Server` | `src/mcp/server/lowlevel/server.py` | 底层服务器，完全控制 | ⭐⭐⭐⭐ |
| `Client` | `src/mcp/client/client.py` | MCP 客户端 | ⭐⭐⭐⭐ |
| `Session` | `src/mcp/shared/session.py` | 会话管理，处理消息 | ⭐⭐⭐⭐⭐ |
| `StdioServerTransport` | `src/mcp/server/stdio.py` | 标准输入输出传输 | ⭐⭐⭐ |
| `StreamableHTTP` | `src/mcp/server/streamable_http.py` | HTTP 传输（推荐） | ⭐⭐⭐⭐⭐ |

### 通信协议实现

MCP 使用 **JSON-RPC 2.0** 协议通信，可以理解为"标准化的对话格式"：

```json
// 请求示例（客户端发给服务端）
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search_weather",
    "arguments": {"city": "北京"}
  }
}

// 响应示例（服务端返回客户端）
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [{"type": "text", "text": "北京今天晴天"}]
  }
}
```

**关键源码**：
- JSON-RPC 消息处理：`src/mcp/shared/jsonrpc_dispatcher.py`
- 请求响应逻辑：`src/mcp/shared/session.py` 的 `RequestResponder` 类
- 错误处理：`src/mcp/shared/exceptions.py`

## 快速上手实践

### 环境配置步骤

```bash
# 1. 安装依赖（推荐使用 uv）
uv pip install mcp

# 或者用 pip
pip install mcp
```

### 运行第一个示例

创建文件 `hello_mcp.py`：

```python
from mcp.server.mcpserver import MCPServer
from mcp.server.stdio import stdio_server
import asyncio

# 1. 创建服务器
server = MCPServer(name="hello-server", version="1.0.0")

# 2. 注册工具（使用装饰器）
@server.tool()
def greet(name: str) -> str:
    """向某人打招呼"""
    return f"你好，{name}！"

# 3. 启动服务
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

运行：
```bash
python hello_mcp.py
```

### 预期输出与验证方法

服务会在后台运行，通过标准输入输出通信。验证方法：

1. **方法一**：使用 Claude Desktop 配置
```json
{
  "mcpServers": {
    "hello": {
      "command": "python",
      "args": ["/path/to/hello_mcp.py"]
    }
  }
}
```

2. **方法二**：编写客户端测试
```python
from mcp.client.client import Client
from mcp.client.stdio import StdioClientTransport

async def test():
    async with Client(StdioClientTransport("python", "hello_mcp.py")) as client:
        await client.initialize()
        tools = await client.list_tools()
        print(f"可用工具: {[t.name for t in tools]}")
        
        result = await client.call_tool("greet", {"name": "张三"})
        print(f"调用结果: {result}")

import asyncio
asyncio.run(test())
```

**预期输出**：
```
可用工具: ['greet']
调用结果: 你好，张三！
```

## 核心知识点总结

### 必须掌握的 8 个概念

1. **MCPServer vs Server**
   - `MCPServer` 是高层封装，使用装饰器，适合快速开发
   - `Server` 是底层接口，手动注册处理器，适合精细控制
   - **为什么重要**：选对 API 层级能事半功倍

2. **Transport（传输层）**
   - stdio：适合本地进程间通信（如 Claude Desktop）
   - SSE：适合服务端推送场景
   - Streamable HTTP：现代 Web 标准，支持双向通信
   - **为什么重要**：传输方式决定部署架构

3. **Tool Decorator（工具装饰器）**
   - `@server.tool()` 自动将函数注册为 MCP 工具
   - 函数签名会自动转换为 JSON Schema
   - **为什么重要**：这是最常用的开发方式

4. **Session（会话）**
   - 管理客户端和服务端的连接生命周期
   - 处理请求路由、错误恢复、进度通知
   - **为什么重要**：理解会话机制才能调试问题

5. **Context（上下文）**
   - 工具函数可以通过 `Context` 访问请求元信息
   - 包括认证信息、进度报告能力等
   - **为什么重要**：实现高级功能（如权限控制）必须掌握

6. **Resource URI（资源标识符）**
   - 每个资源有唯一 URI，如 `file:///path/to/file.txt`
   - 支持模板化，如 `file:///{path}`
   - **为什么重要**：资源定位的核心机制

7. **JSON-RPC 2.0**
   - MCP 的底层通信协议
   - 请求必须有 `id`，响应必须匹配 `id`
   - **为什么重要**：调试协议问题的基础

8. **Lifespan（生命周期）**
   - 服务器启动和关闭时的初始化/清理逻辑
   - 类似 FastAPI 的 lifespan，管理数据库连接等
   - **为什么重要**：生产环境必须处理资源生命周期

## 常见疑问解答

### Q1: MCPServer 和 Server 有什么区别？我应该用哪个？

**答**：
- **MCPServer**（大写 M）：高层 API，用装饰器 `@server.tool()`，代码简洁
- **Server**（小写 s）：底层 API，手动调用 `server.add_tool()`，控制精细

**建议**：90% 的场景用 `MCPServer`，只有在需要自定义协议行为时才用 `Server`。

### Q2: 为什么我的工具注册成功了但客户端看不到？

**答**：检查以下几点：
1. 工具函数必须有 **docstring**（文档字符串），这是工具描述
2. 参数类型必须是 Pydantic 支持的类型
3. 确保调用了 `server.run()` 或 `server.connect()`
4. 查看日志是否有重复工具名称的警告

### Q3: stdio、SSE、HTTP 三种传输方式怎么选？

**答**：
- **stdio**：开发测试用，或桌面应用（如 Claude Desktop）集成
- **SSE**：需要服务端主动推送数据时（如实时日志）
- **Streamable HTTP**：**生产环境首选**，支持现代 Web 标准，可跨网络

**类比**：
- stdio = 对讲机（短距离，点对点）
- SSE = 广播电台（单向推送）
- HTTP = 电话（双向，远距离）

### Q4: 如何调试 MCP 服务？

**答**：
1. 开启调试模式：`MCPServer(debug=True)` 或环境变量 `MCP_DEBUG=true`
2. 查看标准错误输出（stderr），MCP 协议消息走 stdout，日志走 stderr
3. 使用 `mcp dev` 命令（CLI 工具）进行交互式测试
4. 查看源码 `src/mcp/shared/session.py` 中的消息处理逻辑

### Q5: MCP 协议和 FastAPI 有什么关系？可以一起用吗？

**答**：
- MCP 是 **AI 工具协议**，FastAPI 是 **Web API 框架**
- **可以一起用**：MCP 服务可以挂载到 FastAPI/Starlette 应用
- 源码中使用了 Starlette：`src/mcp/server/mcpserver/server.py` 第 17 行
- 典型架构：FastAPI 处理传统 API，MCP 处理 AI 工具调用

```python
# 挂载 MCP 到 Starlette 应用
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

app = Starlette()
manager = StreamableHTTPSessionManager(server, event_store)
app.mount("/mcp", manager.create_app())
```
