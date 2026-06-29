# MCP 客户端接入

> 📅 **更新时间**: 2026-06-29

---

## 目录

- [1. MCP 协议核心概念](#1-mcp-协议核心概念)
- [2. LangGraph 接入 MCP Server](#2-langgraph-接入-mcp-server)
- [3. Spring AI 接入 MCP Server](#3-spring-ai-接入-mcp-server)
- [4. 实战案例](#4-实战案例)

---

## 1. MCP 协议核心概念

### 1.1 MCP 是什么

**MCP（Model Context Protocol）** 是 Anthropic 在 2024 年 11 月发布的**标准化协议**，用于 AI 应用与外部工具/数据源的交互。

> **一句话定义：MCP 就像 AI 界的 USB-C 接口——统一标准连接所有工具。**

### 1.2 解决的核心问题：M×N 困境

在 MCP 出现之前：
- **M 个 AI 应用**（LangGraph、Spring AI、Claude Desktop...）
- **N 个工具**（GitHub、数据库、文件系统...）
- 需要 **M × N** 个集成代码

MCP 通过标准化协议将其简化为 **M + N**：

```
无 MCP（M×N = 50 个集成）:
LangGraph ──→ GitHub
LangGraph ──→ 数据库
Spring AI ──→ GitHub
Spring AI ──→ 数据库
... (指数级增长)

有 MCP（M+N = 12 个实现）:
LangGraph ──┐
Spring AI ──┤── MCP 协议 ──→ GitHub Server
Claude  ────┤              → 数据库 Server
...         ┘              → 文件系统 Server
```

| 应用数(M) | 工具数(N) | 无 MCP (M×N) | 有 MCP (M+N) | 节省 |
|-----------|-----------|-------------|-------------|------|
| 3 | 3 | 9 | 6 | 33% |
| 10 | 20 | 200 | 30 | 85% |
| 50 | 100 | 5000 | 150 | 97% |

### 1.3 三角架构

```
┌─────────────────┐
│   MCP Host      │  (AI 应用：LangGraph / Spring AI)
│                 │
│  ┌───────────┐  │
│  │ MCP Client│  │  (协议客户端：嵌入在 Host 内)
│  └─────┬─────┘  │
│        │        │
│        │ JSON-RPC 2.0
│        │ (stdio / HTTP)
│        │        │
│  ┌─────▼─────┐  │
│  │ MCP Server│  │  (工具提供者：文件系统、GitHub...)
│  └───────────┘  │
└─────────────────┘
```

**三种角色**：
| 角色 | 职责 | 类比 | 示例 |
|------|------|------|------|
| **MCP Host** | 运行 AI 模型的宿主应用 | 电脑主机 | LangGraph Agent、Spring AI 应用 |
| **MCP Client** | Host 内部的协议客户端 | USB 控制器 | `mcp.ClientSession` |
| **MCP Server** | 提供工具/数据的服务 | USB 外设 | `hello_world/server.py` |

### 1.4 三大核心原语

MCP 定义了三种核心原语，代表三种不同类型的交互：

| 原语 | 定义 | 控制方 | 类比 | 示例 |
|------|------|--------|------|------|
| **Tools** | 可执行的操作 | 模型(LLM)决定 | 动词（做事） | `greet`, `add`, `get_weather` |
| **Resources** | 可读取的数据 | 应用(Host)决定 | 名词（数据） | `file://readme.md` |
| **Prompts** | 可复用的提示模板 | 用户决定 | 模板（套路） | `code_review` |

**关键区别**：
- **Tools**：LLM 自主决定调用（可能有副作用）
- **Resources**：应用代码决定读取（只读）
- **Prompts**：用户在 UI 中选择使用（参数化模板）

### 1.5 传输方式对比

| 方式 | 场景 | 优点 | 缺点 |
|------|------|------|------|
| **Stdio** | 本地开发 | 简单、安全、无网络开销 | 只能本地 |
| **HTTP+SSE** | 远程部署 | 支持远程、可穿越防火墙 | 双通道复杂 |
| **Streamable HTTP** | 新标准(2025+) | 简化双通道问题 | 较新，支持少 |

---

## 2. LangGraph 接入 MCP Server

### 2.1 安装依赖

```bash
pip install mcp langgraph langchain-openai
```

### 2.2 连接到本地 MCP Server（Stdio）

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

async def connect_to_server():
    """连接到本地 MCP Server"""
    # 配置 Server 启动参数
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],  # 02章的 hello_world/server.py
        cwd="/path/to/MCP/hello_world",
    )

    # 建立连接
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化连接
            await session.initialize()

            # 列出可用工具
            tools_result = await session.list_tools()
            print("可用工具：")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")

            # 调用工具
            result = await session.call_tool("greet", {"name": "张三"})
            print(f"\n调用结果: {result.content[0].text}")

if __name__ == "__main__":
    asyncio.run(connect_to_server())
```

### 2.3 将 MCP 工具转换为 LangChain 工具

```python
from langchain_core.tools import BaseTool
from mcp import ClientSession
from typing import Type
from pydantic import BaseModel, Field

class MCPToolAdapter(BaseTool):
    """将 MCP 工具适配为 LangChain 工具"""

    session: ClientSession
    mcp_tool: object

    def __init__(self, session, mcp_tool):
        # 动态创建 Pydantic 模型
        class ToolArgs(BaseModel):
            pass

        # 根据 MCP 工具的 schema 添加字段
        for param_name, param_schema in mcp_tool.inputSchema.get("properties", {}).items():
            field_type = str
            if param_schema.get("type") == "integer":
                field_type = int
            elif param_schema.get("type") == "number":
                field_type = float

            field_desc = param_schema.get("description", "")
            setattr(ToolArgs, param_name, Field(..., description=field_desc))

        super().__init__(
            name=mcp_tool.name,
            description=mcp_tool.description,
            args_schema=ToolArgs,
        )

        self.session = session
        self.mcp_tool = mcp_tool

    def _run(self, **kwargs):
        """同步调用（需要异步包装）"""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._arun(**kwargs))

    async def _arun(self, **kwargs):
        """异步调用 MCP 工具"""
        result = await self.session.call_tool(self.mcp_tool.name, kwargs)
        return result.content[0].text


async def create_mcp_tools(session):
    """从 MCP Server 发现并创建所有工具"""
    tools_result = await session.list_tools()
    tools = []

    for mcp_tool in tools_result.tools:
        tool = MCPToolAdapter(session, mcp_tool)
        tools.append(tool)

    return tools
```

### 2.4 在 LangGraph Agent 中使用

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


async def build_mcp_agent():
    """构建接入 MCP 的 Agent"""
    
    # LLM
    llm = ChatOpenAI(
        model="qwen3.6-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="your-api-key",
    )

    # 1. 连接到 MCP Server
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
        cwd="path/to/MCP/hello_world",
    )

    # 2. 建立连接并获取工具
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await create_mcp_tools(session)

            # 3. 绑定工具到 LLM
            llm_with_tools = llm.bind_tools(mcp_tools)

            # 4. 定义 Agent 节点
            def agent_node(state: AgentState) -> AgentState:
                response = llm_with_tools.invoke(state["messages"])
                return {"messages": [response]}

            # 5. 构建图
            graph = StateGraph(AgentState)
            graph.add_node("agent", agent_node)
            graph.add_edge(START, "agent")
            graph.add_edge("agent", END)

            app = graph.compile()

            # 6. 执行
            result = await app.ainvoke({
                "messages": [HumanMessage(content="问候一下张三")]
            })

            print(result["messages"][-1].content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(build_mcp_agent())
```

---

## 3. Spring AI 接入 MCP Server

### 3.1 Maven 依赖

```xml
<!-- Spring AI MCP Client -->
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-mcp-client-spring-boot-starter</artifactId>
</dependency>

<!-- Spring AI OpenAI（兼容 DashScope） -->
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
</dependency>
```

### 3.2 配置 MCP Client（application.yml）

```yaml
spring:
  ai:
    # MCP Client 配置
    mcp:
      client:
        enabled: true
        name: my-agent
        version: 1.0.0
        request-timeout: 30s
        # Stdio 传输
        stdio:
          servers-configuration: classpath:/mcp-servers.json
    # OpenAI 兼容 API（DashScope）
    openai:
      api-key: ${OPENAI_API_KEY}
      base-url: https://dashscope.aliyuncs.com/compatible-mode
      chat:
        options:
          model: qwen3.6-plus
          temperature: 0.7
```

### 3.3 MCP Server 配置（mcp-servers.json）

```json
{
  "mcpServers": [
    {
      "name": "hello-world",
      "command": "python",
      "args": ["server.py"],
      "workingDirectory": "/path/to/MCP/hello_world"
    }
  ]
}
```

### 3.4 Spring AI 自动发现 MCP 工具

```java
package com.example.agent;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.tool.ToolCallingManager;
import org.springframework.ai.mcp.client.McpToolCallbackProvider;
import org.springframework.stereotype.Service;

/**
 * MCP Agent 服务
 *
 * Spring AI 会自动从 MCP Server 发现工具并注册为 Tool Callbacks
 */
@Service
public class McpAgentService {

    private final ChatClient chatClient;

    /**
     * ChatClient 已自动注入 MCP 工具
     * 无需手动注册，Spring AI 会自动发现并绑定
     */
    public McpAgentService(ChatClient.Builder builder,
                           McpToolCallbackProvider mcpProvider) {
        this.chatClient = builder
                .defaultSystem("""
                        你是一个智能助手，可以使用 MCP 工具帮助用户。
                        
                        可用 MCP 工具：
                        - greet: 问候某人
                        - add: 两个数相加
                        - get_weather: 查询天气
                        
                        请根据需求选择合适的工具。
                        """)
                .defaultTools(mcpProvider)  // 注入 MCP 工具
                .build();
    }

    /**
     * 同步调用
     */
    public String chat(String userMessage) {
        return chatClient.prompt()
                .user(userMessage)
                .call()
                .content();
    }

    /**
     * 流式调用
     */
    public Flux<String> chatStream(String userMessage) {
        return chatClient.prompt()
                .user(userMessage)
                .stream()
                .content();
    }
}
```

### 3.5 REST 控制器

```java
package com.example.agent;

import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

@RestController
@RequestMapping("/api/mcp")
public class McpAgentController {

    private final McpAgentService agentService;

    public McpAgentController(McpAgentService agentService) {
        this.agentService = agentService;
    }

    /**
     * 同步对话
     * POST /api/mcp/chat
     * Body: {"message": "问候一下张三"}
     */
    @PostMapping("/chat")
    public ChatResponse chat(@RequestBody ChatRequest request) {
        return new ChatResponse(agentService.chat(request.message()));
    }

    /**
     * 流式对话（SSE）
     * POST /api/mcp/chat/stream
     */
    @PostMapping(value = "/chat/stream", 
                 produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> chatStream(@RequestBody ChatRequest request) {
        return agentService.chatStream(request.message());
    }

    public record ChatRequest(String message) {}
    public record ChatResponse(String content) {}
}
```

### 3.6 Spring AI MCP 客户端编程式接入

与 Python 手动连接 MCP Server 对应的 Java 编程式实现：

```java
package com.example.agent;

import io.modelcontextprotocol.client.McpClient;
import io.modelcontextprotocol.client.transport.StdioClientTransport;
import io.modelcontextprotocol.spec.McpSchema;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.function.FunctionToolCallback;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * MCP 客户端编程式接入服务
 * 
 * 对应 Python 代码：
 * - mcp.ClientSession → McpClient.SyncSpec
 * - stdio_client() → StdioClientTransport
 * - session.list_tools() → mcpClient.listTools()
 * - session.call_tool() → mcpClient.callTool()
 */
@Service
public class McpClientService {

    private final McpClient.SyncSpec mcpClient;
    private final ChatClient chatClient;

    public McpClientService(ChatModel chatModel) {
        // 1. 创建 Stdio 传输（对应 Python 的 stdio_client）
        StdioClientTransport transport = new StdioClientTransport(
            "python",  // 命令
            new String[]{"server.py"},  // 参数
            "/path/to/MCP/hello_world"  // 工作目录
        );

        // 2. 创建 MCP 客户端（对应 Python 的 ClientSession）
        this.mcpClient = McpClient.sync(transport)
            .requestTimeout(java.time.Duration.ofSeconds(30))
            .build();

        // 3. 初始化连接
        mcpClient.initialize();

        // 4. 发现并转换工具
        List<ToolCallback> toolCallbacks = discoverAndConvertTools();

        // 5. 构建 ChatClient 并注入 MCP 工具
        this.chatClient = ChatClient.builder(chatModel)
            .defaultSystem("""
                你是一个智能助手，可以使用 MCP 工具帮助用户。
                请根据需求选择合适的工具。
                """)
            .defaultTools(toolCallbacks.toArray(new ToolCallback[0]))
            .build();
    }

    /**
     * 发现 MCP 工具并转换为 Spring AI ToolCallback
     * 对应 Python 的 MCPToolAdapter
     */
    private List<ToolCallback> discoverAndConvertTools() {
        List<ToolCallback> tools = new ArrayList<>();

        // 列出所有工具（对应 Python 的 session.list_tools()）
        McpSchema.ListToolsResult toolsResult = mcpClient.listTools();
        
        for (McpSchema.Tool mcpTool : toolsResult.tools()) {
            // 将每个 MCP 工具转换为 Spring AI ToolCallback
            ToolCallback callback = FunctionToolCallback.builder(
                    mcpTool.name(),
                    (Map<String, Object> args) -> callMcpTool(mcpTool.name(), args)
                )
                .description(mcpTool.description())
                .inputSchema(mcpTool.inputSchema())
                .build();
            
            tools.add(callback);
            System.out.println("✅ 注册 MCP 工具: " + mcpTool.name());
        }

        return tools;
    }

    /**
     * 调用 MCP 工具
     * 对应 Python 的 session.call_tool()
     */
    private String callMcpTool(String toolName, Map<String, Object> args) {
        McpSchema.CallToolRequest request = new McpSchema.CallToolRequest(toolName, args);
        McpSchema.CallToolResult result = mcpClient.callTool(request);
        
        // 提取结果文本
        return result.content().stream()
            .filter(c -> c instanceof McpSchema.TextContent)
            .map(c -> ((McpSchema.TextContent) c).text())
            .reduce("", (a, b) -> a + b);
    }

    /**
     * 同步调用
     */
    public String chat(String userMessage) {
        return chatClient.prompt()
            .user(userMessage)
            .call()
            .content();
    }

    /**
     * 流式调用
     */
    public reactor.core.publisher.Flux<String> chatStream(String userMessage) {
        return chatClient.prompt()
            .user(userMessage)
            .stream()
            .content();
    }

    /**
     * 清理资源
     */
    public void close() {
        if (mcpClient != null) {
            mcpClient.close();
        }
    }
}
```

---

## 4. 实战案例

### 4.1 LangGraph 连接 hello_world MCP Server

完整示例见：`examples/12-mcp-client-langgraph.py`

**关键步骤**：
1. 启动 MCP Server：`python server.py`
2. 使用 `StdioServerParameters` 配置连接
3. 用 `MCPToolAdapter` 转换工具
4. 绑定到 LangGraph Agent
5. 执行并获取结果

### 4.2 Spring AI 连接 hello_world MCP Server

完整示例见：`examples/spring-ai-demo/`

**关键步骤**：
1. 配置 `application.yml` 中的 MCP Client
2. 创建 `mcp-servers.json` 指定 Server 路径
3. Spring AI 自动发现工具
4. 注入 `McpToolCallbackProvider`
5. ChatClient 自动绑定工具

### 4.3 对比总结

| 维度 | LangGraph | Spring AI |
|------|-----------|-----------|
| 连接方式 | 手动 `StdioServerParameters` | 配置 `mcp-servers.json` |
| 工具发现 | 手动 `list_tools()` + 适配 | 自动发现并注册 |
| 工具绑定 | 手动 `bind_tools()` | 自动注入 `McpToolCallbackProvider` |
| 灵活性 | 高（完全控制） | 中（约定优于配置） |
| 适合场景 | Python Agent 开发 | Java 微服务 |

---

## 5. 最佳实践

### 5.1 错误处理

```python
# LangGraph
try:
    result = await session.call_tool("greet", {"name": "张三"})
except Exception as e:
    print(f"MCP 工具调用失败: {e}")
    # 降级处理：使用备用方案
```

```java
// Spring AI
try {
    String response = agentService.chat("问候张三");
} catch (Exception e) {
    log.error("MCP 调用失败", e);
    // 降级处理
}
```

### 5.2 超时设置

```python
# LangGraph
server_params = StdioServerParameters(
    command="python",
    args=["server.py"],
    cwd="...",
    # 超时设置（如果支持）
)
```

```yaml
# Spring AI
spring:
  ai:
    mcp:
      client:
        request-timeout: 30s  # 30 秒超时
```

### 5.3 多 Server 连接

```python
# LangGraph：连接多个 Server
servers = [
    StdioServerParameters(command="python", args=["server1.py"], cwd="..."),
    StdioServerParameters(command="python", args=["server2.py"], cwd="..."),
]

# 分别连接，合并工具列表
```

```json
// Spring AI：配置多个 Server
{
  "mcpServers": [
    {"name": "hello-world", "command": "python", "args": ["server.py"]},
    {"name": "github", "command": "python", "args": ["github_server.py"]}
  ]
}
```
