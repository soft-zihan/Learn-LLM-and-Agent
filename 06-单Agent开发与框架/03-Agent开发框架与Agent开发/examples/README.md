# Agent App - 智能体应用

一个**真正可运行**的智能体应用，提供 Python 和 Java 两套后端实现，共享同一个 React 前端。

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│  React Frontend (Vite) - 共享前端                          │
│  - 支持切换 Python/Java 后端                                 │
│  - SSE 流式接收                                             │
│  - 工具调用可视化                                            │
─────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
┌──────────────────────────┐  ──────────────────────────┐
│  Python Backend (LangGraph)│  │  Java Backend (Spring AI) │
│  - agent.py (LangGraph)   │  │  - AgentApplication.java   │
│  - mcp_client.py (MCP)    │  │  - mcp/McpClient.java      │
│  - skills.py (Skills)     │  │  - skills/SkillRegistry.java│
│  - memory.py (记忆)       │  │  - memory/MemoryService.java│
│  - tools.py (内置工具)    │  │  - tools/BuiltinTools.java │
│  - subagents.py (子智能体)│  │  - (待补充)                │
│  - human_in_loop.py (人机)│  │  - (待补充)                │
│  - reflection_agent.py    │  │  - (待补充)                │
│  - prompt_engineering.py  │  │  - (待补充)                │
│  - middleware.py (中间件) │  │  - (待补充)                │
│  - main.py (FastAPI)      │  │  - (待补充)                │
│  端口: 8000               │  │  端口: 8080                │
└──────────────────────────┘  ──────────────────────────
                │                       │
                └───────────┬───────────
                            ▼
┌─────────────────────────────────────────────────────────────
│  MCP Server (hello_world)                                   │
│  - greet: 问好工具                                           │
│  - add: 计算工具                                             │
│  - get_weather: 天气查询                                     │
└─────────────────────────────────────────────────────────────
```

## 教程对应关系

| 教程章节 | Python 模块 | Java 模块 | 状态 |
|---------|------------|----------|------|
| 01-LangGraph核心概念 | `agent.py` | `AgentApplication` | ✅ |
| 02-Spring-AI-2.0开发实战 | N/A | `AgentService` | ✅ |
| 03-MCP客户端接入 | `mcp_client.py` | `McpClient.java` | ✅ |
| 04-Skills系统集成 | `skills.py` | `SkillRegistry.java` | ✅ |
| 05-提示词与上下文工程 | `prompt_engineering.py` | 待补充 |  |
| 06-记忆系统设计 | `memory.py` | `MemoryService.java` | ✅ |
| 07-智能体设计模式 | `reflection_agent.py` | 待补充 |  |
| 08-循环与自演进 | `agent.py` (Plan模式) | 待补充 | 🟡 |
| 09-子智能体与路由模式 | `subagents.py` | 待补充 | ✅ |
| 10-人机协作与中断恢复 | `human_in_loop.py` | 待补充 | ✅ |
| 12-测试与可观测性 | 待补充 | 待补充 |  |
| 13-生产部署实践 | `main.py` | 待补充 | 🟡 |

## 快速启动

### 1. 启动 Python 后端

```bash
cd LangGraph
./venv/bin/uvicorn main:app --reload --port 8000
```

### 2. 启动 Java 后端（可选）

```bash
cd spring-ai
mvn spring-boot:run
```

### 3. 启动前端

```bash
cd frontend
npm run dev
```

### 4. 打开浏览器

访问 http://localhost:5173，在页面顶部切换后端。

## 功能演示

### 1. 基础对话
输入"你好"，观察流式输出。

### 2. 工具调用
- "向张三问好" → 调用 `greet` 工具
- "计算 123 + 456" → 调用 `add` 工具
- "北京天气怎么样" → 调用 `get_weather` 工具

### 3. 切换后端
点击页面顶部的 "Python (FastAPI)" 或 "Java (Spring AI)" 按钮。

## 文件结构

```
examples/
├── frontend/                    # 共享前端
│   ├── src/
│   │   ├── App.jsx              # React 聊天界面
│   │   └── App.css              # 样式
│   └── package.json
├── LangGraph/                   # Python 后端（LangGraph + FastAPI）
│   ├── agent.py                 # LangGraph Agent (ReAct + Plan-and-Execute)
│   ├── main.py                  # FastAPI 服务
│   ├── mcp_client.py            # MCP 客户端
│   ├── skills.py                # Skills 系统
│   ├── memory.py                # 记忆系统
│   ├── middleware.py            # 中间件
│   ├── tools.py                 # 内置工具
│   ├── subagents.py             # 子智能体与路由（教程09）
│   ├── human_in_loop.py         # 人机协作与中断恢复（教程10）
│   ├── reflection_agent.py      # 智能体设计模式（教程07）
│   ├── prompt_engineering.py    # 提示词与上下文工程（教程05）
│   ── requirements.txt
├── spring-ai/                   # Java 后端（Spring AI）
│   ├── pom.xml
│   └── src/main/java/com/example/agent/
│       ├── AgentApplication.java
│       ├── mcp/McpClient.java
│       ├── skills/SkillRegistry.java
│       ├── memory/MemoryService.java
│       └── tools/BuiltinTools.java
└── README.md
```

## API 接口

### Python 后端 (http://localhost:8000)

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/chat` | POST | 对话接口（SSE流式） |
| `/api/tools` | GET | 列出可用工具 |
| `/api/health` | GET | 健康检查 |
| `/api/mode` | POST | 切换 Agent 模式 |

### Java 后端 (http://localhost:8080)

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/chat` | POST | 对话接口（SSE流式） |
| `/api/chat/sync` | POST | 同步对话 |
| `/api/tools` | GET | 列出可用工具 |
| `/api/health` | GET | 健康检查 |
# Agent App - 智能体应用

一个**真正可运行**的智能体应用，提供 Python 和 Java 两套后端实现，共享同一个 React 前端。

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│  React Frontend (Vite) - 共享前端                          │
│  - 支持切换 Python/Java 后端                                 │
│  - SSE 流式接收                                             │
│  - 工具调用可视化                                            │
─────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│  Python Backend (LangGraph)│  │  Java Backend (Spring AI) │
│  - agent.py (LangGraph)   │  │  - AgentApplication.java   │
│  - mcp.py (MCP客户端)     │  │  - McpClient.java          │
│  - skills.py (Skills)     │  │  - SkillRegistry.java      │
│  - memory.py (记忆)       │  │  - MemoryService.java      │
│  - tools.py (内置工具)    │  │  - BuiltinTools.java       │
│  端口: 8000               │  │  端口: 8080                │
└──────────────────────────┘  └──────────────────────────
                │                       │
                └───────────┬───────────┘
                            ▼
┌─────────────────────────────────────────────────────────────
│  MCP Server (hello_world)                                   │
│  - greet: 问好工具                                           │
│  - add: 计算工具                                             │
│  - get_weather: 天气查询                                     │
└─────────────────────────────────────────────────────────────
```

## 教程对应关系

| 教程章节 | Python 模块 | Java 模块 | 功能 |
|---------|------------|----------|------|
| 01-LangGraph核心概念 | `agent.py` | `AgentApplication` | ReAct循环 |
| 03-MCP客户端接入 | `mcp.py` | `McpClient.java` | MCP工具桥接 |
| 04-Skills系统集成 | `skills.py` | `SkillRegistry.java` | Skills加载 |
| 05-提示词工程 | `agent.py` | `AgentApplication` | System Prompt |
| 06-记忆系统设计 | `memory.py` | `MemoryService.java` | 三种记忆 |
| 08-循环与自演进 | `agent.py` | `AgentApplication` | Plan-and-Execute |
| 13-生产部署实践 | `main.py` | `AgentController` | API服务化 |

## 快速启动

### 1. 启动 Python 后端

```bash
cd LangGraph
../venv/bin/uvicorn main:app --reload --port 8000
```

### 2. 启动 Java 后端（可选）

```bash
cd spring-ai/demo
mvn spring-boot:run
```

### 3. 启动前端

```bash
cd frontend
npm run dev
```

### 4. 打开浏览器

访问 http://localhost:5173，在页面顶部切换后端。

## 功能演示

### 1. 基础对话
输入"你好"，观察流式输出。

### 2. 工具调用
- "向张三问好" → 调用 `greet` 工具
- "计算 123 + 456" → 调用 `add` 工具
- "北京天气怎么样" → 调用 `get_weather` 工具

### 3. 切换后端
点击页面顶部的 "Python (FastAPI)" 或 "Java (Spring AI)" 按钮。

## 文件结构

```
examples/
├── frontend/                    # 共享前端
│   ├── src/
│   │   ├── App.jsx              # React 聊天界面
│   │   └── App.css              # 样式
│   └── package.json
├── LangGraph/                   # Python 后端（LangGraph + FastAPI）
│   ├── agent.py                 # LangGraph Agent
│   ├── main.py                  # FastAPI 服务
│   ├── mcp.py                   # MCP 客户端
│   ├── skills.py                # Skills 系统
│   ├── memory.py                # 记忆系统
│   ├── middleware.py            # 中间件
│   ├── tools.py                 # 内置工具
│   └── requirements.txt
├── spring-ai/                   # Java 后端（Spring AI）
│   └── demo/
│       └── src/main/java/com/example/agent/
│           ├── AgentApplication.java
│           ├── mcp/McpClient.java
│           ├── skills/SkillRegistry.java
│           ├── memory/MemoryService.java
│           └── tools/BuiltinTools.java
├── venv/                        # Python 虚拟环境
└── README.md
```

## API 接口

### Python 后端 (http://localhost:8000)

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/chat` | POST | 对话接口（SSE流式） |
| `/api/tools` | GET | 列出可用工具 |
| `/api/health` | GET | 健康检查 |
| `/api/mode` | POST | 切换 Agent 模式 |

### Java 后端 (http://localhost:8080)

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/chat` | POST | 对话接口（SSE流式） |
| `/api/chat/sync` | POST | 同步对话 |
| `/api/tools` | GET | 列出可用工具 |
| `/api/health` | GET | 健康检查 |
