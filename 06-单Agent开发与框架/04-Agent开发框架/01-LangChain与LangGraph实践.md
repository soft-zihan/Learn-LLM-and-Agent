# LangChain 与 LangGraph 最新实践(2026 v1.0 稳定版)

> 📅 **更新时间**: 2026-06-17  

---

## 目录

- [1. LangChain v1.0 核心变化与概念](#1-langchain-v10-核心变化与概念)
- [2. LangGraph v1.0 状态图与工作流编排](#2-langgraph-v10-状态图与工作流编排)
- [3. 多 Agent 协作模式](#3-多-agent-协作模式)
- [4. 最新特性与最佳实践(2025-2026)](#4-最新特性与最佳实践2025-2026)
- [5. 实战案例:智能代码审查 Agent](#5-实战案例智能代码审查-agent)
- [6. 参考资源](#6-参考资源)

---

## 1. LangChain v1.0 核心变化与概念

### 1.1 什么是 LangChain v1.0？

LangChain v1.0（2025年10月发布）是一个专注于生产环境的 Agent 开发框架。相比早期版本，v1.0 进行了重大架构简化：

**v1.0 核心改进：**
- **create_agent**：新的标准 Agent 构建方式，替代 `langgraph.prebuilt.create_react_agent`
- **中间件系统（Middleware）**：提供细粒度的 Agent 执行控制
- **标准化内容块（Content Blocks）**：跨提供商的统一 LLM 输出接口
- **简化的命名空间**：专注于 Agent 构建的核心抽象，历史功能迁移至 `langchain-classic`

**核心价值：**
- **快速开发**：通过 `create_agent` 快速构建标准 Agent
- **中间件定制**：通过中间件钩子实现深度定制
- **LangGraph 底层支持**：自动获得持久化、流式输出、检查点等生产级特性

### 1.2 安装与环境配置

```bash
# LangChain v1.0（需要 Python 3.10+）
pip install -U langchain

# LangGraph v1.0
pip install -U langgraph

# 可选：需要兼容旧版功能
pip install langchain-classic

# 模型提供商包
pip install langchain-openai langchain-anthropic langchain-ollama
```

### 1.3 create_agent：LangChain v1.0 的 Agent 标准构建方式

`create_agent` 是 LangChain v1.0 引入的标准 Agent 构建方式，基于 LangGraph 底层实现，提供更简单的接口和更强的定制能力：

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

# 定义工具
@tool
def search_web(query: str) -> str:
    """搜索网络获取最新信息"""
    return f"搜索结果: {query}"

@tool
def calculate(expression: str) -> str:
    """执行数学计算"""
    return str(eval(expression))

# 创建 Agent（v1.0 推荐方式）
agent = create_agent(
    model="openai:gpt-5",  # 或 ChatOpenAI(model="gpt-5") 实例
    tools=[search_web, calculate],
    system_prompt="你是一个有用的助手，可以使用工具帮助用户。"
)

# 执行 Agent
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "北京今天天气如何？如果温度超过30度，计算30-25的差值"}
    ]
})

print(result["messages"][-1]["content"])
```

**create_agent 的核心优势：**
- **简单易用**：3行代码即可创建功能完整的 ReAct Agent
- **中间件支持**：通过中间件实现深度定制（见下文）
- **LangGraph 底层**：自动获得持久化、流式输出、检查点等特性
- **结构化输出**：内置 `response_format` 支持结构化输出

### 1.4 中间件系统（Middleware）：v1.0 的核心特性

中间件是 LangChain v1.0 的核心创新，提供细粒度的 Agent 执行控制：

#### 内置中间件

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware,
    SummarizationMiddleware,
    HumanInTheLoopMiddleware
)

agent = create_agent(
    model="openai:gpt-5",
    tools=[search_web, calculate],
    middleware=[
        # PII 脱敏中间件
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        
        # 对话摘要中间件（防止上下文溢出）
        SummarizationMiddleware(
            model="openai:gpt-5",
            trigger={"tokens": 500}  # 超过500 tokens时触发摘要
        ),
        
        # 人类介入中间件
        HumanInTheLoopMiddleware(
            interrupt_on={
                "calculate": {  # 对特定工具调用要求人类审批
                    "allowed_decisions": ["approve", "edit", "reject"]
                }
            }
        )
    ]
)
```

#### 自定义中间件

中间件提供6个钩子点，覆盖 Agent 执行的完整生命周期：

| 钩子 | 执行时机 | 典型用途 |
|------|---------|---------|
| `before_agent` | Agent 调用前 | 加载记忆、验证输入 |
| `before_model` | 每次 LLM 调用前 | 更新提示词、裁剪消息 |
| `wrap_model_call` | LLM 调用周围 | 拦截和修改请求/响应 |
| `wrap_tool_call` | 工具调用周围 | 拦截和修改工具执行 |
| `after_model` | LLM 响应后 | 验证输出、应用护栏 |
| `after_agent` | Agent 完成后 | 保存结果、清理资源 |

```python
from dataclasses import dataclass
from typing import Callable
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ModelResponse

# 定义上下文 schema
@dataclass
class AgentContext:
    user_expertise: str = "beginner"  # 用户专业水平

# 自定义中间件：根据用户水平动态选择模型和工具
class ExpertiseBasedToolMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        user_level = request.runtime.context.user_expertise
        
        if user_level == "expert":
            # 专家：使用更强大的模型和高级工具
            model = ChatOpenAI(model="gpt-5")
            tools = [advanced_search, data_analysis]
        else:
            # 初学者：使用更快的模型和简单工具
            model = ChatOpenAI(model="gpt-5-nano")
            tools = [simple_search, basic_calculator]
        
        # 覆盖模型和工具后继续执行
        return handler(request.override(model=model, tools=tools))

# 使用自定义中间件
agent = create_agent(
    model="openai:gpt-5",
    tools=[simple_search, advanced_search, basic_calculator, data_analysis],
    middleware=[ExpertiseBasedToolMiddleware()],
    context_schema=AgentContext  # 指定上下文 schema
)

# 执行时传入上下文
result = agent.invoke(
    {"messages": [{"role": "user", "content": "分析这段代码的性能"}]},
    context=AgentContext(user_expertise="expert")
)
```

### 1.5 核心组件（保留的经典抽象）

#### Models（模型层）

LangChain 支持多种模型类型：

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

# OpenAI (gpt-5.2)
llm_openai = ChatOpenAI(
    model="gpt-5.2",
    temperature=0.7,
    max_tokens=1000,
    api_key="your-api-key"
)

# Anthropic (claude-4.6)
llm_anthropic = ChatAnthropic(
    model="claude-4-6-20251001",
    temperature=0.7,
    max_tokens=1000
)

# Ollama (本地部署)
llm_ollama = ChatOllama(
    model="llama3.1:8b",
    temperature=0.7,
    base_url="http://localhost:11434"
)
```

#### Prompts（提示词）

提示词模板是 LLM 应用的灵魂：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 基础提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{domain}助手。你的任务是帮助用户解决{task}相关的问题。"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ("human", "请以{language}回复。")
])

# 格式化提示词
formatted = prompt.format_messages(
    domain="编程",
    task="代码优化",
    input="如何改进这段 Python 代码的性能？",
    language="中文"
)
```

#### Chains（链）

链是 LangChain 的核心抽象，将多个组件串联执行：

```python
from langchain_core.output_parsers import StrOutputParser

# 使用 LCEL (LangChain Expression Language)
chain = prompt | llm_openai | StrOutputParser()

# 执行链
result = chain.invoke({
    "domain": "数据分析",
    "task": "可视化",
    "input": "如何用 matplotlib 创建交互式图表？",
    "language": "中文",
    "chat_history": []
})

print(result)
```

#### Tools（工具）

工具让 LLM 能够执行实际操作：

```python
from langchain_core.tools import tool
from typing import Annotated

# v1.0 推荐：使用 @tool 装饰器（简洁方式）
@tool
def search_web(query: Annotated[str, "搜索查询关键词"]) -> str:
    """搜索网络获取最新信息"""
    # 实际应用中应调用搜索 API
    return f"搜索结果: {query}"

@tool
def calculate(expression: Annotated[str, "数学表达式"]) -> str:
    """执行数学计算"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def get_weather(city: Annotated[str, "城市名称"]) -> str:
    """获取天气信息"""
    # 模拟天气 API 调用
    return f"{city} 今天天气晴朗，温度 25°C"

tools = [search_web, calculate, get_weather]
```

**v1.0 新特性：使用 Pydantic 定义复杂工具参数**

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="搜索查询关键词")
    max_results: int = Field(default=10, description="最大结果数量")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "搜索网络获取最新信息"
    args_schema: type[BaseModel] = SearchInput
    
    def _run(self, query: str, max_results: int = 10) -> str:
        """同步执行搜索"""
        return f"搜索结果: {query} (共 {max_results} 条)"
    
    async def _arun(self, query: str, max_results: int = 10) -> str:
        """异步执行搜索（推荐用于生产）"""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            # 异步 API 调用
            return f"异步搜索结果: {query}"
```

#### Structured Output（结构化输出）

v1.0 原生支持结构化输出，减少额外 LLM 调用：

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class MovieReview(BaseModel):
    title: str = Field(description="电影名称")
    rating: float = Field(ge=0, le=10, description="评分 0-10")
    pros: list[str] = Field(description="优点列表")
    cons: list[str] = Field(description="缺点列表")
    summary: str = Field(description="一句话总结")

# 方式 1：使用 with_structured_output（传统方式）
llm = ChatOpenAI(model="gpt-5")
structured_llm = llm.with_structured_output(MovieReview)
review = structured_llm.invoke("请评析《星际穿越》")
print(review.title)  # 直接访问属性

# 方式 2：在 create_agent 中使用 response_format（v1.0 新特性）
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model="openai:gpt-5",
    tools=[search_web],
    response_format=ToolStrategy(MovieReview),  # 结构化输出集成到主循环
    system_prompt="帮助用户分析电影"
)
```

#### Memory（记忆）

记忆系统让 Agent 能够记住对话历史：

```python
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# 内存存储（开发测试用）
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 创建带记忆的链
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# 使用
result = chain_with_memory.invoke(
    {"input": "我刚才问了什么问题？"},
    config={"configurable": {"session_id": "user_123"}}
)
```

### 1.6 LangChain Expression Language (LCEL)

LCEL 是 LangChain 的声明式编排语言，类似 Unix 管道：

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

# 并行执行
parallel_chain = RunnableParallel({
    "summary": summary_chain,
    "keywords": keyword_chain,
    "sentiment": sentiment_chain
})

# 条件分支
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: x["type"] == "code", code_chain),
    (lambda x: x["type"] == "text", text_chain),
    default_chain
)

# 完整管道
full_pipeline = (
    RunnableLambda(lambda x: preprocess(x))
    | parallel_chain
    | RunnableLambda(lambda x: merge_results(x))
)
```

## 2. LangGraph v1.0 状态图与工作流编排

### 2.1 LangGraph v1.0 重要变化

LangGraph v1.0（2025年10月）是第一个稳定版本，主要变化：

**废弃的 API（不再使用）：**
- ❌ `set_entry_point()` → ✅ `add_edge(START, 'node')`
- ❌ `set_finish_point()` → ✅ `add_edge('node', END)`
- ❌ `langgraph.prebuilt.create_react_agent` → ✅ `langchain.agents.create_agent`
- ❌ `ToolExecutor` → ✅ `ToolNode`

**推荐的导入方式（v1.0）：**
```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessageState, add_messages
from langgraph.prebuilt import create_react_agent, ToolNode  # 仍可用，但推荐 create_agent
```

### 2.2 为什么需要 LangGraph？

LangChain 的 `create_agent` 适合标准 Agent 模式，但复杂场景需要 LangGraph：
- **状态管理**：在多个步骤间维护和更新状态
- **循环执行**：支持条件循环和迭代
- **图结构**：非线性的控制流（分支、合并、循环）
- **人类介入**：在关键节点等待人类审批
- **混合工作流**：确定性组件 + Agent 组件的混合编排

LangGraph 基于有向图（StateGraph）解决了这些问题。

### 2.3 StateGraph 核心概念

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages

# 定义状态结构（v1.0 推荐使用 MessageState 或自定义 TypedDict）
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    current_step: str
    iteration_count: int
    final_answer: str
    needs_human_review: bool

# v1.0 简化方式：使用内置 MessageState
from langgraph.graph.message import MessageState
# MessageState 已经包含 messages: Annotated[list, add_messages]
```

**状态字段注解：**
- `add_messages`：自动追加消息而非覆盖（reducer 模式）
- 自定义 reducer：定义状态如何更新

### 2.4 构建第一个 Agent 图（v1.0 API）

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 定义节点函数
def research_node(state: AgentState) -> AgentState:
    """研究节点：搜索信息"""
    print("🔍 执行研究...")
    
    # 调用 LLM 生成搜索查询
    response = llm_openai.invoke([
        HumanMessage(content="基于以下问题，生成搜索查询：\n" + 
                    state["messages"][-1].content)
    ])
    
    # 执行搜索（模拟）
    search_results = f"搜索结果：关于 {response.content} 的信息..."
    
    return {
        "messages": [AIMessage(content=search_results)],
        "current_step": "research_completed"
    }

def analyze_node(state: AgentState) -> AgentState:
    """分析节点：分析研究结果"""
    print("📊 分析数据...")
    
    response = llm_openai.invoke([
        HumanMessage(content="分析以下信息并总结关键点：\n" + 
                    state["messages"][-1].content)
    ])
    
    return {
        "messages": [AIMessage(content=response.content)],
        "current_step": "analysis_completed"
    }

def synthesize_node(state: AgentState) -> AgentState:
    """综合节点：生成最终答案"""
    print("✍️ 生成答案...")
    
    response = llm_openai.invoke([
        HumanMessage(content="基于分析结果，生成 comprehensive 答案：\n" + 
                    state["messages"][-1].content)
    ])
    
    return {
        "messages": [AIMessage(content=response.content)],
        "final_answer": response.content,
        "current_step": "completed"
    }

# 构建图（v1.0 使用 START/END sentinel）
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("research", research_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("synthesize", synthesize_node)

# v1.0：使用 add_edge(START, 'node') 替代 set_entry_point()
workflow.add_edge(START, "research")

# 添加边
workflow.add_edge("research", "analyze")
workflow.add_edge("analyze", "synthesize")
workflow.add_edge("synthesize", END)

# 编译图
agent = workflow.compile()

# 可视化图结构（v1.0 新增实用功能）
print(agent.get_graph().draw_mermaid())
```

### 2.5 条件边与循环

```python
from langgraph.graph import StateGraph

def should_continue(state: AgentState) -> str:
    """决定下一步：继续研究还是结束"""
    if state.get("iteration_count", 0) < 3:
        return "continue"
    return "end"

# 添加条件边
workflow.add_conditional_edges(
    "analyze",
    should_continue,
    {
        "continue": "research",  # 循环回研究节点
        "end": "synthesize"      # 进入综合节点
    }
)
```

### 2.6 工具调用 Agent

#### 方式 1：使用 create_agent（v1.0 推荐）

```python
from langchain.agents import create_agent

# v1.0 最简单的方式
agent = create_agent(
    model="openai:gpt-5",
    tools=[search_web, calculate, get_weather],
    system_prompt="你是一个有用的助手，可以使用工具帮助用户。"
)

# 执行
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "北京今天天气如何？如果温度超过 30 度，计算 30-25 的差值"}
    ]
})

print(result["messages"][-1]["content"])
```

#### 方式 2：使用 create_react_agent（LangGraph 预构建，仍可用）

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 创建 ReAct Agent（适用于需要更多控制的场景）
react_agent = create_react_agent(
    model="openai:gpt-5",  # 或直接传入 ChatOpenAI 实例
    tools=[search_web, calculate, get_weather],
    prompt=ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手，可以使用工具帮助用户。"),
        MessagesPlaceholder("messages")
    ])
)

# 执行
result = react_agent.invoke({
    "messages": [HumanMessage(content="北京今天天气如何？")]
})

print(result["messages"][-1].content)
```

#### 方式 3：手动构建 ReAct 图（完全控制）

```python
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage

def call_model(state):
    """调用 LLM"""
    messages = state["messages"]
    response = llm_openai.bind_tools([search_web, calculate, get_weather]).invoke(messages)
    return {"messages": [response]}

def should_continue_to_tools(state) -> str:
    """判断是否继续调用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# 构建 ReAct 图
workflow = StateGraph(MessageState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode([search_web, calculate, get_weather]))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue_to_tools,
    {"tools": "tools", END: END}
)
workflow.add_edge("tools", "agent")

agent = workflow.compile()
```

### 2.7 人类介入（Human-in-the-Loop）

```python
from langgraph.types import interrupt

def review_node(state: AgentState) -> AgentState:
    """需要人类审核的节点"""
    
    # 触发中断，等待人类输入
    human_feedback = interrupt({
        "question": "请审核以下内容：",
        "content": state["messages"][-1].content
    })
    
    return {
        "messages": [HumanMessage(content=f"人类反馈：{human_feedback}")],
        "needs_human_review": False
    }

# 在图中添加审核节点
workflow.add_node("review", review_node)
workflow.add_edge("analyze", "review")
workflow.add_edge("review", "synthesize")
```

**使用中断恢复：**

```python
# 第一次执行（会在 review 节点中断）
thread_config = {"configurable": {"thread_id": "123"}}
result = agent.invoke(initial_state, config=thread_config)

# 人类审核后，恢复执行
human_input = "内容审核通过，继续"
result = agent.invoke(
    {"human_feedback": human_input},
    config=thread_config
)
```

### 2.8 检查点与持久化

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# 内存检查点（开发用）
memory_saver = MemorySaver()

# SQLite 检查点（生产用）
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
sqlite_saver = SqliteSaver(conn)

# 编译时指定检查点
agent_with_checkpoint = workflow.compile(checkpointer=sqlite_saver)

# 执行（自动保存状态）
result = agent_with_checkpoint.invoke(
    {"messages": [HumanMessage(content="执行复杂任务...")]},
    config={"configurable": {"thread_id": "session_001"}}
)

# 恢复执行
result = agent_with_checkpoint.invoke(
    None,  # 从上次中断处继续
    config={"configurable": {"thread_id": "session_001"}}
)
```

## 3. 多 Agent 协作模式

### 3.1 Supervisor 模式

一个主 Agent 协调多个专业 Agent：

```python
from langgraph.graph import StateGraph, END
from typing import Literal

class SupervisorState(TypedDict):
    messages: Annotated[List, add_messages]
    next: str  # 下一个执行的 Agent
    final_output: str

def supervisor_node(state: SupervisorState) -> SupervisorState:
    """Supervisor：决定下一个执行的 Agent"""
    
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个 supervisor，管理团队完成任务。
根据当前进度，选择下一个执行的 agent：
- researcher：负责搜索和研究
- coder：负责编写代码
- reviewer：负责审核代码
- finalizer：负责生成最终答案

只回复 agent 名称。"""),
        MessagesPlaceholder("messages")
    ])
    
    chain = supervisor_prompt | llm_openai | StrOutputParser()
    next_agent = chain.invoke({"messages": state["messages"]})
    
    return {"next": next_agent.strip()}

def researcher_node(state: SupervisorState) -> SupervisorState:
    """研究员 Agent"""
    response = llm_openai.invoke([
        HumanMessage(content="作为研究员，请搜索并分析：\n" + 
                    state["messages"][-1].content)
    ])
    return {"messages": [AIMessage(content=response.content)]}

def coder_node(state: SupervisorState) -> SupervisorState:
    """程序员 Agent"""
    response = llm_openai.invoke([
        HumanMessage(content="作为程序员，请编写代码解决：\n" + 
                    state["messages"][-1].content)
    ])
    return {"messages": [AIMessage(content=response.content)]}

def reviewer_node(state: SupervisorState) -> SupervisorState:
    """审核员 Agent"""
    response = llm_openai.invoke([
        HumanMessage(content="作为审核员，请检查代码质量：\n" + 
                    state["messages"][-1].content)
    ])
    return {"messages": [AIMessage(content=response.content)]}

def finalizer_node(state: SupervisorState) -> SupervisorState:
    """总结 Agent"""
    response = llm_openai.invoke([
        HumanMessage(content="基于团队工作，生成最终答案：\n" + 
                    "\n".join([m.content for m in state["messages"]]))
    ])
    return {
        "messages": [AIMessage(content=response.content)],
        "final_output": response.content
    }

# 构建 Supervisor 图
supervisor_graph = StateGraph(SupervisorState)

supervisor_graph.add_node("supervisor", supervisor_node)
supervisor_graph.add_node("researcher", researcher_node)
supervisor_graph.add_node("coder", coder_node)
supervisor_graph.add_node("reviewer", reviewer_node)
supervisor_graph.add_node("finalizer", finalizer_node)

supervisor_graph.set_entry_point("supervisor")

# Supervisor 决定下一个节点
def route_supervisor(state: SupervisorState) -> Literal["researcher", "coder", "reviewer", "finalizer", "END"]:
    next_agent = state.get("next", "").lower()
    if next_agent in ["researcher", "coder", "reviewer", "finalizer"]:
        return next_agent
    return "END"

supervisor_graph.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "coder": "coder",
        "reviewer": "reviewer",
        "finalizer": "finalizer",
        "END": END
    }
)

# 专业 Agent 执行后返回 supervisor
for agent in ["researcher", "coder", "reviewer", "finalizer"]:
    supervisor_graph.add_edge(agent, "supervisor")

supervisor_app = supervisor_graph.compile()
```

### 3.2 Hierarchical（层级）模式

多层级 Agent 团队：

```python
# 第一层：项目管理
pm_graph = StateGraph(PMState)
pm_graph.add_node("planner", planning_node)
pm_graph.add_node("coordinator", coordination_node)

# 第二层：执行团队
execution_graph = StateGraph(ExecState)
execution_graph.add_node("researcher", researcher_node)
execution_graph.add_node("developer", developer_node)
execution_graph.add_node("tester", tester_node)

# 嵌套图
main_graph = StateGraph(MainState)
main_graph.add_node("management", pm_graph.compile())
main_graph.add_node("execution", execution_graph.compile())
main_graph.add_edge("management", "execution")
```

### 3.3 协作模式选择指南

| 模式 | 适用场景 | 复杂度 | 控制粒度 |
|------|---------|--------|---------|
| Supervisor | 任务明确、需要协调 | 中 | 细粒度 |
| Hierarchical | 大型项目、多层管理 | 高 | 最细 |
| Swarm | 自主协作、快速响应 | 低 | 粗粒度 |
| Pipeline | 线性流程、阶段明确 | 低 | 中等 |

## 4. 最新特性与最佳实践（2025-2026）

### 4.1 LangChain v1.0 核心变化（2026）

LangChain 在 2026 年初发布了 v1.0 稳定版本，带来了重大架构改进：

**1. 统一的模块导入系统**

```python
# v1.0 之前的导入方式（已废弃）
from langchain.llms import OpenAI  # ❌ 不再支持
from langchain.chat_models import ChatOpenAI  # ❌ 不再支持

# v1.0 推荐导入方式
from langchain_openai import ChatOpenAI  # ✅ 独立包
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
```

**2. 强类型状态管理**

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# v1.0 推荐使用 Pydantic v2 进行状态验证
class AgentState(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    current_step: str = Field(default="init")
    metadata: dict[str, any] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True  # 自动验证状态更新
```

**3. 原生异步优先架构**

```python
# v1.0 所有核心 API 默认异步
from langchain_core.runnables import Runnable

# 异步调用（推荐）
async def run_agent():
    result = await chain.ainvoke({"input": "hello"})
    
    # 流式输出
    async for chunk in chain.astream({"input": "hello"}):
        print(chunk, end="", flush=True)

# 同步调用仍然支持，但会调用异步底层
result = chain.invoke({"input": "hello"})
```

**4. 改进的工具定义系统**

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# v1.0 使用 Pydantic 定义工具参数
class SearchInput(BaseModel):
    query: str = Field(description="搜索查询关键词")
    max_results: int = Field(default=10, description="最大结果数量")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "搜索网络获取最新信息"
    args_schema: type[BaseModel] = SearchInput
    
    def _run(self, query: str, max_results: int = 10) -> str:
        """同步执行搜索"""
        # 实现搜索逻辑
        return f"搜索结果: {query} (共 {max_results} 条)"
    
    async def _arun(self, query: str, max_results: int = 10) -> str:
        """异步执行搜索（推荐）"""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            # 异步 API 调用
            return f"异步搜索结果: {query}"

# 使用装饰器简化（仍支持）
@tool
def calculate(expression: Annotated[str, "数学表达式"]) -> str:
    """执行数学计算"""
    return str(eval(expression))
```

**5. 结构化输出原生支持**

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class MovieReview(BaseModel):
    title: str = Field(description="电影名称")
    rating: float = Field(ge=0, le=10, description="评分 0-10")
    pros: list[str] = Field(description="优点列表")
    cons: list[str] = Field(description="缺点列表")
    summary: str = Field(description="一句话总结")

# v1.0 直接使用 with_structured_output
llm = ChatOpenAI(model="gpt-5")
structured_llm = llm.with_structured_output(MovieReview)

review = structured_llm.invoke("请评析《星际穿越》")
print(review.title)  # 直接访问属性
print(review.rating)
```

### 4.2 LangGraph 2025-2026 新特性

**1. 改进的 create_react_agent API（v0.2+）**

```python
from langgraph.prebuilt import create_react_agent

# 2026 最新 API：支持更多配置选项
react_agent = create_react_agent(
    model="gpt-5",  # 直接传入模型名称或 ChatOpenAI 实例
    tools=[search_web, calculate, get_weather],
    prompt=ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手，可以使用工具帮助用户。"),
        MessagesPlaceholder("messages")
    ]),
    # 新增配置选项
    version="v2",  # 使用最新版本的行为
    response_format="json",  # 可选：json, text
    interrupt_on=[],  # 需要中断的节点列表
    debug=False  # 调试模式
)

# 执行
result = react_agent.invoke({
    "messages": [HumanMessage(content="北京今天天气如何？")]
})

# 2026 新增：流式工具调用事件
async for event in react_agent.astream_events(
    {"messages": [HumanMessage(content="查询天气并计算")]},
    version="v2"
):
    if event["event"] == "on_tool_start":
        print(f"🔧 调用工具: {event['name']}")
    elif event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
```

**2. 原生人类审核节点（Human-in-the-Loop v2）**

```python
from langgraph.types import interrupt, Command

# 2026 改进：支持结构化中断请求
def review_node(state: AgentState) -> Command:
    """需要人类审核的节点"""
    
    # 触发中断，等待人类输入
    human_feedback = interrupt({
        "question": "请审核以下内容：",
        "content": state["messages"][-1].content,
        "options": ["approve", "reject", "modify"]  # 可选操作
    })
    
    # 使用 Command 明确指定下一步
    if human_feedback == "approve":
        return Command(goto="synthesize")
    elif human_feedback == "reject":
        return Command(goto="research")  # 返回重新研究
    else:
        return Command(
            goto="synthesize",
            update={"messages": [HumanMessage(content=f"修改意见：{human_feedback}")]}
        )
```

**3. 子图通信与嵌套状态**

```python
from langgraph.types import Send

class ParentState(TypedDict):
    messages: Annotated[List, add_messages]
    subgraph_results: dict
    parallel_tasks: list[str]

def parallel_execution_node(state: ParentState):
    """2026 新增：动态并行子图"""
    tasks = state.get("parallel_tasks", [])
    
    # 并行启动多个子图实例
    return [
        Send("child_graph", {"task": task, "parent_messages": state["messages"]})
        for task in tasks
    ]

def merge_results_node(state: ParentState) -> ParentState:
    """合并并行子图的结果"""
    # LangGraph v0.2+ 自动处理 Send 的结果聚合
    return {"subgraph_results": state.get("child_results", [])}
```

**4. 流式输出优化**

```python
# 流式执行图 - 2026 最新 API
async for event in agent.astream_events(
    {"messages": [HumanMessage(content="执行任务")]},
    version="v2",
    stream_mode="values"  # 新增：控制流式模式
):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
    elif event["event"] == "on_tool_start":
        print(f"\n🔧 调用工具: {event['name']}")
    elif event["event"] == "on_tool_end":
        print(f"✅ 工具完成")
```

**5. 改进的子图通信**

```python
class ParentState(TypedDict):
    messages: Annotated[List, add_messages]
    subgraph_results: dict

def parent_node(state: ParentState) -> ParentState:
    """调用子图并获取结果"""
    subgraph = child_graph.compile()
    
    result = subgraph.invoke({
        "messages": state["messages"]
    })
    
    return {"subgraph_results": result}
```

### 4.3 2026 年 LangChain/LangGraph 最佳实践

**1. 使用 gpt-5/claude-4 的结构化输出能力**

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# 2026 最佳实践：充分利用新一代模型的结构化输出
class TaskPlan(BaseModel):
    steps: list[str] = Field(description="任务步骤列表")
    required_tools: list[str] = Field(description="需要的工具")
    estimated_complexity: str = Field(description="复杂度评估: low/medium/high")
    risks: list[str] = Field(description="潜在风险")

llm = ChatOpenAI(model="gpt-5")
planner = llm.with_structured_output(TaskPlan)

plan = planner.invoke("帮我分析并优化这段 Python 代码")
print(f"步骤: {plan.steps}")
print(f"需要工具: {plan.required_tools}")
```

**2. LangGraph 状态机设计模式**

```python
from enum import Enum
from typing import Literal

# 2026 推荐：使用枚举定义状态
class AgentPhase(str, Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    phase: AgentPhase
    retry_count: int
    max_retries: int = 3

def route_by_phase(state: AgentState) -> Literal["plan", "execute", "review", "end", "retry"]:
    """基于状态的智能路由"""
    if state["phase"] == AgentPhase.PLANNING:
        return "plan"
    elif state["phase"] == AgentPhase.EXECUTING:
        return "execute"
    elif state["phase"] == AgentPhase.REVIEWING:
        if has_errors(state):
            if state["retry_count"] < state["max_retries"]:
                return "retry"
            return "end"  # 超过重试次数
        return "end"
    return "end"
```

**3. 工具调用的错误恢复策略**

```python
from langgraph.types import RetryPolicy
import asyncio

# 2026 最佳实践：为不同工具配置不同的重试策略
workflow.add_node(
    "database_query",
    db_query_node,
    retry=RetryPolicy(
        max_attempts=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        jitter=True,  # 添加随机抖动避免雷群效应
        retry_on=[ConnectionError, TimeoutError]
    )
)

workflow.add_node(
    "llm_generation",
    llm_node,
    retry=RetryPolicy(
        max_attempts=2,  # LLM 调用重试次数较少
        initial_delay=2.0,
        retry_on=[RateLimitError]  # 只重试速率限制错误
    )
)
```

**4. 可观测性工程实践**

```python
import os
from langsmith import Client, traceable
from opentelemetry import trace, metrics

# 2026 推荐：LangSmith + OpenTelemetry 双栈监控
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "production-agent"

# OpenTelemetry 指标收集
tracer = trace.get_tracer("langgraph-agent")
meter = metrics.get_meter("langgraph-agent")

node_duration_histogram = meter.create_histogram(
    "agent.node.execution.duration",
    description="节点执行时间"
)

@traceable(run_type="chain")
def monitored_node(state: AgentState) -> AgentState:
    with tracer.start_as_current_span("node_execution") as span:
        span.set_attribute("node.type", "research")
        span.set_attribute("state.phase", state.get("phase", "unknown"))
        
        start_time = time.time()
        try:
            result = actual_processing(state)
            span.set_status(StatusCode.OK)
            return result
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise
        finally:
            duration = time.time() - start_time
            node_duration_histogram.record(duration, {"node": "research"})
```

**5. 生产级检查点配置**

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.redis import RedisSaver
import asyncpg

# 2026 推荐：使用高性能持久化检查点
async def create_production_checkpointer():
    """PostgreSQL 检查点（推荐用于生产）"""
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="agent_user",
        password="secure_password",
        database="agent_db"
    )
    
    return PostgresSaver(conn)

# 或使用 Redis（适合高并发场景）
redis_saver = RedisSaver(
    host="localhost",
    port=6379,
    db=0,
    password="redis_password",
    ttl=86400  # 24 小时自动清理
)

# 编译时配置
agent = workflow.compile(
    checkpointer=await create_production_checkpointer(),
    # 2026 新增：配置中断策略
    interrupt_before=["review"],  # 在审核节点前中断
    interrupt_after=["generate"]   # 在生成节点后中断
)
```

### 4.4 性能优化最佳实践

**1. 并行化独立任务**

```python
from langgraph.types import Send

def parallel_research(state: AgentState):
    """并行执行多个研究任务"""
    topics = ["topic1", "topic2", "topic3"]
    
    return [
        Send("research_node", {"topic": topic})
        for topic in topics
    ]

workflow.add_node("parallel_research", parallel_research)
```

**2. 缓存与去重**

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embedding(text: str) -> List[float]:
    """缓存 embedding 结果"""
    return embedding_model.encode(text)

# 或者在图中使用缓存节点
def cached_search(state: AgentState) -> AgentState:
    cache_key = hash(state["messages"][-1].content)
    
    if cache_key in cache:
        return cache[cache_key]
    
    result = perform_search(state)
    cache[cache_key] = result
    return result
```

**3. 超时与重试**

```python
from langgraph.types import RetryPolicy

workflow.add_node(
    "external_api_call",
    api_call_node,
    retry=RetryPolicy(
        max_attempts=3,
        backoff_factor=2,
        retry_on=[TimeoutError, ConnectionError]
    )
)
```

### 4.5 调试与可观测性

**1. LangSmith 集成**

```python
import os
from langsmith import Client, traceable

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-agent-project"

@traceable
def my_agent_chain(input: str) -> str:
    return agent.invoke({"messages": [HumanMessage(content=input)]})
```

**2. 本地调试**

```python
from langgraph.types import Command

# 逐步执行
snapshot = agent.invoke(
    {"messages": [HumanMessage(content="test")]},
    {"configurable": {"thread_id": "debug_001"}}
)

# 查看状态
print(agent.get_state({"configurable": {"thread_id": "debug_001"}}))

# 手动修改状态后继续
agent.update_state(
    {"configurable": {"thread_id": "debug_001"}},
    {"messages": [HumanMessage(content="修改后的输入")]}
)
```

### 4.6 生产部署建议

**1. 异步化**

```python
import asyncio
from langgraph.graph import StateGraph

async def async_research_node(state: AgentState) -> AgentState:
    """异步研究节点"""
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com/search") as response:
            data = await response.json()
    
    return {"messages": [AIMessage(content=str(data))]}

# 异步执行
async for event in agent.astream(initial_state):
    print(event)
```

**2. 错误处理**

```python
from langgraph.errors import GraphInterrupt, NodeInterrupt

def robust_node(state: AgentState) -> AgentState:
    try:
        result = risky_operation(state)
        return {"messages": [AIMessage(content=result)]}
    except Exception as e:
        # 记录错误并返回友好消息
        return {
            "messages": [AIMessage(content=f"处理出错: {str(e)}")],
            "error": str(e)
        }
```

**3. 监控指标**

```python
from prometheus_client import Counter, Histogram

node_execution_counter = Counter(
    'agent_node_executions_total',
    'Total node executions',
    ['node_name']
)

node_latency_histogram = Histogram(
    'agent_node_latency_seconds',
    'Node execution latency',
    ['node_name']
)

def monitored_node(state: AgentState) -> AgentState:
    start_time = time.time()
    
    result = actual_node(state)
    
    duration = time.time() - start_time
    node_execution_counter.inc()
    node_latency_histogram.observe(duration)
    
    return result
```

## 5. 实战案例：智能代码审查 Agent

### 5.1 完整实现

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

class CodeReviewState(TypedDict):
    messages: Annotated[List, add_messages]
    code: str
    issues: List[dict]
    suggestions: List[str]
    review_completed: bool

# 节点实现
def parse_code_node(state: CodeReviewState) -> CodeReviewState:
    """解析代码，提取关键信息"""
    code = state["messages"][-1].content
    
    # 识别语言、框架等
    llm = ChatOpenAI(model="gpt-5")
    response = llm.invoke([
        HumanMessage(content=f"""分析以下代码，提取：
1. 编程语言
2. 使用的框架/库
3. 主要功能
4. 潜在风险点

代码：
{code}

以 JSON 格式返回。""")
    ])
    
    return {
        "code": code,
        "messages": [AIMessage(content=response.content)]
    }

def static_analysis_node(state: CodeReviewState) -> CodeReviewState:
    """静态分析：检查代码规范、潜在 bug"""
    llm = ChatOpenAI(model="gpt-5")
    
    response = llm.invoke([
        HumanMessage(content=f"""作为静态分析工具，检查以下代码的问题：

- 语法错误
- 潜在 bug
- 安全漏洞
- 性能问题
- 代码规范违规

代码：
{state["code"]}

列出所有问题，按严重程度排序。""")
    ])
    
    return {
        "issues": [{"type": "static", "description": response.content}],
        "messages": [AIMessage(content=response.content)]
    }

def best_practices_node(state: CodeReviewState) -> CodeReviewState:
    """最佳实践检查"""
    llm = ChatOpenAI(model="gpt-5")
    
    response = llm.invoke([
        HumanMessage(content=f"""检查以下代码是否符合最佳实践：

- 设计模式使用
- SOLID 原则
- DRY 原则
- 错误处理
- 日志记录
- 测试覆盖建议

代码：
{state["code"]}

提供改进建议。""")
    ])
    
    return {
        "suggestions": [response.content],
        "messages": [AIMessage(content=response.content)]
    }

def generate_report_node(state: CodeReviewState) -> CodeReviewState:
    """生成审查报告"""
    llm = ChatOpenAI(model="gpt-5")
    
    response = llm.invoke([
        HumanMessage(content=f"""基于以下分析结果，生成结构化的代码审查报告：

问题：
{state["issues"]}

建议：
{state["suggestions"]}

报告格式：
1. 总体评价
2. 关键问题（必须修复）
3. 改进建议（建议修复）
4. 优秀实践（值得保留）
5. 评分（1-10）""")
    ])
    
    return {
        "messages": [AIMessage(content=response.content)],
        "review_completed": True
    }

# 构建图
review_graph = StateGraph(CodeReviewState)

review_graph.add_node("parse", parse_code_node)
review_graph.add_node("static_analysis", static_analysis_node)
review_graph.add_node("best_practices", best_practices_node)
review_graph.add_node("report", generate_report_node)

review_graph.set_entry_point("parse")
review_graph.add_edge("parse", "static_analysis")
review_graph.add_edge("static_analysis", "best_practices")
review_graph.add_edge("best_practices", "report")
review_graph.add_edge("report", END)

# 编译
code_reviewer = review_graph.compile()

# 使用
result = code_reviewer.invoke({
    "messages": [HumanMessage(content="""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""")]
})

print(result["messages"][-1].content)
```

### 5.2 添加人类审核

```python
from langgraph.types import interrupt

def human_review_node(state: CodeReviewState) -> CodeReviewState:
    """人类审核 AI 的审查结果"""
    
    # 中断等待人类反馈
    feedback = interrupt({
        "question": "请审核 AI 的代码审查结果",
        "ai_report": state["messages"][-1].content,
        "action": "approve/modify/reject"
    })
    
    return {
        "messages": [HumanMessage(content=f"人类审核意见：{feedback}")],
        "review_completed": feedback == "approve"
    }

# 在报告中添加人类审核
review_graph.add_node("human_review", human_review_node)
review_graph.add_edge("report", "human_review")
review_graph.add_edge("human_review", END)
```

## 6. 参考资源

### 官方文档（v1.0）
- [LangChain 统一文档 (2026)](https://docs.langchain.com/) - Python 和 JavaScript 统一文档
- [LangChain v1.0 发布说明](https://docs.langchain.com/oss/python/releases/langchain-v1)
- [LangGraph v1.0 文档](https://docs.langchain.com/oss/python/langgraph)
- [LangChain 博客 - 最新公告](https://www.langchain.com/blog)

### 核心教程
- [LangChain and LangGraph v1.0 里程碑公告](https://www.langchain.com/blog/langchain-langgraph-1dot0)
- [LangChain v1.0 迁移指南](https://docs.langchain.com/oss/python/migrate/langchain-v1)
- [Building LangGraph: Designing an Agent Runtime](https://www.langchain.com/blog/building-langgraph)
- [LangGraph v1.0 API 教程](https://dev.to/agentsindex/langgraph-tutorial-build-a-working-react-agent-with-the-v10-api-3bc1)

### 中间件与 Agent 工程
- [LangChain 中间件完整指南](https://docs.langchain.com/oss/python/langchain/middleware)
- [create_agent API 参考](https://reference.langchain.com/python/langchain/agents/factory/create_agent)
- [Context Engineering with Middleware](https://docs.langchain.com/oss/python/langchain/context-engineering)

### 实战指南
- [Building AI Agents with LangGraph (2026 Edition)](https://ai.gopubby.com/building-ai-agents-with-langgraph-2026-edition-a-step-by-step-guide-494d36e801f9)
- [LangGraph 2026 年更新与迁移指南](https://www.agentframeworkhub.com/blog/langgraph-news-updates-2026)
- [Production-Ready Agent Patterns](https://docs.langchain.com/oss/python/langgraph/tutorials/production-patterns/)
- [Human-in-the-Loop 最佳实践](https://docs.langchain.com/oss/python/langgraph/how-tos/human-in-the-loop/)

### 开源项目与工具
- [LangGraph GitHub 仓库](https://github.com/langchain-ai/langgraph)
- [LangChain GitHub 仓库](https://github.com/langchain-ai/langchain)
- [LangSmith 可观测性平台](https://smith.langchain.com/)
- [LangChain Changelog](https://changelog.langchain.com/)

### 模型 API 参考
- [OpenAI GPT-5 API 文档](https://platform.openai.com/docs/models/gpt-5)
- [Anthropic Claude 4 API 文档](https://docs.anthropic.com/claude/docs/claude-4)

### 框架选择指南

| 场景 | 推荐框架 | 理由 |
|------|---------|------|
| 快速构建标准 Agent | LangChain `create_agent` | 3行代码，内置中间件 |
| 需要定制 Agent 循环 | LangChain + 中间件 | 6个钩子点，灵活定制 |
| 复杂工作流编排 | LangGraph StateGraph | 图结构，条件分支，循环 |
| 混合确定性+Agent | LangGraph | 完美混合节点类型 |
| 长期运行业务流程 | LangGraph | 持久化状态，检查点 |
| 高敏感需要人工审核 | LangGraph | 原生 Human-in-the-Loop |

**最佳实践：从 LangChain 开始，按需降级到 LangGraph**
- 先用 `create_agent` 快速原型
- 需要定制时添加中间件
- 复杂工作流时使用 LangGraph 图结构
- 两者可以混合使用（create_agent 可以作为 LangGraph 的节点）
