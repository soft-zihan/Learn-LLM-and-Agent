# LangGraph 核心概念与实战

> 📅 **更新时间**: 2026-06-29

---

## 目录

- [1. LangGraph 是什么](#1-langgraph-是什么)
- [2. 核心概念：State、Nodes、Edges](#2-核心概念statenodesedges)
- [3. 构建你的第一个图](#3-构建你的第一个图)
- [4. 条件边与路由](#4-条件边与路由)
- [5. 工具调用 Agent](#5-工具调用-agent)
- [6. 流式输出](#6-流式输出)
- [7. 检查点与持久化](#7-检查点与持久化)
- [8. 人类介入（Human-in-the-Loop）](#8-人类介入human-in-the-loop)
- [9. 子图与嵌套图](#9-子图与嵌套图)
- [10. 并行执行与异步](#10-并行执行与异步)
- [11. 错误处理与重试](#11-错误处理与重试)
- [12. 生产级最佳实践](#12-生产级最佳实践)

---

## 1. LangGraph 是什么

### 1.1 为什么需要 LangGraph

传统 LangChain 的局限性：
- **线性流程**：只能按固定顺序执行
- **无状态**：无法在步骤间维护复杂状态
- **难以循环**：不支持条件循环
- **无法分支**：没有条件路由能力

LangGraph 解决这些问题：
- **图结构**：节点、边、条件边构成有向图
- **状态管理**：通过 State 在节点间传递和更新数据
- **循环支持**：原生支持条件循环（ReAct、Reflection 等模式）
- **分支合并**：条件边实现动态路由

### 1.2 LangGraph vs LangChain

```python
# LangChain：线性链
chain = prompt | llm | parser  # 只能线性执行

# LangGraph：图结构
workflow = StateGraph(State)
workflow.add_node("research", research_node)
workflow.add_node("analyze", analyze_node)
workflow.add_conditional_edges("analyze", route_function)  # 可以循环、分支
```

### 1.3 安装

```bash
pip install -U langgraph langchain-openai
```

---

## 2. 核心概念：State、Nodes、Edges

### 2.1 State（状态）

State 是图执行过程中维护和传递的数据结构：

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import operator

class AgentState(TypedDict):
    """
    Agent 状态定义
    
    每个字段的行为由注解（annotation）决定：
    - add_messages: 追加消息（不覆盖）
    - operator.add: 追加列表
    - 默认：新值覆盖旧值
    """
    # 消息历史（自动追加）
    messages: Annotated[list, add_messages]
    
    # 列表追加模式
    findings: Annotated[list, operator.add]
    
    # 覆盖模式（默认）
    current_step: str
    result: str
    
    # 计数器
    iteration_count: int
```

**Reducer 详解**：

```python
# 自定义 Reducer
def merge_dicts(old: dict, new: dict) -> dict:
    """合并字典（而不是覆盖）"""
    merged = old.copy()
    merged.update(new)
    return merged

class CustomState(TypedDict):
    metadata: Annotated[dict, merge_dicts]
```

### 2.2 Nodes（节点）

节点是普通的 Python 函数，接收当前状态，返回状态更新：

```python
def research_node(state: AgentState) -> dict:
    """
    研究节点
    
    参数：
    - state: 当前完整状态
    
    返回：
    - dict: 只包含需要更新的字段
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    
    llm = ChatOpenAI(model="gpt-4o")
    
    # 获取用户输入
    user_query = state["messages"][-1].content
    
    # 执行研究
    response = llm.invoke([
        HumanMessage(content=f"研究以下问题：{user_query}")
    ])
    
    # 返回状态更新（只返回需要更新的字段）
    return {
        "messages": [response],
        "findings": [f"研究完成：{response.content[:100]}..."],
        "current_step": "research_completed",
        "iteration_count": state.get("iteration_count", 0) + 1
    }
```

**节点类型**：

```python
# 1. 普通函数节点
def simple_node(state: AgentState) -> dict:
    return {"result": "done"}

# 2. 异步节点
async def async_node(state: AgentState) -> dict:
    result = await some_async_operation()
    return {"result": result}

# 3. LLM 调用节点
def llm_node(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 4. 工具调用节点
from langgraph.prebuilt import ToolNode

tools = [search_tool, calculate_tool]
tool_node = ToolNode(tools)
```

### 2.3 Edges（边）

边定义节点之间的连接：

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("research", research_node)
workflow.add_node("analyze", analyze_node)

# 添加边
workflow.add_edge(START, "research")  # 从 START 开始
workflow.add_edge("research", "analyze")
workflow.add_edge("analyze", END)  # 到 END 结束

# 编译图
agent = workflow.compile()
```

---

## 3. 构建你的第一个图

### 3.1 完整示例：研究 Agent

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# 1. 定义状态
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    research_result: str
    analysis_result: str
    final_answer: str

# 2. 创建 LLM
llm = ChatOpenAI(model="gpt-4o")

# 3. 定义节点
def research_node(state: ResearchState) -> dict:
    """研究节点"""
    print("🔍 开始研究...")
    
    user_query = state["messages"][-1].content
    
    response = llm.invoke([
        HumanMessage(content=f"深入研究：{user_query}")
    ])
    
    return {
        "messages": [response],
        "research_result": response.content
    }

def analyze_node(state: ResearchState) -> dict:
    """分析节点"""
    print("📊 分析结果...")
    
    research_result = state["research_result"]
    
    response = llm.invoke([
        HumanMessage(content=f"分析以下研究结果并总结关键点：\n{research_result}")
    ])
    
    return {
        "messages": [response],
        "analysis_result": response.content
    }

def synthesize_node(state: ResearchState) -> dict:
    """综合节点"""
    print("✍️ 生成最终答案...")
    
    analysis = state["analysis_result"]
    
    response = llm.invoke([
        HumanMessage(content=f"基于分析结果，生成综合答案：\n{analysis}")
    ])
    
    return {
        "messages": [response],
        "final_answer": response.content
    }

# 4. 构建图
workflow = StateGraph(ResearchState)

# 添加节点
workflow.add_node("research", research_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("synthesize", synthesize_node)

# 添加边
workflow.add_edge(START, "research")
workflow.add_edge("research", "analyze")
workflow.add_edge("analyze", "synthesize")
workflow.add_edge("synthesize", END)

# 编译图
agent = workflow.compile()

# 5. 可视化图结构
print(agent.get_graph().draw_mermaid())
# 输出 Mermaid 图，可在 Markdown 中渲染

# 6. 执行
result = agent.invoke({
    "messages": [HumanMessage(content="AI Agent 的未来发展趋势是什么？")]
})

print("\n✅ 最终答案：")
print(result["final_answer"])
```

### 3.2 执行流程追踪

```python
# 逐步执行，查看每个节点的输出
for event in agent.stream({
    "messages": [HumanMessage(content="研究量子计算")]
}):
    print(f"事件：{event}")
    # 输出每个节点的状态更新
```

---

## 4. 条件边与路由

### 4.1 条件边基础

```python
def should_continue(state: ResearchState) -> str:
    """决定下一步"""
    iteration = state.get("iteration_count", 0)
    
    if iteration < 3:
        return "continue"  # 继续循环
    return "end"  # 结束

# 添加条件边
workflow.add_conditional_edges(
    "analyze",  # 从哪个节点
    should_continue,  # 路由函数
    {
        "continue": "research",  # 如果返回 "continue"，去 research 节点
        "end": "synthesize"  # 如果返回 "end"，去 synthesize 节点
    }
)
```

### 4.2 使用 Literal 类型安全路由

```python
from typing import Literal

def route_by_quality(state: ResearchState) -> Literal["improve", "finalize"]:
    """根据答案质量路由"""
    answer = state.get("final_answer", "")
    
    if len(answer) < 100 or "不确定" in answer:
        return "improve"
    return "finalize"

workflow.add_conditional_edges(
    "synthesize",
    route_by_quality,
    {
        "improve": "research",
        "finalize": END
    }
)
```

### 4.3 LLM 路由

```python
def llm_router(state: ResearchState) -> str:
    """使用 LLM 决定路由"""
    router_prompt = f"""
    根据当前进度，决定下一步：
    
    当前状态：{state.get('current_step')}
    迭代次数：{state.get('iteration_count', 0)}
    
    可选：
    - research：需要更多研究
    - analyze：需要分析
    - finalize：完成
    
    只返回节点名称。
    """
    
    response = llm.invoke([HumanMessage(content=router_prompt)])
    return response.content.strip()

workflow.add_conditional_edges(
    "analyze",
    llm_router,
    {
        "research": "research",
        "analyze": "analyze",
        "finalize": "synthesize"
    }
)
```

---

## 5. 工具调用 Agent

### 5.1 使用 create_agent（推荐）

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """搜索网络获取信息"""
    return f"搜索结果：{query}"

@tool
def calculate(expression: str) -> str:
    """执行计算"""
    return str(eval(expression))

# 创建 Agent（底层使用 LangGraph）
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_web, calculate],
    system_prompt="你是有用的助手，可以使用工具帮助用户"
)

# 执行
result = agent.invoke({
    "messages": [HumanMessage(content="北京今天天气？如果温度>30，计算30-25")]
})

print(result["messages"][-1]["content"])
```

### 5.2 手动构建 ReAct 图

```python
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage

def call_model(state: dict) -> dict:
    """调用 LLM"""
    llm_with_tools = llm.bind_tools([search_web, calculate])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: dict) -> str:
    """判断是否继续调用工具"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"  # 有工具调用，去 tools 节点
    return END  # 没有工具调用，结束

# 构建 ReAct 图
workflow = StateGraph({"messages": list})

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode([search_web, calculate]))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)
workflow.add_edge("tools", "agent")  # 工具执行后回到 agent

agent = workflow.compile()
```

---

## 6. 流式输出

### 6.1 基础流式

```python
# 流式执行
for event in agent.stream({
    "messages": [HumanMessage(content="研究 AI 趋势")]
}):
    print(event)
    # 每个节点完成后输出结果
```

### 6.2 流式 Token

```python
# 流式输出 LLM token
async for event in agent.astream_events({
    "messages": [HumanMessage(content="写一首诗")]
}, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
    elif event["event"] == "on_tool_start":
        print(f"\n🔧 调用工具：{event['name']}")
    elif event["event"] == "on_tool_end":
        print(f"\n✅ 工具完成")
```

### 6.3 流式与状态

```python
# 查看每个节点的状态更新
for event in agent.stream(
    {"messages": [HumanMessage(content="研究任务")]},
    stream_mode="updates"  # 只输出状态更新
):
    print(f"节点更新：{event}")
```

---

## 7. 检查点与持久化

### 7.1 MemorySaver（开发用）

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# 编译时指定检查点
agent = workflow.compile(checkpointer=memory)

# 执行（使用 thread_id 区分会话）
result1 = agent.invoke(
    {"messages": [HumanMessage(content="你好")]},
    config={"configurable": {"thread_id": "session_1"}}
)

# 同一会话，记住历史
result2 = agent.invoke(
    {"messages": [HumanMessage(content="我刚才说了什么？")]},
    config={"configurable": {"thread_id": "session_1"}}
)

# 不同会话，记忆隔离
result3 = agent.invoke(
    {"messages": [HumanMessage(content="你好")]},
    config={"configurable": {"thread_id": "session_2"}}
)
```

### 7.2 SqliteSaver（生产用）

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
sqlite_memory = SqliteSaver(conn)

agent = workflow.compile(checkpointer=sqlite_memory)

# 执行
result = agent.invoke(
    {"messages": [HumanMessage(content="记住我喜欢 Python")]},
    config={"configurable": {"thread_id": "user_123"}}
)

# 重启后恢复
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
sqlite_memory = SqliteSaver(conn)
agent = workflow.compile(checkpointer=sqlite_memory)

result = agent.invoke(
    {"messages": [HumanMessage(content="我喜欢什么？")]},
    config={"configurable": {"thread_id": "user_123"}}
)
```

### 7.3 查看和修改状态

```python
# 查看当前状态
thread_config = {"configurable": {"thread_id": "session_1"}}
snapshot = agent.get_state(thread_config)

print(f"当前节点：{snapshot.metadata.get('step')}")
print(f"消息历史：{snapshot.values.get('messages')}")

# 修改状态后继续
agent.update_state(
    thread_config,
    {"messages": [HumanMessage(content="修改后的输入")]}
)

# 继续执行
result = agent.invoke(None, config=thread_config)
```

---

## 8. 人类介入（Human-in-the-Loop）

### 8.1 interrupt() 函数

```python
from langgraph.types import interrupt

def review_node(state: dict) -> dict:
    """人类审核节点"""
    # 中断执行，等待人类输入
    feedback = interrupt({
        "question": "请审核以下草稿：",
        "draft": state.get("draft"),
        "options": ["approve", "revise", "reject"]
    })
    
    return {"human_feedback": feedback}

# 必须使用检查点才能使用 interrupt
memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)

# 第一次执行（会中断）
thread_config = {"configurable": {"thread_id": "review_001"}}

try:
    result = agent.invoke(
        {"messages": [HumanMessage(content="写一份报告")]},
        config=thread_config
    )
except Exception as e:
    print(f"中断，等待审核：{e}")

# 人类审核后恢复
result = agent.invoke(
    {"human_feedback": "approve"},
    config=thread_config
)
```

### 8.2 中断前修改状态

```python
# 在中断处查看状态
snapshot = agent.get_state(thread_config)
print(f"草稿：{snapshot.values.get('draft')}")

# 修改草稿
agent.update_state(
    thread_config,
    {"draft": "修改后的内容"}
)

# 恢复执行
result = agent.invoke(None, config=thread_config)
```

---

## 9. 子图与嵌套图

### 9.1 子图作为节点

```python
# 子图：搜索智能体
search_workflow = StateGraph({"messages": list, "results": list})
search_workflow.add_node("search", search_node)
search_workflow.add_node("filter", filter_node)
search_workflow.add_edge(START, "search")
search_workflow.add_edge("search", "filter")
search_workflow.add_edge("filter", END)

search_agent = search_workflow.compile()

# 主图：使用子图作为节点
main_workflow = StateGraph({"messages": list})
main_workflow.add_node("search_agent", search_agent)  # 子图作为节点
main_workflow.add_node("analyze", analyze_node)

main_workflow.add_edge(START, "search_agent")
main_workflow.add_edge("search_agent", "analyze")
main_workflow.add_edge("analyze", END)

agent = main_workflow.compile()
```

### 9.2 动态子图

```python
from langgraph.types import Send

def parallel_execution(state: dict) -> list:
    """并行启动多个子图"""
    tasks = ["task1", "task2", "task3"]
    
    return [
        Send("subagent", {"task": task})
        for task in tasks
    ]

workflow.add_node("parallel", parallel_execution)
workflow.add_conditional_edges("parallel", ...)
```

---

## 10. 并行执行与异步

### 10.1 异步节点

```python
async def async_search(state: dict) -> dict:
    """异步搜索节点"""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com/search") as response:
            data = await response.json()
    
    return {"messages": [AIMessage(content=str(data))]}

# 异步执行
result = await agent.ainvoke({
    "messages": [HumanMessage(content="异步搜索")]
})
```

### 10.2 并行流

```python
# 并行执行多个独立任务
async for event in agent.astream({
    "messages": [HumanMessage(content="并行任务")]
}):
    print(event)
```

---

## 11. 错误处理与重试

### 11.1 节点内重试

```python
def robust_node(state: dict) -> dict:
    """带重试的节点"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            result = risky_operation()
            return {"messages": [AIMessage(content=result)]}
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)  # 指数退避
                print(f"重试 {attempt + 1}/{max_retries}")
            else:
                return {
                    "messages": [AIMessage(content=f"失败：{str(e)}")]
                }
```

### 11.2 全局错误处理

```python
def error_handler(state: dict) -> dict:
    """错误处理节点"""
    error = state.get("error")
    
    return {
        "messages": [AIMessage(content=f"处理出错：{error}，已回退到默认方案")]
    }

# 在图中添加错误处理
workflow.add_node("error_handler", error_handler)
workflow.add_edge("error_handler", END)
```

---

## 12. 生产级最佳实践

### 12.1 状态设计

```python
# ✅ 推荐：最小状态
class MinimalState(TypedDict):
    messages: Annotated[list, add_messages]
    result: str

# ❌ 不推荐：冗余状态
class BadState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str  # 冗余
    llm_response: str  # 冗余
```

### 12.2 节点拆分

```python
# ✅ 推荐：按职责拆分
def search_node(state): ...
def filter_node(state): ...
def analyze_node(state): ...

# ❌ 不推荐：一个节点做太多事
def complex_node(state):
    # 搜索 + 过滤 + 分析 + 报告
    ...
```

### 12.3 可视化调试

```python
# 生成 Mermaid 图
print(agent.get_graph().draw_mermaid())

# 在 Markdown 中渲染
# ```mermaid
# graph TD
#     START --> research
#     research --> analyze
#     analyze --> synthesize
#     synthesize --> END
# ```
```

### 12.4 成本控制

```python
# 限制最大迭代次数
def should_continue(state: dict) -> str:
    if state.get("iteration_count", 0) >= 5:
        return "end"  # 防止无限循环
    return "continue"
```

---

## 参考资源

- [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [Building AI Agents with LangGraph (2026 Edition)](https://ai.gopubby.com/building-ai-agents-with-langgraph-2026-edition-a-step-by-step-guide-494d36e801f9)
