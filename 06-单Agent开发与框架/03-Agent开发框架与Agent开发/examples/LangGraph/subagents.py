"""
子智能体与路由模式 - 对应教程 09-子智能体与路由模式

本模块演示了多 Agent 协作的核心模式：

1. 字典式子智能体（教程09.1）
   - 多个专 Agent（搜索、数据库、知识库、代码）
   - 路由器根据任务类型选择对应的子 Agent
   - 图结构：START → router → (条件路由) → subagent → END

2. 并行子智能体（教程09.2）
   - 同时启动多个子 Agent 并行执行
   - 最后汇总所有结果
   - 图结构：START → [search, db, knowledge] → aggregate → END

3. 主智能体调度（教程09.3）
   - 主 Agent 作为调度者，决定调用哪些子 Agent
   - 子 Agent 作为子图嵌入主图
"""

# 从公共配置导入 LLM（避免硬编码 API 密钥）
from config import llm

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import asyncio

# ── 子智能体的工具 ──────────────────────────────────────────

@tool
def search_web(query: str) -> str:
    """搜索网络获取最新信息。"""
    return f"网络搜索结果：{query}"

@tool
def query_database(sql: str) -> str:
    """查询数据库。"""
    return f"数据库结果：{sql}"

@tool
def search_knowledge(query: str) -> str:
    """搜索知识库。"""
    return f"知识库结果：{query}"

@tool
def analyze_code(code: str) -> str:
    """分析代码。"""
    return f"代码分析结果：{code[:100]}..."

# ── 创建子智能体（教程09：使用 LangGraph prebuilt 创建）────────
# 每个子 Agent 是一个小型 ReAct Agent，只拥有特定工具
# 使用 langgraph.prebuilt.create_react_agent 简化创建

from langgraph.prebuilt import create_react_agent

search_agent = create_react_agent(
    llm.bind_tools([search_web]),
    [search_web],
)

db_agent = create_react_agent(
    llm.bind_tools([query_database]),
    [query_database],
)

knowledge_agent = create_react_agent(
    llm.bind_tools([search_knowledge]),
    [search_knowledge],
)

code_agent = create_react_agent(
    llm.bind_tools([analyze_code]),
    [analyze_code],
)

# ── 字典式子智能体（教程09.1）───────────────────────────────

# 注册子智能体字典
subagents = {
    "search": search_agent,
    "database": db_agent,
    "knowledge": knowledge_agent,
    "code": code_agent,
}

# 子智能体描述（用于路由决策）
subagent_descriptions = {
    "search": "搜索网络获取最新信息、新闻、文档等",
    "database": "查询数据库、执行SQL、获取数据",
    "knowledge": "搜索内部知识库、文档、FAQ",
    "code": "分析代码、审查代码、调试",
}

class SubAgentState(MessagesState):
    """子智能体状态"""
    subagent_name: Optional[str] = None
    subagent_result: Optional[str] = None


def route_to_subagent(state: SubAgentState) -> str:
    """路由函数：根据任务类型选择子智能体"""
    # 这里可以接入LLM进行智能路由
    last_message = state["messages"][-1].content.lower()
    
    if any(word in last_message for word in ["搜索", "查询", "新闻", "最新"]):
        return "search"
    elif any(word in last_message for word in ["数据库", "sql", "数据"]):
        return "database"
    elif any(word in last_message for word in ["知识", "文档", "faq"]):
        return "knowledge"
    elif any(word in last_message for word in ["代码", "程序", "分析"]):
        return "code"
    
    return "search"  # 默认


async def run_subagent(state: SubAgentState):
    """运行选中的子智能体"""
    subagent_name = state.get("subagent_name")
    if subagent_name not in subagents:
        return {"subagent_result": f"未知子智能体: {subagent_name}"}
    
    agent = subagents[subagent_name]
    result = await agent.ainvoke({"messages": state["messages"]})
    
    return {"subagent_result": result["messages"][-1].content}


def build_subagent_graph():
    """构建子智能体图（教程09核心）"""
    graph = StateGraph(SubAgentState)
    
    # 为每个子智能体添加节点
    for name in subagents:
        graph.add_node(name, run_subagent)
    
    # 路由节点
    def router_node(state: SubAgentState):
        return {"subagent_name": route_to_subagent(state)}
    
    graph.add_node("router", router_node)
    graph.add_edge(START, "router")
    
    # 条件边：从路由器到各子智能体
    graph.add_conditional_edges(
        "router",
        lambda state: state.get("subagent_name", "search"),
        {name: name for name in subagents}
    )
    
    # 所有子智能体都指向 END
    for name in subagents:
        graph.add_edge(name, END)
    
    return graph.compile()


# ── 并行子智能体（教程09.2）─────────────────────────────────

class ParallelState(TypedDict):
    """并行子智能体状态"""
    messages: list
    search_result: Optional[str] = None
    db_result: Optional[str] = None
    knowledge_result: Optional[str] = None


async def run_search(state: ParallelState):
    """运行搜索子智能体"""
    result = await search_agent.ainvoke({"messages": state["messages"]})
    return {"search_result": result["messages"][-1].content}


async def run_db(state: ParallelState):
    """运行数据库子智能体"""
    result = await db_agent.ainvoke({"messages": state["messages"]})
    return {"db_result": result["messages"][-1].content}


async def run_knowledge(state: ParallelState):
    """运行知识库子智能体"""
    result = await knowledge_agent.ainvoke({"messages": state["messages"]})
    return {"knowledge_result": result["messages"][-1].content}


def aggregate_results(state: ParallelState) -> str:
    """汇总所有子智能体结果"""
    results = []
    if state.get("search_result"):
        results.append(f"搜索结果：{state['search_result']}")
    if state.get("db_result"):
        results.append(f"数据库结果：{state['db_result']}")
    if state.get("knowledge_result"):
        results.append(f"知识库结果：{state['knowledge_result']}")
    return "\n\n".join(results)


def build_parallel_subagents_graph():
    """构建并行子智能体图（教程09：异步并行执行）"""
    graph = StateGraph(ParallelState)
    
    # 添加并行节点
    graph.add_node("search", run_search)
    graph.add_node("database", run_db)
    graph.add_node("knowledge", run_knowledge)
    
    # 并行执行（同时启动）
    graph.add_edge(START, "search")
    graph.add_edge(START, "database")
    graph.add_edge(START, "knowledge")
    
    # 所有节点完成后汇总
    def aggregator(state: ParallelState):
        return {"messages": [AIMessage(content=aggregate_results(state))]}
    
    graph.add_node("aggregate", aggregator)
    graph.add_edge("search", "aggregate")
    graph.add_edge("database", "aggregate")
    graph.add_edge("knowledge", "aggregate")
    graph.add_edge("aggregate", END)
    
    return graph.compile()


# ── 主智能体（调度者）────────────────────────────────────────

def build_main_agent():
    """构建主智能体（教程09：调度与汇总）"""
    
    main_agent_llm = llm.bind_tools([search_web, query_database, search_knowledge, analyze_code])
    
    async def main_agent_node(state: MessagesState):
        """主智能体：理解任务，调度子智能体"""
        # 主智能体决定是否调用工具（实际是调度子智能体）
        response = await main_agent_llm.ainvoke([
            {"role": "system", "content": """你是主智能体，负责任务规划和调度。

你有以下子智能体可以调度：
- search: 网络搜索专家
- database: 数据库专家
- knowledge: 知识库专家
- code: 代码分析专家

根据用户需求，选择合适的子智能体。
"""},
            *state["messages"],
        ])
        return {"messages": [response]}
    
    def should_dispatch(state: MessagesState) -> str:
        """检查是否需要调度子智能体"""
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "subagents"
        return END
    
    graph = StateGraph(MessagesState)
    graph.add_node("main", main_agent_node)
    graph.add_node("subagents", build_subagent_graph())
    graph.add_edge(START, "main")
    graph.add_conditional_edges("main", should_dispatch, {"subagents": "subagents", END: END})
    graph.add_edge("subagents", "main")
    
    return graph.compile()
