"""
智能体核心模块 - 对应教程章节：
- 01-LangGraph核心概念：StateGraph + ReAct循环
- 08-循环与自演进：Plan-and-Execute模式

本模块是整个 Agent 应用的核心，实现了两种经典的 Agent 模式：

1. ReAct 模式（Reasoning + Acting）
   - 最经典的 Agent 模式：思考 → 行动 → 观察 → 思考 → ...
   - 图结构：START → agent → (has_tool_calls?) → tools → agent → ... → END
   - 适合：需要灵活工具调用的场景

2. Plan-and-Execute 模式
   - 先规划再执行：planner 生成计划 → executor 逐步执行
   - 图结构：START → planner → executor → executor → ... → END
   - 适合：复杂多步骤任务

关键概念（教程01）：
- StateGraph：LangGraph 的核心抽象，定义状态流转
- MessagesState：内置的消息状态类型
- MemorySaver：检查点机制，实现对话记忆（教程06）
- ToolNode：预构建的工具执行节点
- conditional_edges：条件路由，决定下一步走向
"""

import json
import os
from typing import Optional, Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# 从公共配置模块导入（避免重复配置）
from config import API_KEY, BASE_URL, MODEL
# MCP 客户端：连接外部工具服务器（教程03）
from mcp_client import init_mcp, shutdown_mcp, get_mcp_tools


# ── 类型定义 ─────────────────────────────────────────────────

# Agent 模式类型：支持两种模式切换
AgentMode = Literal["react", "plan_execute"]


# ── 提示词工程（教程05）──────────────────────────────────────
# 提示词设计遵循 CLEAR 原则：清晰、逻辑、明确、可操作、鲁棒

SYSTEM_PROMPT = """你是一个有用的AI助手。

## 能力
你可以使用工具来帮助用户完成任务。当需要用到工具时，请主动调用它们。

## 回复规范
- 使用中文回复
- 调用工具后，根据工具结果给出友好的总结
- 如果不需要工具，直接回答即可
"""

PLAN_PROMPT = """你是一个任务规划专家。

## 职责
分析用户的复杂任务，将其分解为可执行的步骤计划。

## 输出格式
输出一个 JSON 数组，每个元素包含：
- step: 步骤编号
- action: 具体行动描述
- tool: 需要使用的工具名称（如果需要）
- expected_output: 预期输出
"""

EXECUTE_PROMPT = """你是一个任务执行专家。

## 职责
根据给定的计划步骤，逐步执行每个步骤。

## 当前步骤
{current_step}

## 已完成步骤的结果
{completed_results}

## 要求
- 只执行当前步骤
- 使用指定的工具（如果有）
- 返回执行结果
"""


# ── Plan-and-Execute 数据模型（教程08）───────────────────────

class PlanStep(BaseModel):
    """
    计划步骤模型
    
    用于 Plan-and-Execute 模式中，planner LLM 通过 structured output
    生成结构化的步骤列表。每个步骤包含：
    - step: 步骤编号，用于跟踪执行进度
    - action: 具体行动描述，告诉 executor 要做什么
    - tool: 可选的工具名称，如果该步骤需要调用工具
    - expected_output: 预期输出，用于验证执行结果
    """
    step: int = Field(description="步骤编号")
    action: str = Field(description="具体行动描述")
    tool: Optional[str] = Field(default=None, description="需要使用的工具名称")
    expected_output: str = Field(description="预期输出")


class Plan(BaseModel):
    """
    执行计划模型
    
    包含所有步骤的列表，planner LLM 会生成这个结构。
    使用 Pydantic BaseModel 确保 LLM 输出格式正确。
    """
    steps: list[PlanStep] = Field(description="步骤列表")


# ── Plan-and-Execute Agent 构建（教程08）─────────────────────

def build_plan_agent(tools: list, checkpointer: MemorySaver):
    """
    构建 Plan-and-Execute Agent（教程08：循环与自演进）
    
    这是一种"先规划再执行"的 Agent 模式：
    1. Planner 节点：分析用户任务，生成分步计划
    2. Executor 节点：逐步执行计划中的每个步骤
    3. 条件路由：检查是否还有更多步骤需要执行
    
    图结构：
        START → planner → (has_plan?) → executor → (more_steps?) → ... → END
    
    Args:
        tools: 可用工具列表（LangChain StructuredTool）
        checkpointer: 检查点器，用于对话记忆（教程06）
    
    Returns:
        编译好的 LangGraph 可执行图
    """
    # Planner LLM：使用 structured output 确保输出符合 Plan 模型
    # with_structured_output() 会让 LLM 以 JSON 格式输出，自动解析为 Pydantic 对象
    planner_llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
        api_key=API_KEY,
        base_url=BASE_URL,
    ).with_structured_output(Plan)

    # Executor LLM：绑定工具，让它能调用工具执行具体步骤
    # bind_tools() 将工具描述注入到 LLM 的 system prompt 中
    executor_llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
        api_key=API_KEY,
        base_url=BASE_URL,
    ).bind_tools(tools)

    # 定义 Plan-and-Execute 的状态结构
    # 继承 MessagesState（包含 messages 列表），额外添加计划相关字段
    class PlanState(MessagesState):
        plan: Optional[Plan] = None           # 当前执行计划
        current_step_index: int = 0           # 当前执行到第几步
        step_results: list[dict] = []         # 每步的执行结果

    # ── 节点定义 ──

    async def planner_node(state: PlanState):
        """
        规划节点：分析用户任务，生成分步执行计划
        
        这是 Plan-and-Execute 模式的核心：
        - 接收用户消息
        - 使用 planner LLM 生成结构化计划
        - 初始化执行状态（步骤索引归零，清空历史结果）
        """
        response = await planner_llm.ainvoke([
            {"role": "system", "content": PLAN_PROMPT},
            *state["messages"],
        ])
        return {"plan": response, "current_step_index": 0, "step_results": []}

    async def executor_node(state: PlanState):
        """
        执行节点：执行当前步骤
        
        每次执行：
        1. 取出当前步骤（current_step_index 指向的步骤）
        2. 构建包含已完成步骤结果的上下文
        3. 调用 executor LLM 执行当前步骤
        4. 记录步骤结果，推进索引
        """
        # 边界检查：如果没有计划或已执行完所有步骤
        if not state["plan"] or state["current_step_index"] >= len(state["plan"].steps):
            return {"messages": []}

        current_step = state["plan"].steps[state["current_step_index"]]
        
        # 构建执行上下文：让 executor 知道之前步骤的结果
        completed_results = "\n".join([
            f"步骤 {r['step']}: {r['result']}" 
            for r in state["step_results"]
        ]) or "暂无"

        execute_prompt = EXECUTE_PROMPT.format(
            current_step=f"步骤 {current_step.step}: {current_step.action}",
            completed_results=completed_results,
        )

        # 执行当前步骤
        response = await executor_llm.ainvoke([
            {"role": "system", "content": execute_prompt},
            *state["messages"],
        ])

        # 记录步骤结果
        step_result = {
            "step": current_step.step,
            "action": current_step.action,
            "result": response.content if hasattr(response, "content") else str(response),
        }
        new_step_results = state["step_results"] + [step_result]

        return {
            "messages": [response],
            "current_step_index": state["current_step_index"] + 1,
            "step_results": new_step_results,
        }

    # ── 路由函数 ──

    def should_continue_plan(state: PlanState) -> str:
        """
        条件路由：检查是否还有更多步骤需要执行
        
        返回 "executor" 表示继续执行下一步
        返回 END 表示所有步骤已完成
        """
        if not state["plan"]:
            return END
        if state["current_step_index"] >= len(state["plan"].steps):
            return END
        return "executor"

    # ── 构建 StateGraph ──
    # 这是 LangGraph 的核心 API（教程01）
    graph = StateGraph(PlanState)
    graph.add_node("planner", planner_node)     # 添加规划节点
    graph.add_node("executor", executor_node)   # 添加执行节点
    graph.add_edge(START, "planner")            # 入口 → 规划
    graph.add_conditional_edges(                # 规划后 → 检查是否有计划
        "planner", should_continue_plan,
        {"executor": "executor", END: END}
    )
    graph.add_conditional_edges(                # 执行后 → 检查是否还有步骤
        "executor", should_continue_plan,
        {"executor": "executor", END: END}
    )

    return graph.compile(checkpointer=checkpointer)


# ── ReAct Agent 构建（教程01核心）────────────────────────────

def build_react_agent(tools: list, checkpointer: MemorySaver):
    """
    构建 ReAct Agent（教程01核心模式）
    
    ReAct = Reasoning + Acting，最经典的 Agent 模式：
    1. Agent 节点：LLM 分析消息，决定是否调用工具
    2. Tools 节点：执行工具调用
    3. 条件路由：检查 LLM 输出是否包含工具调用
    
    图结构：
        START → agent → (has_tool_calls?) → tools → agent → ... → END
    
    这个循环会一直进行，直到 LLM 认为不需要调用工具为止。
    
    Args:
        tools: 可用工具列表（LangChain StructuredTool）
        checkpointer: 检查点器，用于对话记忆（教程06）
    
    Returns:
        编译好的 LangGraph 可执行图
    """
    # 绑定工具到 LLM
    # bind_tools() 会将工具的名称、描述、参数 schema 注入到 LLM 的 prompt 中
    # LLM 就能知道有哪些工具可用，以及如何调用它们
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
        api_key=API_KEY,
        base_url=BASE_URL,
    ).bind_tools(tools)

    # ── 节点定义 ──

    async def agent_node(state: MessagesState):
        """
        LLM 决策节点：分析消息，决定是否调用工具
        
        这是 ReAct 循环中的 "Reasoning" 部分：
        - 将系统提示 + 历史消息 + 当前消息发送给 LLM
        - LLM 可能返回普通文本（直接回答）
        - 也可能返回工具调用请求（需要执行工具）
        """
        response = await llm.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            *state["messages"],
        ])
        return {"messages": [response]}

    # ── 路由函数 ──

    def should_continue(state: MessagesState) -> str:
        """
        路由函数：检查最后一条消息是否包含工具调用
        
        这是 ReAct 循环的关键分支：
        - 如果 LLM 返回了 tool_calls → 路由到 "tools" 节点执行工具
        - 否则 → 路由到 END，结束循环
        
        LangGraph 的 conditional_edges 就是这个作用：
        根据当前状态动态决定下一步走哪条路
        """
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    # ── 构建 StateGraph ──
    # LangGraph 的核心 API（教程01）：
    # 1. 创建 StateGraph，指定状态类型
    # 2. 添加节点（node）：每个节点是一个处理函数
    # 3. 添加边（edge）：定义节点之间的流转关系
    # 4. 编译（compile）：生成可执行的图
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)           # 添加 Agent 决策节点
    graph.add_node("tools", ToolNode(tools))       # 添加工具执行节点（LangGraph 预构建）
    graph.add_edge(START, "agent")                 # 入口 → Agent
    graph.add_conditional_edges(                   # Agent → 条件判断
        "agent", should_continue,
        {"tools": "tools", END: END}               # 有工具调用 → tools，否则 → 结束
    )
    graph.add_edge("tools", "agent")               # 工具执行完 → 回到 Agent 继续推理

    return graph.compile(checkpointer=checkpointer)


# ── 全局实例管理 ─────────────────────────────────────────────

# 检查点器：LangGraph 内置的对话记忆机制（教程06）
# 每次对话都会保存状态，通过 thread_id 隔离不同对话
checkpointer = MemorySaver()

# 两种模式的 Agent 实例（延迟初始化）
react_agent = None            # ReAct 模式
plan_agent = None             # Plan-and-Execute 模式

# 当前激活的模式
current_mode: AgentMode = "react"


async def init_agent():
    """
    初始化 Agent（应用启动时调用）
    
    执行流程：
    1. 连接 MCP Server，发现可用工具（教程03）
    2. 构建 ReAct Agent 图
    3. 构建 Plan-and-Execute Agent 图
    4. 两种模式都预构建好，切换模式时直接使用
    """
    global react_agent, plan_agent
    
    # 连接 MCP Server 并获取工具列表
    await init_mcp()
    tools = get_mcp_tools()
    
    # 构建两种模式的 Agent
    react_agent = build_react_agent(tools, checkpointer)
    plan_agent = build_plan_agent(tools, checkpointer)
    
    print(f"[Agent] 初始化完成，当前模式：{current_mode}")
    print(f"[Agent] 可用工具：{[t.name for t in tools]}")


async def get_agent(mode: AgentMode = None):
    """
    获取已初始化的 Agent
    
    Args:
        mode: 指定模式（react 或 plan_execute），None 表示使用当前模式
    
    Returns:
        对应模式的编译好的 LangGraph 图
    """
    global react_agent, plan_agent, current_mode
    
    if mode:
        current_mode = mode
    
    # 延迟初始化：如果还没初始化过，先初始化
    if react_agent is None or plan_agent is None:
        await init_agent()
    
    return plan_agent if current_mode == "plan_execute" else react_agent


# ── 对话接口（教程01 + 06 + 08）──────────────────────────────

async def chat(user_message: str, thread_id: str = "default", mode: AgentMode = None):
    """
    与 Agent 对话的主入口
    
    使用 LangGraph 的 astream_events 实现流式输出，
    前端通过 SSE（Server-Sent Events）实时接收。
    
    Args:
        user_message: 用户输入的消息
        thread_id: 对话线程ID（教程06：通过 thread_id 实现记忆隔离）
                   同一个 thread_id 的对话会共享上下文
        mode: Agent 模式（react 或 plan_execute）
    
    Yields:
        事件字典，前端根据 type 字段渲染不同内容：
        - {"type": "token", "content": "..."}           LLM 流式输出的文字片段
        - {"type": "tool_call", "name": "...", "args": {...}}  Agent 决定调用工具
        - {"type": "tool_result", "name": "...", "result": "..."} 工具执行结果
        - {"type": "done", "content": "完整回复"}       对话完成
    """
    graph = await get_agent(mode)
    config = {"configurable": {"thread_id": thread_id}}

    full_response = ""

    # astream_events 是 LangGraph 的流式事件 API
    # version="v2" 使用最新的事件格式
    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
        version="v2",
    ):
        kind = event.get("event", "")

        # LLM 流式 token（每生成一个 token 就触发一次）
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                full_response += chunk.content
                yield {"type": "token", "content": chunk.content}

        # 工具调用开始（Agent 决定调用某个工具）
        elif kind == "on_tool_start":
            tool_name = event.get("name", "")
            tool_input = event.get("data", {}).get("input", {})
            yield {"type": "tool_call", "name": tool_name, "args": tool_input}

        # 工具调用结束（工具执行完成，返回结果）
        elif kind == "on_tool_end":
            tool_name = event.get("name", "")
            tool_output = event.get("data", {}).get("output", "")
            if hasattr(tool_output, "content"):
                tool_output = tool_output.content
            yield {"type": "tool_result", "name": tool_name, "result": str(tool_output)}

    # 最后发送完成事件，包含完整的回复内容
    yield {"type": "done", "content": full_response}


# ── 模式切换接口 ─────────────────────────────────────────────

def set_mode(mode: AgentMode):
    """
    切换 Agent 模式
    
    Args:
        mode: "react" 或 "plan_execute"
    
    Raises:
        ValueError: 如果传入不支持的模式
    """
    global current_mode
    if mode in ["react", "plan_execute"]:
        current_mode = mode
        print(f"[Agent] 模式切换为：{mode}")
    else:
        raise ValueError(f"不支持的模式：{mode}，可选：react, plan_execute")
