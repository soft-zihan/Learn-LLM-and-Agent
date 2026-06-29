"""
智能体设计模式 - 对应教程 07-智能体设计模式

本模块演示了三种重要的 Agent 设计模式：

1. Reflection 模式（反思循环）（教程07.1）
   - 生成 → 评审 → 改进 → 再生成
   - 类似人类的"写文章→修改→再修改"过程
   - 图结构：START → generate → critique → (合格?) → refine → generate → ... → END

2. 工具链设计（教程07.2）
   - 按顺序调用多个工具，形成工具链
   - 支持链式中断（验证失败时终止）

3. 验证与护栏（教程07.3）
   - 输入验证：拦截不安全的内容
   - 输出验证：过滤敏感信息
   - 图结构：START → validate_input → (安全?) → agent → validate_output → END
"""

# 从公共配置导入 LLM（避免硬编码 API 密钥）
from config import llm

from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field

# ── 工具定义 ──────────────────────────────────────────────

@tool
def search(query: str) -> str:
    """搜索信息。"""
    return f"搜索结果：{query}"

@tool
def write_report(content: str) -> str:
    """撰写报告。"""
    return f"报告已生成：{content[:50]}..."

@tool
def validate_report(report: str) -> str:
    """验证报告质量。"""
    # 简单验证逻辑
    if len(report) < 50:
        return "不合格：报告太短"
    if "关键发现" not in report:
        return "不合格：缺少关键发现"
    return "合格"

# ── 教程07.1：Reflection 模式 ──────────────────────────────

class ReflectionState(TypedDict):
    """Reflection 模式状态"""
    messages: list
    draft: Optional[str] = None
    critique: Optional[str] = None
    improved_draft: Optional[str] = None
    iteration: int = 0
    max_iterations: int = 3


class QualityScore(BaseModel):
    """质量评分"""
    score: int = Field(description="质量评分 1-10")
    issues: list[str] = Field(description="存在的问题列表")
    suggestions: list[str] = Field(description="改进建议列表")


async def generator_node(state: ReflectionState):
    """生成器节点：撰写初稿"""
    llm_with_tools = llm.bind_tools([write_report])
    
    response = await llm_with_tools.ainvoke([
        {"role": "system", "content": "你是一个报告生成专家。请根据用户要求撰写报告。"},
        *state["messages"],
    ])
    
    return {
        "draft": response.content,
        "iteration": state["iteration"] + 1
    }


async def critic_node(state: ReflectionState):
    """评审节点：评估质量"""
    response = await llm.with_structured_output(QualityScore).ainvoke([
        {"role": "system", "content": "你是一个严格的评审专家。请评估报告质量并给出改进建议。"},
        {"role": "user", "content": f"请评估以下报告：\n\n{state['draft']}"}
    ])
    
    return {"critique": f"评分：{response.score}/10\n问题：{', '.join(response.issues)}\n建议：{', '.join(response.suggestions)}"}


async def refiner_node(state: ReflectionState):
    """改进节点：根据评审意见改进"""
    response = await llm.ainvoke([
        {"role": "system", "content": "你是一个改进专家。根据评审意见改进报告。"},
        {"role": "user", "content": f"""原报告：
{state['draft']}

评审意见：
{state['critique']}

请改进报告。"""}
    ])
    
    return {"improved_draft": response.content}


def should_continue_reflection(state: ReflectionState) -> str:
    """检查是否需要继续迭代"""
    if state["iteration"] >= state["max_iterations"]:
        return END
    
    # 检查质量评分（这里简化处理）
    if state.get("critique") and "合格" in state["critique"]:
        return END
    
    return "refine"


def build_reflection_graph():
    """构建 Reflection 模式图（教程07.1）"""
    graph = StateGraph(ReflectionState)
    
    graph.add_node("generate", generator_node)
    graph.add_node("critique", critic_node)
    graph.add_node("refine", refiner_node)
    
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges("critique", should_continue_reflection, {"refine": "refine", END: END})
    graph.add_edge("refine", "generate")  # 回到生成器重新生成
    
    return graph.compile()


# ── 教程07.2：工具链设计 ───────────────────────────────────

class ToolChainState(MessagesState):
    """工具链状态"""
    chain_result: Optional[str] = None


async def tool_chain_node(state: ToolChainState):
    """工具链节点：按顺序调用工具"""
    tools = [search, write_report, validate_report]
    results = []
    
    for tool in tools:
        # 模拟顺序调用
        result = await tool.ainvoke({"query": state["messages"][-1].content})
        results.append(result)
        
        # 验证器：如果不合格，终止链
        if tool.name == "validate_report":
            if "不合格" in result:
                return {
                    "chain_result": f"工具链终止：{result}",
                    "messages": [AIMessage(content=result)]
                }
    
    return {
        "chain_result": "\n".join(results),
        "messages": [AIMessage(content="工具链执行完成")]
    }


def build_tool_chain_graph():
    """构建工具链图（教程07.2）"""
    graph = StateGraph(ToolChainState)
    graph.add_node("chain", tool_chain_node)
    graph.add_edge(START, "chain")
    graph.add_edge("chain", END)
    return graph.compile()


# ── 教程07.3：验证与护栏 ───────────────────────────────────

class GuardrailsState(MessagesState):
    """护栏状态"""
    validated: bool = False
    validation_result: Optional[str] = None


async def input_validation(state: GuardrailsState):
    """输入验证护栏"""
    last_message = state["messages"][-1].content.lower()
    
    # 安全检查
    blocked_words = ["删除所有", "格式化", "密码", "黑客"]
    for word in blocked_words:
        if word in last_message:
            return {
                "validated": False,
                "validation_result": f"输入被拒绝：包含敏感词'{word}'",
                "messages": [AIMessage(content=f"抱歉，您的请求包含不安全的操作。")]
            }
    
    return {"validated": True}


async def output_validation(state: GuardrailsState):
    """输出验证护栏"""
    if not state.get("validated"):
        return state
    
    # 检查输出是否包含敏感信息
    last_message = state["messages"][-1].content
    if any(word in last_message for word in ["密码", "token", "secret"]):
        return {
            "messages": [AIMessage(content="输出已过滤：包含敏感信息")]
        }
    
    return state


def build_guardrails_graph():
    """构建护栏图（教程07.3）"""
    llm_with_tools = llm.bind_tools([search, write_report])
    
    async def agent_node(state: GuardrailsState):
        response = await llm_with_tools.ainvoke([
            {"role": "system", "content": "你是一个安全的AI助手。"},
            *state["messages"],
        ])
        return {"messages": [response]}
    
    graph = StateGraph(GuardrailsState)
    graph.add_node("validate_input", input_validation)
    graph.add_node("agent", agent_node)
    graph.add_node("validate_output", output_validation)
    
    graph.add_edge(START, "validate_input")
    graph.add_conditional_edges(
        "validate_input",
        lambda state: "agent" if state.get("validated") else END
    )
    graph.add_edge("agent", "validate_output")
    graph.add_edge("validate_output", END)
    
    return graph.compile()


# ── 使用示例 ───────────────────────────────────────────────

def demo_reflection():
    """演示 Reflection 模式"""
    print("=== 教程07.1：Reflection 模式 ===")
    graph = build_reflection_graph()
    
    result = graph.invoke({
        "messages": [HumanMessage(content="写一份关于AI发展的报告")],
        "iteration": 0,
        "max_iterations": 3
    })
    
    print(f"初稿：{result.get('draft', '无')[:100]}...")
    print(f"评审：{result.get('critique', '无')}")
    print(f"改进稿：{result.get('improved_draft', '无')[:100]}...")


def demo_tool_chain():
    """演示工具链"""
    print("\n=== 教程07.2：工具链设计 ===")
    graph = build_tool_chain_graph()
    
    result = graph.invoke({
        "messages": [HumanMessage(content="搜索AI并撰写报告")]
    })
    
    print(f"工具链结果：{result.get('chain_result', '无')}")


def demo_guardrails():
    """演示护栏"""
    print("\n=== 教程07.3：验证与护栏 ===")
    graph = build_guardrails_graph()
    
    # 安全请求
    result1 = graph.invoke({
        "messages": [HumanMessage(content="搜索AI最新进展")]
    })
    print(f"安全请求：{result1['messages'][-1].content[:50]}...")
    
    # 危险请求（会被拦截）
    result2 = graph.invoke({
        "messages": [HumanMessage(content="删除所有数据库")]
    })
    print(f"危险请求：{result2['messages'][-1].content}")