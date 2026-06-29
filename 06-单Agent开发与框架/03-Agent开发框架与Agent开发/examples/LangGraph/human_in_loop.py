"""
人机协作与中断恢复 - 对应教程 10-人机协作与中断恢复

本模块演示了 LangGraph 的人机协作机制：

1. 人类审批（教程10.1）
   - 使用 interrupt() 函数暂停 Agent 执行
   - 等待人类确认后再继续
   - 典型场景：危险操作（删除数据、发送邮件）

2. 中断恢复（教程10.2）
   - 使用 Command 模式恢复执行
   - 支持“继续”、“取消”、“修改参数”等操作

3. 人类编辑工具参数（教程10.3）
   - Agent 生成工具调用参数后，人类可以编辑
   - 典型场景：邮件内容审核、金额确认

核心 API：
- interrupt(value): 暂停图执行，将 value 传递给人类
- Command(resume=value): 恢复图执行，将 value 传回节点
"""

from typing import TypedDict, Optional

# 从公共配置导入 LLM（避免硬编码 API 密钥）
from config import llm

from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import interrupt, Command

# ── 工具定义 ──────────────────────────────────────────────

@tool
def delete_data(table: str, condition: str) -> str:
    """删除数据库数据（危险操作）。"""
    return f"已删除 {table} 中满足 {condition} 的数据"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件。"""
    return f"邮件已发送给 {to}"

# ── 教程10.1：人类审批（interrupt）─────────────────────────

class ApprovalState(MessagesState):
    """需要审批的状态"""
    needs_approval: bool = False
    approval_result: Optional[str] = None


def delete_node(state: ApprovalState):
    """删除操作节点（需要人类审批）"""
    last_message = state["messages"][-1]
    
    # 教程10核心：interrupt() 函数
    # 暂停执行，等待人类输入
    approval = interrupt({
        "question": "确认要删除数据吗？",
        "details": {
            "operation": "delete_data",
            "table": last_message.content,
        }
    })
    
    return {
        "needs_approval": True,
        "approval_result": approval,
        "messages": [ToolMessage(content=f"用户审批结果：{approval}", tool_call_id="")]
    }


def build_approval_graph():
    """构建需要审批的图（教程10.1）"""
    llm_with_tools = llm.bind_tools([delete_data])
    
    async def agent_node(state: ApprovalState):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}
    
    def check_tool_calls(state: ApprovalState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            # 检查是否是危险操作
            tool_name = last.tool_calls[0].get("name", "")
            if tool_name == "delete_data":
                return "delete"
        return END
    
    graph = StateGraph(ApprovalState)
    graph.add_node("agent", agent_node)
    graph.add_node("delete", delete_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", check_tool_calls, {"delete": "delete", END: END})
    graph.add_edge("delete", END)
    
    return graph.compile()


# ── 教程10.2：中断恢复（Command）───────────────────────────

class ResumeState(MessagesState):
    """支持中断恢复的状态"""
    paused: bool = False


async def long_running_node(state: ResumeState):
    """长时间运行的节点（可以被中断和恢复）"""
    # 模拟长时间操作
    interrupt_value = interrupt({
        "type": "progress",
        "message": "任务执行到50%，是否继续？",
        "options": ["继续", "取消", "修改参数"]
    })
    
    if interrupt_value == "取消":
        return {
            "paused": True,
            "messages": [AIMessage(content="任务已取消")]
        }
    
    return {"messages": [AIMessage(content=f"任务完成，用户选择：{interrupt_value}")]}


def build_resume_graph():
    """构建支持中断恢复的图（教程10.2）"""
    graph = StateGraph(ResumeState)
    graph.add_node("work", long_running_node)
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    return graph.compile()


# ── 教程10.3：人类编辑工具参数 ─────────────────────────────

class EditParamsState(MessagesState):
    """支持编辑工具参数的状态"""
    tool_params: Optional[dict] = None


async def agent_with_edit(state: EditParamsState):
    """Agent节点（允许人类编辑工具参数）"""
    llm_with_tools = llm.bind_tools([send_email])
    response = await llm_with_tools.ainvoke(state["messages"])
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_call = response.tool_calls[0]
        
        # 人类可以编辑工具参数
        edited_params = interrupt({
            "type": "edit_params",
            "tool": tool_call["name"],
            "params": tool_call["args"],
            "message": "请确认或编辑工具参数"
        })
        
        return {
            "tool_params": edited_params,
            "messages": [
                response,
                ToolMessage(
                    content=f"使用编辑后的参数执行：{edited_params}",
                    tool_call_id=tool_call["id"]
                )
            ]
        }
    
    return {"messages": [response]}


def build_edit_params_graph():
    """构建支持编辑参数的图（教程10.3）"""
    graph = StateGraph(EditParamsState)
    graph.add_node("agent", agent_with_edit)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()


# ── 使用示例 ───────────────────────────────────────────────

def demo_approval():
    """演示人类审批流程"""
    print("=== 教程10.1：人类审批 ===")
    graph = build_approval_graph()
    
    # 第一次调用会中断
    result = graph.invoke({
        "messages": [HumanMessage(content="删除 users 表中 status=inactive 的数据")]
    })
    print(f"中断，需要审批")
    
    # 人类审批后恢复
    # 实际使用时通过 command 恢复
    print("等待人类审批...")


def demo_resume():
    """演示中断恢复"""
    print("\n=== 教程10.2：中断恢复 ===")
    graph = build_resume_graph()
    
    # 第一次调用会暂停
    result = graph.invoke({"messages": [], "paused": False})
    print(f"暂停状态: {result.get('paused')}")
    
    # 恢复执行
    # result = graph.invoke(None)  # 从暂停处恢复


def demo_edit_params():
    """演示编辑工具参数"""
    print("\n=== 教程10.3：编辑工具参数 ===")
    graph = build_edit_params_graph()
    
    result = graph.invoke({
        "messages": [HumanMessage(content="发送邮件给 boss@company.com，主题是周报")]
    })
    print(f"工具参数: {result.get('tool_params')}")
