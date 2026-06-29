"""
提示词与上下文工程 - 对应教程 05-提示词与上下文工程

本模块演示了构建生产级 Agent 时的提示词工程最佳实践：

1. CLEAR 提示词原则（教程05.1）
   - C - Clear（清晰）：避免歧义
   - L - Logical（逻辑）：结构化组织
   - E - Explicit（明确）：具体要求
   - A - Actionable（可操作）：可执行的指令
   - R - Robust（鲁棒）：容错性强

2. 上下文组装技术（教程05.2）
   - 将用户画像、对话历史、检索文档等组装为完整上下文

3. 上下文窗口管理（教程05.3）
   - 裁剪策略：保留最近 N 条消息
   - 压缩策略：使用 LLM 摘要压缩历史
   - 优先级策略：高优先级消息始终保留

4. 动态上下文注入（教程05.4）
   - 根据运行时变量动态添加上下文

5. RAG 上下文注入（教程05.5）
   - 检索增强生成的上下文管理

6. 长任务上下文保持（教程05.6）
   - 超长对话的上下文压缩
"""

from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
import json

# 从公共配置导入 LLM（避免硬编码 API 密钥）
from config import llm


# ── 教程05.1：CLEAR 提示词原则 ─────────────────────────────
# CLEAR 原则是提示词工程的最佳实践框架，
# 确保提示词清晰、有逻辑、明确、可操作、鲁棒

CLEAR_PROMPT = """你是一个{role}。

## 能力
{capabilities}

## 约束
{constraints}

## 输出格式
{output_format}

## 示例
{examples}
"""

def create_clear_prompt(
    role: str,
    capabilities: str,
    constraints: str,
    output_format: str,
    examples: str
) -> str:
    """
    创建符合 CLEAR 原则的提示词（教程05.1）
    
    CLEAR 原则：
    - C - Clear（清晰）：角色定义明确
    - L - Logical（逻辑）：能力→约束→格式→示例，层次分明
    - E - Explicit（明确）：输出格式具体
    - A - Actionable（可操作）：能力列表可执行
    - R - Robust（鲁棒）：示例提供参照
    
    Args:
        role: AI 的角色定义（如"数据分析师"）
        capabilities: 能力列表
        constraints: 约束条件
        output_format: 期望的输出格式
        examples: 输入输出示例
    
    Returns:
        格式化后的提示词字符串
    """
    return CLEAR_PROMPT.format(
        role=role,
        capabilities=capabilities,
        constraints=constraints,
        output_format=output_format,
        examples=examples
    )


# ── 教程05.2：上下文组装技术 ───────────────────────────────
# 在实际应用中，Agent 收到的不仅仅是用户消息，
# 还需要组装用户画像、对话历史、检索文档等上下文信息

class ContextState(TypedDict):
    """
    上下文组装的状态定义
    
    包含多种上下文来源：
    - messages: 当前对话消息
    - user_profile: 用户画像（姓名、角色等）
    - conversation_history: 历史对话记录
    - retrieved_docs: RAG 检索到的文档
    """
    messages: list
    user_profile: Optional[dict] = None
    conversation_history: Optional[list] = None
    retrieved_docs: Optional[list] = None


def assemble_context(state: ContextState) -> list:
    """
    组装完整上下文（教程05.2）
    
    将多个上下文来源按优先级组装为消息列表：
    1. 系统提示（最高优先级）
    2. 用户画像（个性化）
    3. 对话历史（连贯性）
    4. 检索文档（知识增强）
    5. 当前消息（用户输入）
    
    这种组装方式确保了 Agent 能获取到全面的上下文信息。
    """
    context_parts = []
    
    # 1. 系统提示（始终在最前面）
    system_prompt = "你是一个有用的助手。"
    context_parts.append(SystemMessage(content=system_prompt))
    
    # 2. 用户画像（如果有，帮助 Agent 个性化回复）
    if state.get("user_profile"):
        profile_str = json.dumps(state["user_profile"], ensure_ascii=False)
        context_parts.append(SystemMessage(content=f"用户画像：{profile_str}"))
    
    # 3. 对话历史（最近10条消息，保持连贯性）
    if state.get("conversation_history"):
        history = state["conversation_history"][-10:]
        context_parts.extend(history)
    
    # 4. 检索到的文档（RAG 场景：注入外部知识）
    if state.get("retrieved_docs"):
        docs_str = "\n\n".join(state["retrieved_docs"])
        context_parts.append(SystemMessage(content=f"参考文档：\n{docs_str}"))
    
    # 5. 当前消息（用户最新输入）
    context_parts.append(state["messages"][-1])
    
    return context_parts


# ── 教程05.3：上下文窗口管理 ───────────────────────────────
# LLM 有上下文窗口限制（如 4K/8K/128K tokens），
# 需要管理上下文长度，避免超出限制或浪费 token

class TrimmingState(TypedDict):
    """上下文裁剪状态"""
    messages: list
    token_limit: int = 4000


def trim_context(state: TrimmingState) -> list:
    """
    裁剪上下文 - 策略1：保留系统消息 + 最近N条（教程05.3）
    
    最简单的裁剪策略：
    - 系统消息始终保留（包含重要指令）
    - 只保留最近8条用户/助手消息
    - 丢弃更早的历史消息
    
    优点：简单高效
    缺点：可能丢失重要的早期对话
    """
    messages = state["messages"]
    
    # 分离系统消息和普通消息
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    user_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    
    # 只保留最近8条
    recent_messages = user_messages[-8:]
    
    return system_messages + recent_messages


def compress_context(state: TrimmingState) -> list:
    """
    压缩上下文 - 策略2：Token 级别精确裁剪（教程05.3）
    
    使用 LangChain 内置的 trim_messages 工具：
    - 按 token 数量裁剪（而非消息数量）
    - strategy="last"：保留最近的消息
    - allow_partial=False：不允许部分消息（保持完整性）
    
    优点：精确控制 token 用量
    缺点：仍然会丢失早期信息
    """
    messages = state["messages"]
    
    trimmed = trim_messages(
        messages,
        strategy="last",                    # 保留最近的消息
        token_counter=llm,                  # 使用 LLM 的 tokenizer 计算 token 数
        max_tokens=state.get("token_limit", 4000),
        allow_partial=False,                # 不截断单条消息
    )
    
    return trimmed


def priority_context(state: TrimmingState) -> list:
    """
    优先级上下文 - 策略3：按重要性保留（教程05.3）
    
    不是所有消息都同等重要：
    - 包含"重要"、"关键"、"注意"等关键词的消息 → 高优先级，始终保留
    - 其他消息 → 普通优先级，只保留最近5条
    
    优点：保留关键信息，不会丢失重要上下文
    缺点：关键词匹配可能不够准确
    """
    messages = state["messages"]
    
    priority_messages = []
    normal_messages = []
    
    for msg in messages:
        content = getattr(msg, "content", "")
        # 高优先级关键词
        if any(keyword in content for keyword in ["重要", "关键", "注意"]):
            priority_messages.append(msg)
        else:
            normal_messages.append(msg)
    
    # 所有高优先级 + 最近5条普通消息
    return priority_messages + normal_messages[-5:]


# ── 教程05.4：动态上下文注入 ───────────────────────────────
# 在运行时根据变量动态注入上下文信息

class DynamicContextState(MessagesState):
    """动态上下文状态，支持运行时变量注入"""
    context_vars: dict = {}


def inject_context(state: DynamicContextState) -> list:
    """
    动态注入上下文变量（教程05.4）
    
    将 context_vars 中的键值对格式化为系统消息，
    插入到第一条系统消息之后。
    
    典型用途：
    - 注入当前时间、日期
    - 注入用户身份信息
    - 注入当前项目/任务信息
    """
    messages = list(state["messages"])
    
    if state.get("context_vars"):
        # 将变量格式化为可读文本
        context_str = "\n".join([
            f"- {k}: {v}" for k, v in state["context_vars"].items()
        ])
        
        # 在第一条系统消息后插入上下文
        for i, msg in enumerate(messages):
            if isinstance(msg, SystemMessage):
                messages.insert(
                    i + 1,
                    SystemMessage(content=f"## 当前上下文\n{context_str}")
                )
                break
    
    return messages


# ── 教程05.5：RAG 场景上下文注入 ───────────────────────────
# RAG（检索增强生成）是 Agent 获取外部知识的核心方式

class RAGState(TypedDict):
    """RAG 状态：包含查询和检索到的文档"""
    messages: list
    query: str
    documents: list


async def retrieve_documents(state: RAGState) -> list:
    """
    检索相关文档（教程05.5：RAG）
    
    这里是模拟检索，实际应调用向量数据库（如 FAISS、Pinecone）。
    RAG 的核心流程：
    1. 用户提问 → 2. 向量检索 → 3. 注入上下文 → 4. LLM 生成回答
    """
    query = state["query"]
    
    # 模拟检索结果
    documents = [
        f"文档1：关于{query}的信息...",
        f"文档2：关于{query}的更多细节...",
    ]
    
    return {"documents": documents}


def inject_rag_context(state: RAGState) -> list:
    """
    注入 RAG 上下文到提示词
    
    将检索到的文档格式化后注入系统提示，
    让 LLM 基于检索结果回答问题。
    """
    docs_str = "\n\n".join([
        f"## 文档 {i+1}\n{doc}"
        for i, doc in enumerate(state.get("documents", []))
    ])
    
    rag_prompt = f"""请根据以下参考文档回答问题。

{docs_str}

如果文档中没有相关信息，请说明无法回答。
"""
    
    return [
        SystemMessage(content=rag_prompt),
        state["messages"][-1]
    ]


def build_rag_graph():
    """
    构建 RAG 图（教程05.5）
    
    简单的 RAG 流程：
    START → retrieve（检索文档） → inject（注入上下文） → END
    """
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_documents)
    graph.add_node("inject", inject_rag_context)
    
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "inject")
    graph.add_edge("inject", END)
    
    return graph.compile()


# ── 教程05.6：长任务上下文保持 ─────────────────────────────
# 超长对话会超出上下文窗口限制，需要压缩历史

class LongTaskState(TypedDict):
    """长任务状态，支持上下文摘要压缩"""
    messages: list
    task_summary: Optional[str] = None
    subtask: Optional[str] = None
    subtask_result: Optional[str] = None


async def summarize_context(state: LongTaskState) -> str:
    """
    摘要压缩上下文（教程05.6：长任务）
    
    当对话历史过长时，使用 LLM 将历史消息压缩为简短摘要，
    保留关键信息，大幅减少 token 消耗。
    
    这是一种"有损压缩"：会丢失细节，但保留关键信息。
    """
    # 取最近10条消息进行摘要
    history_str = "\n".join([
        f"{m.type}: {m.content}"
        for m in state["messages"][-10:]
    ])
    
    response = await llm.ainvoke([
        {"role": "system", "content": "请总结以下对话的关键信息，保持简洁。"},
        {"role": "user", "content": history_str}
    ])
    
    return response.content


def build_long_task_graph():
    """
    构建长任务图（教程05.6）
    
    当消息超过20条时，自动压缩历史为摘要，
    避免上下文过长导致的问题。
    """
    
    async def task_node(state: LongTaskState):
        # 如果上下文太长，先用 LLM 摘要压缩
        if len(state["messages"]) > 20:
            summary = await summarize_context(state)
            return {
                "task_summary": summary,
                "messages": [
                    SystemMessage(content=f"对话摘要：{summary}"),
                    state["messages"][-1]  # 保留最新消息
                ]
            }
        return {"messages": state["messages"]}
    
    graph = StateGraph(LongTaskState)
    graph.add_node("task", task_node)
    graph.add_edge(START, "task")
    graph.add_edge("task", END)
    
    return graph.compile()


# ── 使用示例 ───────────────────────────────────────────────

def demo_clear_prompt():
    """演示 CLEAR 提示词原则"""
    print("=== 教程05.1：CLEAR 提示词 ===")
    
    prompt = create_clear_prompt(
        role="数据分析师",
        capabilities="- 数据分析\n- 可视化\n- 报告撰写",
        constraints="- 使用中文\n- 数据准确\n- 逻辑清晰",
        output_format="JSON 格式：\n{\n  \"分析\": \"...\",\n  \"建议\": [...]\n}",
        examples="输入：分析销售数据\n输出：{\"分析\": \"销售增长20%\", \"建议\": [\"加大营销\"]}"
    )
    
    print(prompt)


def demo_context_assembly():
    """演示上下文组装"""
    print("\n=== 教程05.2：上下文组装 ===")
    
    state = {
        "messages": [HumanMessage(content="帮我分析一下")],
        "user_profile": {"name": "张三", "role": "经理"},
        "conversation_history": [
            HumanMessage(content="上周销售数据如何？"),
            AIMessage(content="上周销售增长15%"),
        ],
        "retrieved_docs": ["文档1：销售报告", "文档2：市场分析"]
    }
    
    context = assemble_context(state)
    print(f"组装后上下文条数：{len(context)}")
    for i, msg in enumerate(context):
        print(f"  {i+1}. {type(msg).__name__}: {msg.content[:30]}...")


def demo_context_trimming():
    """演示上下文裁剪三种策略"""
    print("\n=== 教程05.3：上下文裁剪 ===")
    
    # 模拟长对话（1条系统消息 + 40条对话）
    messages = [SystemMessage(content="你是助手")]
    for i in range(20):
        messages.append(HumanMessage(content=f"消息 {i+1}"))
        messages.append(AIMessage(content=f"回复 {i+1}"))
    
    state = {"messages": messages, "token_limit": 4000}
    
    # 策略1：保留系统消息 + 最近8条
    trimmed = trim_context(state)
    print(f"原始消息数：{len(messages)}")
    print(f"裁剪后消息数：{len(trimmed)}")
    
    # 策略2：使用 LangChain trim（按 token 数）
    compressed = compress_context(state)
    print(f"压缩后消息数：{len(compressed)}")
    
    # 策略3：优先级保留
    priority = priority_context(state)
    print(f"优先级保留消息数：{len(priority)}")


def demo_dynamic_context():
    """演示动态上下文注入"""
    print("\n=== 教程05.4：动态上下文注入 ===")
    
    state = {
        "messages": [HumanMessage(content="今天的任务是什么？")],
        "context_vars": {
            "日期": "2026-06-29",
            "用户": "张三",
            "项目": "Agent开发"
        }
    }
    
    context = inject_context(state)
    print(f"注入后上下文条数：{len(context)}")
    for msg in context:
        if isinstance(msg, SystemMessage):
            print(f"系统消息：{msg.content[:50]}...")
