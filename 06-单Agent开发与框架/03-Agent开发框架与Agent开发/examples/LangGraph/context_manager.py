"""
上下文管理 - 对应教程 05-提示词与上下文工程

本模块实现教程中提到的三种上下文管理策略：

1. 上下文裁剪（trim_context）
   - 当消息数/token数超过阈值时，删除早期消息
   - 保留 System 消息和最近的消息

2. 上下文压缩（summarize_conversation）
   - 用 LLM 生成对话摘要，替代原始消息
   - 大幅减少 token 消耗

3. 上下文统计（get_context_stats）
   - 统计消息数、估算 token 数
   - 提供使用率百分比

教程对应：
- 05-提示词与上下文工程：2.3 上下文窗口管理策略
"""

from typing import Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage


# ── 上下文统计 ────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数
    
    简单估算：中文约 1.5 字符/token，英文约 4 字符/token
    这里用保守估计：总字符数 / 2
    
    Args:
        text: 要估算的文本
    
    Returns:
        估算的 token 数
    """
    return len(text) // 2


def get_context_stats(messages: list[dict], max_tokens: int = 100000) -> dict:
    """
    获取上下文统计信息
    
    用于前端展示当前上下文使用情况：
    - 消息数量
    - 估算 token 数
    - 使用率百分比
    - 是否接近限制
    
    Args:
        messages: 消息列表，每项包含 role 和 content
        max_tokens: 最大 token 限制
    
    Returns:
        统计信息字典
    """
    total_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
    usage_percent = (total_tokens / max_tokens * 100) if max_tokens > 0 else 0
    
    return {
        "message_count": len(messages),
        "estimated_tokens": total_tokens,
        "max_tokens": max_tokens,
        "usage_percent": round(usage_percent, 1),
        "is_near_limit": usage_percent > 80,
        "role_counts": _count_by_role(messages),
    }


def _count_by_role(messages: list[dict]) -> dict:
    """按角色统计消息数"""
    counts = {"user": 0, "assistant": 0, "system": 0, "tool": 0}
    for m in messages:
        role = m.get("role", "unknown")
        counts[role] = counts.get(role, 0) + 1
    return counts


# ── 策略 1：上下文裁剪 ────────────────────────────────────────

def trim_context(messages: list[dict], max_tokens: int = 80000) -> list[dict]:
    """
    上下文裁剪：删除早期消息，保留最近的消息
    
    策略：
    1. 保留所有 System 消息（优先级最高）
    2. 从最早的非 System 消息开始删除
    3. 至少保留 2 条非 System 消息（保持对话连贯）
    
    对应教程 05：策略 1 - 上下文裁剪
    
    Args:
        messages: 原始消息列表
        max_tokens: 最大 token 限制
    
    Returns:
        裁剪后的消息列表
    """
    total_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
    
    if total_tokens <= max_tokens:
        return messages  # 未超限，直接返回
    
    # 分离 System 消息和其他消息
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]
    
    # 从最早的开始删除，至少保留 2 条
    while len(other_msgs) > 2:
        other_msgs.pop(0)
        total_tokens = sum(estimate_tokens(m.get("content", "")) for m in system_msgs + other_msgs)
        if total_tokens <= max_tokens:
            break
    
    return system_msgs + other_msgs


# ── 策略 2：上下文压缩（Summarization）───────────────────────

async def summarize_conversation(messages: list[dict], llm=None) -> str:
    """
    上下文压缩：用 LLM 生成对话摘要
    
    将多轮对话压缩为一段摘要，大幅减少 token 消耗。
    使用便宜的模型（如 gpt-4o-mini）做摘要，节省成本。
    
    对应教程 05：策略 2 - 上下文压缩
    
    Args:
        messages: 要压缩的消息列表
        llm: 用于生成摘要的 LLM（可选，默认使用 config 中的 llm）
    
    Returns:
        对话摘要文本
    """
    if not messages:
        return ""
    
    # 构建对话文本
    conversation_text = "\n".join([
        f"{m.get('role', 'unknown')}: {m.get('content', '')}"
        for m in messages
    ])
    
    # 如果没有传入 llm，使用 config 中的
    if llm is None:
        from config import llm
    
    summary_prompt = f"""请将以下对话压缩为简洁的摘要，保留关键信息和决策：

{conversation_text}

摘要（200字以内）："""
    
    response = await llm.ainvoke([
        {"role": "user", "content": summary_prompt}
    ])
    
    return response.content if hasattr(response, "content") else str(response)


def build_compressed_context(summary: str, recent_messages: list[dict]) -> list[dict]:
    """
    构建压缩后的上下文
    
    将摘要 + 最近几条消息组合成新的上下文：
    - System 消息包含摘要
    - 后面跟着最近的消息（保留对话连贯性）
    
    Args:
        summary: 对话摘要
        recent_messages: 最近的消息（保留原样）
    
    Returns:
        压缩后的消息列表
    """
    compressed = [
        {"role": "system", "content": f"之前的对话摘要：{summary}"}
    ]
    compressed.extend(recent_messages)
    return compressed


# ── 策略 3：上下文优先级管理 ──────────────────────────────────

def prioritize_context(messages: list[dict], max_tokens: int) -> list[dict]:
    """
    按优先级选择上下文
    
    优先级排序：
    1. System 消息（最高优先级，必须保留）
    2. Tool 结果（工具调用结果，重要上下文）
    3. User 消息（用户输入）
    4. Assistant 消息（AI 回复，优先级最低）
    
    对应教程 05：策略 3 - 上下文优先级管理
    
    Args:
        messages: 原始消息列表
        max_tokens: 最大 token 限制
    
    Returns:
        按优先级选择的消息列表
    """
    priorities = {
        "system": 100,
        "tool": 80,
        "user": 60,
        "assistant": 40,
    }
    
    # 按优先级排序（高优先级在前）
    sorted_msgs = sorted(
        messages,
        key=lambda m: priorities.get(m.get("role", ""), 0),
        reverse=True
    )
    
    # 选择消息直到达到限制
    selected = []
    total_tokens = 0
    
    for msg in sorted_msgs:
        msg_tokens = estimate_tokens(msg.get("content", ""))
        if total_tokens + msg_tokens <= max_tokens:
            selected.append(msg)
            total_tokens += msg_tokens
        else:
            break
    
    # 按原始顺序返回（保持对话逻辑）
    selected.sort(key=lambda m: messages.index(m) if m in messages else 0)
    
    return selected
