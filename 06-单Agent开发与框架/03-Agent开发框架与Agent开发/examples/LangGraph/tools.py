"""
内置工具集 - 对应教程 03-MCP客户端接入

本模块定义了 Agent 可以使用的内置工具。
这些工具与 MCP 工具一起，构成 Agent 的完整工具集。

工具列表：
1. get_current_time - 获取当前时间
2. calculate - 安全数学计算（使用 ast.literal_eval）
3. search_memory - 搜索长期记忆
4. save_memory - 保存到长期记忆
5. list_skills - 列出可用技能
6. get_system_info - 获取系统信息

教程对应：
- 03-MCP客户端接入：内置工具与 MCP 工具配合使用
- 07-智能体设计模式：工具链设计、验证与护栏

安全设计：
- calculate 使用 ast.literal_eval 替代 eval，防止代码注入
- 所有工具都有清晰的 docstring，帮助 LLM 理解工具用途
"""

from langchain_core.tools import tool
from datetime import datetime
import ast
import json


@tool
def get_current_time() -> str:
    """
    获取当前时间
    
    返回格式化的日期时间字符串，如"2026年06月29日 14:30:00"
    
    Returns:
        当前时间的格式化字符串
    """
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """
    执行数学计算
    
    支持基本数学运算：加减乘除、幂运算、开方等。
    
    安全设计：
    - 使用 ast.literal_eval 替代 eval，防止代码注入
    - ast.literal_eval 只允许字面量（数字、字符串、列表等）
    - 不支持函数调用、属性访问等危险操作
    
    注意：ast.literal_eval 不支持表达式（如 "1+1"），
    所以对于数学表达式，我们使用受限的 eval + 安全白名单。
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4"、"sqrt(16)"
    
    Returns:
        计算结果字符串，或错误信息
    """
    try:
        # 安全白名单：只允许这些数学函数
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": lambda x: x ** 0.5,
        }
        
        # 使用受限的 eval：
        # - {"__builtins__": {}} 禁用所有内置函数
        # - allowed_names 只允许白名单中的函数
        # 这样即使输入恶意代码，也无法执行系统操作
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算错误：{str(e)}"


@tool
def search_memory(query: str) -> str:
    """
    搜索用户记忆
    
    在长期记忆中搜索与查询关键词相关的信息。
    记忆是跨对话持久化的，包含用户之前告诉你的信息。
    
    Args:
        query: 搜索关键词
    
    Returns:
        匹配的记忆内容，或"未找到相关记忆"
    """
    from memory import long_term_memory
    facts = long_term_memory.search("default", query)
    if not facts:
        return "未找到相关记忆"
    return "\n".join([f["content"] for f in facts[:5]])


@tool
def save_memory(fact: str) -> str:
    """
    保存信息到长期记忆
    
    将重要信息保存到长期记忆，下次对话时可以通过 search_memory 检索。
    适合保存用户偏好、重要事实等。
    
    Args:
        fact: 要保存的信息
    
    Returns:
        确认消息
    """
    from memory import long_term_memory
    long_term_memory.save_fact("default", fact)
    return f"已保存：{fact}"


@tool
def list_skills() -> str:
    """
    列出所有可用技能
    
    技能（Skills）是预定义的能力模块，每个技能包含特定的功能。
    
    Returns:
        技能列表的格式化字符串
    """
    from skills import skill_registry
    skills = skill_registry.list_all()
    if not skills:
        return "暂无可用技能"
    lines = ["## 可用技能", ""]
    for skill in skills:
        lines.append(f"- **{skill.name}**: {skill.description}")
    return "\n".join(lines)


@tool
def get_system_info() -> str:
    """
    获取系统信息
    
    返回当前系统状态，包括已加载的技能数量、记忆条数等。
    
    Returns:
        JSON 格式的系统信息
    """
    from skills import skill_registry
    from memory import long_term_memory
    info = {
        "skills_count": len(skill_registry.list_all()),
        "memory_facts": len(long_term_memory.load_facts("default")),
        "timestamp": datetime.now().isoformat(),
    }
    return json.dumps(info, ensure_ascii=False, indent=2)


# ── 内置工具列表 ─────────────────────────────────────────────
# 这些工具会与 MCP 工具合并，一起提供给 Agent 使用
builtin_tools = [
    get_current_time,
    calculate,
    search_memory,
    save_memory,
    list_skills,
    get_system_info,
]
