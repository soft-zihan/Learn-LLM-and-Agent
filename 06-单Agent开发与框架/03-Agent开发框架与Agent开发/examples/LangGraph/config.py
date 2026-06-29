"""
公共配置模块 - 统一管理 API 配置和 LLM 实例

所有需要调用 LLM 的模块都应该从这里导入配置，避免重复代码。

使用方式：
    from config import API_KEY, BASE_URL, MODEL, get_llm
    
    # 或者直接使用预配置的 LLM 实例
    from config import llm
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载 .env 文件（包含 API 密钥等敏感配置）
load_dotenv()

# ── API 配置（从环境变量读取）──────────────────────────────
# 这些配置会被所有模块共享，修改 .env 文件即可全局生效

API_KEY: str = os.getenv("OPENAI_API_KEY", "")
BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")
MODEL: str = os.getenv("OPENAI_MODEL", "")


def get_llm(temperature: float = 0, **kwargs) -> ChatOpenAI:
    """
    获取配置好的 LLM 实例
    
    Args:
        temperature: 温度参数，控制输出的随机性（0=确定性，1=创造性）
        **kwargs: 其他传递给 ChatOpenAI 的参数
    
    Returns:
        配置好的 ChatOpenAI 实例
    
    Example:
        >>> llm = get_llm(temperature=0.7)
        >>> response = llm.invoke("你好")
    """
    return ChatOpenAI(
        model=MODEL,
        temperature=temperature,
        api_key=API_KEY,
        base_url=BASE_URL,
        **kwargs
    )


# 预配置的默认 LLM 实例（temperature=0，适合大多数场景）
llm = get_llm(temperature=0)
