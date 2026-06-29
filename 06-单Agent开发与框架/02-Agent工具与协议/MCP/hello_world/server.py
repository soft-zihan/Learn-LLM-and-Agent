"""
MCP Hello World Server - 最简示例

运行方式:
    python server.py

使用 MCP Inspector 测试:
    npx @modelcontextprotocol/inspector python server.py
"""

from fastmcp import FastMCP

# 创建 Server 实例
mcp = FastMCP(
    "hello-world",
    instructions="一个示例 MCP Server，提供基础的工具和资源",
)


# ── 工具定义 ──────────────────────────────────────────────


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True,
    }
)
def greet(name: str) -> str:
    """向某人问好

    Args:
        name: 要问候的人名
    """
    return f"Hello, {name}!"


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
    }
)
def add(a: int, b: int) -> int:
    """两个数相加

    Args:
        a: 第一个数
        b: 第二个数
    """
    return a + b


@mcp.tool
def get_weather(city: str) -> dict:
    """获取城市天气（模拟）

    Args:
        city: 城市名称
    """
    # 模拟数据，实际项目中调用天气 API
    return {
        "city": city,
        "temperature": "22°C",
        "condition": "晴",
        "humidity": "45%",
    }


# ── 资源定义 ──────────────────────────────────────────────


@mcp.resource("info://about")
def about() -> str:
    """关于本 Server"""
    return "Hello World MCP Server - 提供 greet、add、get_weather 工具"


@mcp.resource("config://app")
def app_config() -> str:
    """应用配置信息"""
    return """{
  "appName": "Hello World MCP",
  "version": "1.0.0",
  "features": ["greet", "add", "get_weather"]
}"""


# ── Prompt 模板 ───────────────────────────────────────────


@mcp.prompt()
def review_prompt(code: str) -> list[dict]:
    """生成代码审查提示

    Args:
        code: 待审查的代码
    """
    return [
        {
            "role": "system",
            "content": {"type": "text", "text": "你是一位资深代码审查专家。请审查以下代码并给出建议。"},
        },
        {
            "role": "user",
            "content": {"type": "text", "text": f"请审查以下代码:\n\n```python\n{code}\n```"},
        },
    ]


# ── 启动 ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Stdio 传输（默认，用于本地集成）
    mcp.run()

    # HTTP 传输（用于远程部署）
    # mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
