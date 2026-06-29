"""
中间件系统 - 对应教程 09-中间件与治理

本模块实现了 Agent 的中间件链（Middleware Chain），
在 Agent 执行的前后插入自定义逻辑，实现横切关注点。

中间件类型：
1. LoggingMiddleware - 日志记录（请求追踪、性能监控）
2. SafetyMiddleware - 安全检查（拦截危险内容）
3. RateLimitMiddleware - 限流控制（防止滥用）
4. ContextMiddleware - 上下文注入（动态添加系统信息）

设计模式：
- 责任链模式：中间件按顺序执行，每个可以修改输入/输出
- 洋葱模型：请求从外到内穿过中间件，响应从内到外返回

教程对应：
- 09-中间件与治理：中间件架构设计
- 12-测试与可观测性：日志与追踪
"""

import time
from typing import Callable, Optional


class Middleware:
    """
    中间件基类
    
    定义了四个钩子（hook）方法，子类可以覆盖实现：
    - before_agent: Agent 处理前调用（可修改消息）
    - after_agent: Agent 处理后调用（可修改响应）
    - before_tool: 工具调用前调用（可拦截/修改参数）
    - after_tool: 工具调用后调用（可修改结果）
    """
    
    async def before_agent(self, messages: list, config: dict) -> list:
        """Agent 处理前的钩子，可修改消息列表"""
        return messages
    
    async def after_agent(self, response: str, config: dict) -> str:
        """Agent 处理后的钩子，可修改响应"""
        return response
    
    async def before_tool(self, tool_name: str, tool_args: dict, config: dict) -> Optional[dict]:
        """
        工具调用前的钩子
        返回 None 表示拦截此次调用，返回 dict 表示放行
        """
        return tool_args
    
    async def after_tool(self, tool_name: str, tool_result: str, config: dict) -> str:
        """工具调用后的钩子，可修改工具结果"""
        return tool_result


class LoggingMiddleware(Middleware):
    """
    日志中间件：记录所有 Agent 和工具调用
    
    用途：
    - 调试：追踪请求流程
    - 监控：统计调用频率
    - 审计：记录操作历史
    """
    
    def __init__(self):
        self._logs: list[dict] = []
    
    async def before_agent(self, messages: list, config: dict) -> list:
        self._log("AGENT_START", {"message_count": len(messages)})
        return messages
    
    async def after_agent(self, response: str, config: dict) -> str:
        self._log("AGENT_END", {"response_length": len(response)})
        return response
    
    async def before_tool(self, tool_name: str, tool_args: dict, config: dict) -> Optional[dict]:
        self._log("TOOL_CALL", {"tool": tool_name, "args": tool_args})
        return tool_args
    
    async def after_tool(self, tool_name: str, tool_result: str, config: dict) -> str:
        self._log("TOOL_RESULT", {"tool": tool_name})
        return tool_result
    
    def _log(self, event: str, data: dict):
        """记录日志事件"""
        self._logs.append({
            "timestamp": time.time(),
            "event": event,
            "data": data
        })
        print(f"[Middleware] {event}: {data}")
    
    def get_logs(self) -> list:
        """获取所有日志"""
        return self._logs


class SafetyMiddleware(Middleware):
    """
    安全中间件：拦截危险操作
    
    检查内容：
    - 用户输入中的危险关键词
    - 工具调用中的危险命令
    
    这是 Agent 安全的第一道防线（教程07.3：验证与护栏）
    """
    
    # 危险内容模式列表
    BLOCKED_PATTERNS = ["删除所有文件", "格式化硬盘", "发送密码"]
    
    async def before_agent(self, messages: list, config: dict) -> list:
        """检查用户输入是否包含危险内容"""
        last_message = messages[-1]["content"] if messages else ""
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in last_message:
                raise ValueError(f"检测到不安全的内容：{pattern}")
        return messages
    
    async def before_tool(self, tool_name: str, tool_args: dict, config: dict) -> Optional[dict]:
        """检查工具调用是否包含危险命令"""
        if tool_name == "execute_command":
            command = tool_args.get("command", "")
            # 拦截危险的系统命令
            if any(cmd in command for cmd in ["rm -rf", "format", "del /f"]):
                return None  # 返回 None 表示拦截
        return tool_args


class RateLimitMiddleware(Middleware):
    """
    限流中间件：控制工具调用频率
    
    使用滑动窗口算法：
    - 记录每次调用的时间戳
    - 清理窗口外的旧记录
    - 检查窗口内的调用次数是否超限
    
    防止 Agent 陷入工具调用死循环或被恶意滥用。
    """
    
    def __init__(self, max_calls: int = 10, window_seconds: int = 60):
        """
        Args:
            max_calls: 时间窗口内允许的最大调用次数
            window_seconds: 时间窗口大小（秒）
        """
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._calls: list[float] = []  # 调用时间戳列表
    
    async def before_tool(self, tool_name: str, tool_args: dict, config: dict) -> Optional[dict]:
        """检查是否超过调用频率限制"""
        now = time.time()
        # 滑动窗口：清理过期的时间戳
        self._calls = [t for t in self._calls if now - t < self.window_seconds]
        
        if len(self._calls) >= self.max_calls:
            raise ValueError(f"调用频率超限，请在 {self.window_seconds} 秒后重试")
        
        self._calls.append(now)
        return tool_args


class ContextMiddleware(Middleware):
    """
    上下文中间件：动态注入上下文信息
    
    在 Agent 处理前，将预设的上下文变量注入到消息列表中。
    这些信息会作为系统消息传递给 LLM。
    
    典型用途：
    - 注入当前时间
    - 注入用户身份
    - 注入环境信息
    """
    
    def __init__(self):
        self._context: dict[str, str] = {}
    
    def set_context(self, key: str, value: str):
        """设置上下文变量"""
        self._context[key] = value
    
    async def before_agent(self, messages: list, config: dict) -> list:
        """将上下文变量注入到消息列表"""
        if self._context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in self._context.items())
            context_msg = {"role": "system", "content": f"## 当前上下文\n{context_str}"}
            messages.insert(0, context_msg)
        return messages


class MiddlewareManager:
    """
    中间件管理器：统一管理中间件链
    
    负责：
    - 注册中间件
    - 按顺序执行中间件链
    - 处理中间件的短路（before_tool 返回 None 时终止）
    """
    
    def __init__(self):
        self._middlewares: list[Middleware] = []
    
    def add(self, middleware: Middleware):
        """注册一个中间件到链尾"""
        self._middlewares.append(middleware)
    
    async def run_before_agent(self, messages: list, config: dict) -> list:
        """依次执行所有中间件的 before_agent"""
        for mw in self._middlewares:
            messages = await mw.before_agent(messages, config)
        return messages
    
    async def run_after_agent(self, response: str, config: dict) -> str:
        """依次执行所有中间件的 after_agent"""
        for mw in self._middlewares:
            response = await mw.after_agent(response, config)
        return response
    
    async def run_before_tool(self, tool_name: str, tool_args: dict, config: dict) -> Optional[dict]:
        """
        依次执行所有中间件的 before_tool
        任一中间件返回 None，立即终止（短路）
        """
        for mw in self._middlewares:
            tool_args = await mw.before_tool(tool_name, tool_args, config)
            if tool_args is None:
                return None  # 中间件拦截
        return tool_args
    
    async def run_after_tool(self, tool_name: str, tool_result: str, config: dict) -> str:
        """依次执行所有中间件的 after_tool"""
        for mw in self._middlewares:
            tool_result = await mw.after_tool(tool_name, tool_result, config)
        return tool_result


# ── 全局中间件管理器实例 ─────────────────────────────────────

middleware_manager = MiddlewareManager()


def init_middleware():
    """
    初始化中间件系统（应用启动时调用）
    
    按顺序注册中间件：
    1. 日志（最先执行，记录所有操作）
    2. 安全（第二道防线）
    3. 限流（防止滥用）
    4. 上下文（最后执行，注入上下文）
    """
    middleware_manager.add(LoggingMiddleware())
    middleware_manager.add(SafetyMiddleware())
    middleware_manager.add(RateLimitMiddleware(max_calls=20))
    middleware_manager.add(ContextMiddleware())
    print("[Middleware] 中间件系统初始化完成")
