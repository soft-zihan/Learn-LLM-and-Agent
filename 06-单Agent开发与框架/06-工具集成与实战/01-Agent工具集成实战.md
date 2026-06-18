# Agent工具集成实战

> 📅 **更新时间**: 2026-06-17  

---

## 目录

- [1. 工具调用基础](#1-工具调用基础)
- [2. 工具实现模式](#2-工具实现模式)
- [3. 工具调用流程](#3-工具调用流程)
- [4. 错误处理与重试](#4-错误处理与重试)
- [5. 工具编排与组合](#5-工具编排与组合)
- [6. 工具测试与调试](#6-工具测试与调试)
- [7. 性能优化](#7-性能优化)
- [8. 参考资料](#8-参考资料)

---

## 1. 工具调用基础

### 1.1 工具定义与注册

工具调用是 Agent 系统的核心能力,允许 LLM 通过调用外部工具扩展其功能边界。

#### OpenAI 工具格式

```python
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import json

class ToolParameter(BaseModel):
    """工具参数定义"""
    type: str = Field(..., description="参数类型")
    description: str = Field(..., description="参数描述")
    enum: Optional[List[str]] = None
    required: bool = Field(False, description="是否必需")

class ToolDefinition(BaseModel):
    """工具定义"""
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    parameters: Dict[str, Any] = Field(..., description="参数 schema")

class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools: Dict[str, Dict] = {}
        self.functions: Dict[str, callable] = {}
    
    def register(self, tool_def: ToolDefinition, func: callable):
        """注册工具"""
        self.tools[tool_def.name] = tool_def.dict()
        self.functions[tool_def.name] = func
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict]:
        """获取工具 schema"""
        return self.tools.get(tool_name)
    
    def get_all_schemas(self) -> List[Dict]:
        """获取所有工具 schema"""
        return list(self.tools.values())
    
    async def execute_tool(self, tool_name: str, arguments: Dict) -> Any:
        """执行工具"""
        if tool_name not in self.functions:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        func = self.functions[tool_name]
        return await func(**arguments)

# 示例工具注册
registry = ToolRegistry()

# 定义搜索引擎工具
search_tool = ToolDefinition(
    name="web_search",
    description="Search the web for current information",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5
            },
            "language": {
                "type": "string",
                "description": "Search language",
                "enum": ["en", "zh", "es", "fr"]
            }
        },
        "required": ["query"]
    }
)

async def web_search(query: str, num_results: int = 5, language: str = "en"):
    """执行网络搜索"""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.search.com/search",
            params={"q": query, "num": num_results, "lang": language}
        ) as response:
            results = await response.json()
            return {
                "query": query,
                "results": results[:num_results],
                "total_results": len(results)
            }

registry.register(search_tool, web_search)

# 定义计算器工具
calculator_tool = ToolDefinition(
    name="calculator",
    description="Perform mathematical calculations",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
)

async def calculator(expression: str):
    """执行计算"""
    try:
        # 安全计算（限制可用函数）
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": lambda x: x ** 0.5
        }
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}

registry.register(calculator_tool, calculator)
```

### 1.2 工具选择策略

```python
class ToolSelector:
    """工具选择器"""
    
    def __init__(self, llm_client, tool_registry: ToolRegistry):
        self.llm = llm_client
        self.registry = tool_registry
    
    async def select_tools(self, user_input: str, max_tools: int = 3) -> List[str]:
        """选择合适的工具"""
        # 获取所有可用工具
        all_tools = self.registry.get_all_schemas()
        
        # 让 LLM 选择工具
        prompt = f"""
Given the user request, select the most appropriate tools from the available ones.

User request: {user_input}

Available tools:
{json.dumps(all_tools, indent=2)}

Return a JSON array of tool names (maximum {max_tools}):
"""
        
        response = await self.llm.generate(prompt, max_tokens=200)
        
        try:
            selected = json.loads(response)
            return selected[:max_tools]
        except:
            # 如果解析失败,返回所有工具
            return [t["name"] for t in all_tools[:max_tools]]
    
    async def generate_with_tools(
        self,
        messages: List[Dict],
        temperature: float = 0.7
    ) -> Dict:
        """使用工具生成"""
        # 获取工具 schema
        tools = self.registry.get_all_schemas()
        
        # 调用 LLM
        response = await self.llm.chat_completion(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature
        )
        
        return response

# 使用示例
async def tool_selection_example():
    """工具选择示例"""
    selector = ToolSelector(llm_client, registry)
    
    user_input = "What is the current price of Apple stock and calculate 15% increase?"
    
    # 选择工具
    selected_tools = await selector.select_tools(user_input)
    print(f"Selected tools: {selected_tools}")
    # 输出: ['web_search', 'calculator']
    
    # 生成响应
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": user_input}
    ]
    
    response = await selector.generate_with_tools(messages)
    print(f"Response: {response}")
```

## 2. 工具实现模式

### 2.1 同步工具

```python
class SynchronousTools:
    """同步工具实现"""
    
    @staticmethod
    def get_current_time() -> str:
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    @staticmethod
    def read_file(file_path: str) -> str:
        """读取文件内容"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def write_file(file_path: str, content: str) -> Dict:
        """写入文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"status": "success", "file": file_path}
    
    @staticmethod
    def list_directory(path: str) -> List[str]:
        """列出目录"""
        import os
        return os.listdir(path)
    
    @staticmethod
    def database_query(query: str, params: Dict = None) -> List[Dict]:
        """执行数据库查询"""
        import sqlite3
        
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        conn.close()
        
        return [dict(zip(columns, row)) for row in results]

# 注册同步工具
sync_tools = SynchronousTools()

for tool_name in ['get_current_time', 'read_file', 'write_file', 'list_directory']:
    tool_def = ToolDefinition(
        name=tool_name,
        description=f"Execute {tool_name}",
        parameters={"type": "object", "properties": {}}
    )
    registry.register(tool_def, getattr(sync_tools, tool_name))
```

### 2.2 异步工具

```python
import aiohttp
import asyncio

class AsynchronousTools:
    """异步工具实现"""
    
    @staticmethod
    async def fetch_url(url: str, method: str = "GET") -> Dict:
        """获取 URL 内容"""
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url) as response:
                return {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "body": await response.text()
                }
    
    @staticmethod
    async def send_email(
        to: str,
        subject: str,
        body: str,
        html: bool = False
    ) -> Dict:
        """发送邮件"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = 'agent@example.com'
        msg['To'] = to
        
        if html:
            msg.attach(MIMEText(body, 'html'))
        else:
            msg.attach(MIMEText(body, 'plain'))
        
        # 发送邮件（异步）
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: smtplib.SMTP('localhost').send_message(msg)
        )
        
        return {"status": "sent", "to": to}
    
    @staticmethod
    async def call_api(
        endpoint: str,
        method: str = "POST",
        data: Dict = None,
        headers: Dict = None
    ) -> Dict:
        """调用外部 API"""
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                endpoint,
                json=data,
                headers=headers
            ) as response:
                return {
                    "status": response.status,
                    "data": await response.json()
                }
    
    @staticmethod
    async def run_command(command: str, timeout: int = 30) -> Dict:
        """执行 shell 命令"""
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode(),
                "stderr": stderr.decode()
            }
        except asyncio.TimeoutError:
            process.kill()
            return {"error": "Command timed out"}

# 注册异步工具
async_tools = AsynchronousTools()

for tool_name in ['fetch_url', 'send_email', 'call_api', 'run_command']:
    tool_def = ToolDefinition(
        name=tool_name,
        description=f"Execute {tool_name}",
        parameters={"type": "object", "properties": {}}
    )
    registry.register(tool_def, getattr(async_tools, tool_name))
```

### 2.3 带状态的工具

```python
class StatefulTool:
    """带状态的工具"""
    
    def __init__(self):
        self.conversation_memory = []
        self.user_preferences = {}
        self.session_data = {}
    
    def remember(self, key: str, value: Any) -> Dict:
        """记住信息"""
        self.session_data[key] = value
        return {"status": "remembered", "key": key}
    
    def recall(self, key: str) -> Any:
        """回忆信息"""
        return self.session_data.get(key)
    
    def update_preferences(self, user_id: str, preferences: Dict) -> Dict:
        """更新用户偏好"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        self.user_preferences[user_id].update(preferences)
        return {"status": "updated", "preferences": self.user_preferences[user_id]}
    
    def get_preferences(self, user_id: str) -> Dict:
        """获取用户偏好"""
        return self.user_preferences.get(user_id, {})
    
    def add_to_memory(self, message: str, metadata: Dict = None) -> Dict:
        """添加到记忆"""
        self.conversation_memory.append({
            "message": message,
            "metadata": metadata or {},
            "timestamp": time.time()
        })
        return {"status": "added", "memory_size": len(self.conversation_memory)}
    
    def search_memory(self, query: str, limit: int = 5) -> List[Dict]:
        """搜索记忆"""
        # 简单关键词匹配
        results = []
        for item in self.conversation_memory:
            if query.lower() in item["message"].lower():
                results.append(item)
                if len(results) >= limit:
                    break
        
        return results

# 注册状态工具
stateful_tool = StatefulTool()

for tool_name in ['remember', 'recall', 'update_preferences', 'get_preferences', 
                  'add_to_memory', 'search_memory']:
    tool_def = ToolDefinition(
        name=tool_name,
        description=f"Execute {tool_name}",
        parameters={"type": "object", "properties": {}}
    )
    registry.register(tool_def, getattr(stateful_tool, tool_name))
```

## 3. 工具调用流程

### 3.1 ReAct 模式

ReAct（Reasoning + Acting）是最常用的工具调用模式。

```python
class ReActAgent:
    """ReAct Agent"""
    
    def __init__(self, llm_client, tool_registry: ToolRegistry):
        self.llm = llm_client
        self.registry = tool_registry
        self.max_iterations = 10
    
    async def run(self, query: str) -> str:
        """执行 ReAct 循环"""
        messages = [
            {
                "role": "system",
                "content": """You are an AI assistant that can use tools to help users.
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Available tools:
{tools}
""".format(
                    tool_names=", ".join(self.registry.tools.keys()),
                    tools=json.dumps(self.registry.get_all_schemas(), indent=2)
                )
            },
            {"role": "user", "content": f"Question: {query}"}
        ]
        
        for iteration in range(self.max_iterations):
            # 生成下一步
            response = await self.llm.chat_completion(
                messages=messages,
                stop=["Observation:"],
                temperature=0
            )
            
            content = response.choices[0].message.content
            
            # 解析 Thought/Action/Action Input
            thought, action, action_input = self._parse_response(content)
            
            if action is None:
                # 没有 Action,返回 Final Answer
                return self._extract_final_answer(content)
            
            # 执行工具
            try:
                observation = await self.registry.execute_tool(
                    action,
                    json.loads(action_input)
                )
                observation_str = json.dumps(observation, ensure_ascii=False)
            except Exception as e:
                observation_str = f"Error: {str(e)}"
            
            # 添加到消息历史
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation_str}"
            })
        
        return "Maximum iterations reached"
    
    def _parse_response(self, content: str) -> tuple:
        """解析响应"""
        import re
        
        thought_match = re.search(r'Thought:\s*(.*?)$', content, re.MULTILINE)
        action_match = re.search(r'Action:\s*(.*?)$', content, re.MULTILINE)
        action_input_match = re.search(r'Action Input:\s*(.*?)$', content, re.MULTILINE)
        
        thought = thought_match.group(1) if thought_match else None
        action = action_match.group(1) if action_match else None
        action_input = action_input_match.group(1) if action_input_match else None
        
        return thought, action, action_input
    
    def _extract_final_answer(self, content: str) -> str:
        """提取最终答案"""
        import re
        
        match = re.search(r'Final Answer:\s*(.*)', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content

# 使用示例
async def react_agent_example():
    """ReAct Agent 示例"""
    agent = ReActAgent(llm_client, registry)
    
    query = "What is the population of Tokyo and calculate 10% of it?"
    
    result = await agent.run(query)
    print(f"Result: {result}")
    
    # 执行过程：
    # Question: What is the population of Tokyo and calculate 10% of it?
    # Thought: I need to search for Tokyo's population
    # Action: web_search
    # Action Input: {"query": "Tokyo population 2024"}
    # Observation: {"results": [{"title": "...", "snippet": "14 million people"}]}
    # Thought: Now I need to calculate 10% of 14 million
    # Action: calculator
    # Action Input: {"expression": "14000000 * 0.1"}
    # Observation: {"result": 1400000.0}
    # Thought: I now know the final answer
    # Final Answer: The population of Tokyo is approximately 14 million, and 10% of it is 1.4 million.
```

### 3.2 工具调用对比表

| 模式 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| ReAct | 可解释性强、灵活 | 需要多次 LLM 调用 | 复杂推理任务 |
| Function Calling | 原生支持、简单 | 受限于模型能力 | 简单工具调用 |
| Plan-and-Execute | 可并行执行 | 规划可能不准确 | 多步骤任务 |
| Reflexion | 自我修正、高质量 | 计算成本高 | 高精度要求场景 |

## 4. 错误处理与重试

### 4.1 错误处理策略

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class ToolExecutor:
    """工具执行器"""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.registry = tool_registry
        self.error_counts = {}
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError))
    )
    async def execute_with_retry(self, tool_name: str, arguments: Dict) -> Any:
        """带重试的执行"""
        try:
            result = await self.registry.execute_tool(tool_name, arguments)
            return result
        except Exception as e:
            self.error_counts[tool_name] = self.error_counts.get(tool_name, 0) + 1
            raise
    
    async def execute_with_fallback(
        self,
        tool_name: str,
        arguments: Dict,
        fallback_tool: str = None
    ) -> Any:
        """带降级策略的执行"""
        try:
            return await self.execute_with_retry(tool_name, arguments)
        except Exception as e:
            print(f"Tool {tool_name} failed: {e}")
            
            if fallback_tool:
                print(f"Falling back to {fallback_tool}")
                return await self.execute_with_retry(fallback_tool, arguments)
            
            # 返回错误信息
            return {
                "error": str(e),
                "tool": tool_name,
                "fallback_used": fallback_tool is not None
            }
    
    def get_error_stats(self) -> Dict:
        """获取错误统计"""
        return {
            "error_counts": self.error_counts,
            "total_errors": sum(self.error_counts.values())
        }

# 使用示例
async def error_handling_example():
    """错误处理示例"""
    executor = ToolExecutor(registry)
    
    # 正常执行
    result1 = await executor.execute_with_retry("calculator", {"expression": "2+2"})
    print(f"Result: {result1}")
    
    # 带降级执行
    result2 = await executor.execute_with_fallback(
        "web_search",
        {"query": "test"},
        fallback_tool="calculator"  # 降级工具
    )
    print(f"Result with fallback: {result2}")
    
    # 错误统计
    stats = executor.get_error_stats()
    print(f"Error stats: {stats}")
```

### 4.2 超时控制

```python
import asyncio

class TimeoutHandler:
    """超时处理器"""
    
    @staticmethod
    async def execute_with_timeout(
        coro,
        timeout: float,
        default_value: Any = None
    ) -> Any:
        """带超时执行"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            print(f"Operation timed out after {timeout} seconds")
            return default_value
    
    @staticmethod
    async def execute_with_circuit_breaker(
        tool_name: str,
        func: callable,
        *args,
        failure_threshold: int = 5,
        recovery_timeout: float = 60,
        **kwargs
    ) -> Any:
        """带熔断器执行"""
        # 简化版熔断器
        if not hasattr(execute_with_circuit_breaker, 'states'):
            execute_with_circuit_breaker.states = {}
        
        state = execute_with_circuit_breaker.states.get(tool_name, {
            "failures": 0,
            "last_failure": 0,
            "state": "closed"  # closed, open, half-open
        })
        
        current_time = time.time()
        
        # 检查熔断器状态
        if state["state"] == "open":
            if current_time - state["last_failure"] < recovery_timeout:
                raise Exception(f"Circuit breaker open for {tool_name}")
            else:
                state["state"] = "half-open"
        
        try:
            result = await func(*args, **kwargs)
            
            # 成功,重置
            if state["state"] == "half-open":
                state["state"] = "closed"
                state["failures"] = 0
            
            return result
        
        except Exception as e:
            state["failures"] += 1
            state["last_failure"] = current_time
            
            if state["failures"] >= failure_threshold:
                state["state"] = "open"
                print(f"Circuit breaker opened for {tool_name}")
            
            raise
        finally:
            execute_with_circuit_breaker.states[tool_name] = state
```

## 5. 工具编排与组合

### 5.1 工具链

```python
class ToolChain:
    """工具链"""
    
    def __init__(self):
        self.steps = []
    
    def add_step(self, tool_name: str, arguments: Dict, condition: callable = None):
        """添加步骤"""
        self.steps.append({
            "tool": tool_name,
            "arguments": arguments,
            "condition": condition
        })
    
    async def execute(self, context: Dict = None) -> Dict:
        """执行工具链"""
        if context is None:
            context = {}
        
        results = []
        
        for step in self.steps:
            # 检查条件
            if step["condition"] and not step["condition"](context):
                continue
            
            # 替换参数中的变量
            arguments = self._substitute_variables(step["arguments"], context)
            
            # 执行工具
            result = await registry.execute_tool(step["tool"], arguments)
            
            # 更新上下文
            context[f"result_{step['tool']}"] = result
            results.append(result)
        
        return {"results": results, "context": context}
    
    def _substitute_variables(self, arguments: Dict, context: Dict) -> Dict:
        """替换变量"""
        import re
        
        substituted = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                # 替换 {variable} 格式的变量
                value = re.sub(
                    r'\{(\w+)\}',
                    lambda m: str(context.get(m.group(1), m.group(0))),
                    value
                )
            substituted[key] = value
        
        return substituted

# 使用示例
async def tool_chain_example():
    """工具链示例"""
    chain = ToolChain()
    
    # 搜索 -> 提取 -> 计算
    chain.add_step("web_search", {"query": "{query}"})
    chain.add_step(
        "extract_info",
        {"data": "{result_web_search}", "field": "population"}
    )
    chain.add_step(
        "calculator",
        {"expression": "{result_extract_info} * 0.1"}
    )
    
    result = await chain.execute({"query": "Tokyo population"})
    print(f"Tool chain result: {result}")
```

### 5.2 并行工具调用

```python
class ParallelToolExecutor:
    """并行工具执行器"""
    
    def __init__(self, max_concurrent: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_parallel(
        self,
        tool_calls: List[tuple]
    ) -> List[Any]:
        """并行执行多个工具调用"""
        tasks = []
        
        for tool_name, arguments in tool_calls:
            task = self._execute_with_semaphore(tool_name, arguments)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def _execute_with_semaphore(
        self,
        tool_name: str,
        arguments: Dict
    ) -> Any:
        """带信号量执行"""
        async with self.semaphore:
            return await registry.execute_tool(tool_name, arguments)

# 使用示例
async def parallel_execution_example():
    """并行执行示例"""
    executor = ParallelToolExecutor(max_concurrent=3)
    
    # 并行搜索多个查询
    tool_calls = [
        ("web_search", {"query": "Python programming"}),
        ("web_search", {"query": "Machine learning"}),
        ("web_search", {"query": "Data science"}),
        ("web_search", {"query": "Artificial intelligence"}),
    ]
    
    results = await executor.execute_parallel(tool_calls)
    
    for i, result in enumerate(results):
        print(f"Result {i}: {result}")
```

## 6. 工具测试与调试

### 6.1 单元测试

```python
import pytest

class TestTools:
    """工具测试"""
    
    @pytest.mark.asyncio
    async def test_calculator_tool(self):
        """测试计算器工具"""
        result = await calculator("2 + 2")
        assert result["result"] == 4
        
        result = await calculator("10 * 5")
        assert result["result"] == 50
    
    @pytest.mark.asyncio
    async def test_web_search_tool(self):
        """测试搜索引擎工具"""
        result = await web_search("Python", num_results=3)
        assert "query" in result
        assert "results" in result
        assert len(result["results"]) <= 3
    
    @pytest.mark.asyncio
    async def test_tool_registry(self):
        """测试工具注册表"""
        registry_test = ToolRegistry()
        
        tool_def = ToolDefinition(
            name="test_tool",
            description="Test tool",
            parameters={"type": "object", "properties": {}}
        )
        
        async def test_func():
            return {"status": "ok"}
        
        registry_test.register(tool_def, test_func)
        
        # 测试执行
        result = await registry_test.execute_tool("test_tool", {})
        assert result["status"] == "ok"
    
    @pytest.mark.asyncio
    async def test_react_agent(self):
        """测试 ReAct Agent"""
        agent = ReActAgent(mock_llm_client, registry)
        
        result = await agent.run("What is 2+2?")
        assert "4" in result or "four" in result.lower()

def test_tool_chain():
    """测试工具链"""
    chain = ToolChain()
    chain.add_step("calculator", {"expression": "2+2"})
    
    # 这里需要异步测试
    # 使用 pytest-asyncio
    pass
```

### 6.2 集成测试

```python
class IntegrationTester:
    """集成测试器"""
    
    def __init__(self, agent):
        self.agent = agent
    
    async def run_test_suite(self):
        """运行测试套件"""
        test_cases = [
            {
                "name": "Simple calculation",
                "input": "What is 15% of 200?",
                "expected_tools": ["calculator"],
                "expected_keywords": ["30"]
            },
            {
                "name": "Web search",
                "input": "What is the latest news about AI?",
                "expected_tools": ["web_search"],
                "expected_keywords": ["AI", "news"]
            },
            {
                "name": "Multi-step task",
                "input": "Find the population of Paris and calculate 20% of it",
                "expected_tools": ["web_search", "calculator"],
                "expected_keywords": ["Paris", "population"]
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            result = await self.run_test_case(test_case)
            results.append(result)
        
        return results
    
    async def run_test_case(self, test_case: Dict) -> Dict:
        """运行单个测试用例"""
        start_time = time.time()
        
        try:
            output = await self.agent.run(test_case["input"])
            
            # 验证结果
            passed = all(
                keyword.lower() in output.lower()
                for keyword in test_case["expected_keywords"]
            )
            
            return {
                "name": test_case["name"],
                "passed": passed,
                "output": output,
                "execution_time": time.time() - start_time
            }
        
        except Exception as e:
            return {
                "name": test_case["name"],
                "passed": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }

# 运行测试
async def run_integration_tests():
    """运行集成测试"""
    agent = ReActAgent(llm_client, registry)
    tester = IntegrationTester(agent)
    
    results = await tester.run_test_suite()
    
    print("Test Results:")
    for result in results:
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"{status}: {result['name']} ({result['execution_time']:.2f}s)")
```

## 7. 性能优化

### 7.1 工具调用优化

```python
class ToolCallOptimizer:
    """工具调用优化器"""
    
    def __init__(self):
        self.cache = {}
        self.call_counts = {}
    
    def should_use_cache(self, tool_name: str, arguments: Dict) -> bool:
        """判断是否使用缓存"""
        # 幂等操作可以缓存
        idempotent_tools = ["calculator", "web_search", "get_current_time"]
        return tool_name in idempotent_tools
    
    def get_cached_result(self, tool_name: str, arguments: Dict) -> Optional[Any]:
        """获取缓存结果"""
        cache_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
        return self.cache.get(cache_key)
    
    def cache_result(self, tool_name: str, arguments: Dict, result: Any, ttl: int = 3600):
        """缓存结果"""
        cache_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
        self.cache[cache_key] = {
            "result": result,
            "timestamp": time.time(),
            "ttl": ttl
        }
    
    async def execute_optimized(
        self,
        tool_name: str,
        arguments: Dict
    ) -> Any:
        """优化执行"""
        # 检查缓存
        if self.should_use_cache(tool_name, arguments):
            cached = self.get_cached_result(tool_name, arguments)
            if cached and time.time() - cached["timestamp"] < cached["ttl"]:
                return cached["result"]
        
        # 执行工具
        result = await registry.execute_tool(tool_name, arguments)
        
        # 缓存结果
        if self.should_use_cache(tool_name, arguments):
            self.cache_result(tool_name, arguments, result)
        
        # 更新统计
        self.call_counts[tool_name] = self.call_counts.get(tool_name, 0) + 1
        
        return result
    
    def get_optimization_stats(self) -> Dict:
        """获取优化统计"""
        return {
            "cache_size": len(self.cache),
            "call_counts": self.call_counts
        }

# 使用示例
async def optimization_example():
    """优化示例"""
    optimizer = ToolCallOptimizer()
    
    # 第一次调用（执行）
    result1 = await optimizer.execute_optimized(
        "calculator",
        {"expression": "2+2"}
    )
    print(f"Result 1: {result1}")
    
    # 第二次调用（缓存）
    result2 = await optimizer.execute_optimized(
        "calculator",
        {"expression": "2+2"}
    )
    print(f"Result 2: {result2}")
    
    # 统计
    stats = optimizer.get_optimization_stats()
    print(f"Stats: {stats}")
```

### 7.2 批量处理

```python
class BatchToolProcessor:
    """批量工具处理器"""
    
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
    
    async def process_batch(
        self,
        tool_name: str,
        argument_list: List[Dict]
    ) -> List[Any]:
        """批量处理"""
        results = []
        
        # 分批处理
        for i in range(0, len(argument_list), self.batch_size):
            batch = argument_list[i:i+self.batch_size]
            
            # 并行执行批次
            tasks = [
                registry.execute_tool(tool_name, args)
                for args in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        
        return results

# 使用示例
async def batch_processing_example():
    """批量处理示例"""
    processor = BatchToolProcessor(batch_size=5)
    
    # 批量计算
    calculations = [
        {"expression": f"{i} * {i}"}
        for i in range(1, 21)
    ]
    
    results = await processor.process_batch("calculator", calculations)
    
    for i, result in enumerate(results):
        print(f"Calculation {i+1}: {result}")
```

## 8. 参考资料

### 框架与库

- **LangChain Tools**: https://python.langchain.com/docs/modules/tools/
- **OpenAI Function Calling**: https://platform.openai.com/docs/guides/function-calling
- **CrewAI Tools**: https://docs.crewai.com/how-to/Create-a-New-Tool
- **AutoGPT Plugins**: https://docs.agpt.co/

### 最佳实践

1. 始终定义清晰的工具 schema
2. 实现完善的错误处理和重试机制
3. 使用缓存优化重复调用
4. 对工具进行单元测试和集成测试
5. 记录工具调用日志用于调试
6. 实施速率限制和超时控制
7. 定期审查和优化工具性能

---

> 📅 **最后更新**: 2025-06
> 📊 **难度等级**: Level 4（高级）
> 🔗 **相关文档**: 
> - [单 Agent 开发框架](../01-单Agent开发框架/README.md)
> - [MCP 协议](../../07-多Agent与Agent工程化/03-MCP协议与Skills/README.md)
> - [工具调用与 RAG](../../05-工具调用与RAG/README.md)
