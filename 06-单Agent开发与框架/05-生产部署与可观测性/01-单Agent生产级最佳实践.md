# 单Agent生产级最佳实践

> 📅 **更新时间**: 2026-06-18  

---

## 目录

- [1. 生产环境架构设计](#1-生产环境架构设计)
- [2. 错误处理与恢复](#2-错误处理与恢复)
- [3. 性能优化](#3-性能优化)
- [4. 安全最佳实践](#4-安全最佳实践)
- [5. 部署与运维](#5-部署与运维)
- [6. 监控与告警](#6-监控与告警)
- [7. 测试策略](#7-测试策略)
- [8. 生产级进阶实践](#8-生产级进阶实践)
- [9. 参考资料](#9-参考资料)

---

## 1. 生产环境架构设计

### 1.1 核心架构模式

生产级 Agent 系统需要稳健的架构设计来确保可靠性、可扩展性和可维护性。

```python
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import logging

class AgentState(str, Enum):
    """Agent 状态"""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    ERROR = "error"
    STOPPED = "stopped"

class AgentPriority(str, Enum):
    """Agent 优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AgentConfig:
    """Agent 配置"""
    name: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.7
    max_retries: int = 3
    timeout: float = 60.0
    enable_caching: bool = True
    enable_logging: bool = True
    priority: AgentPriority = AgentPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentContext:
    """Agent 上下文"""
    agent_id: str
    session_id: str
    user_id: str
    state: AgentState = AgentState.IDLE
    start_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()

class ProductionAgent:
    """生产级 Agent"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.context: Optional[AgentContext] = None
        self.tools: Dict[str, Callable] = {}
        self.cache = {}
        self.logger = logging.getLogger(f"agent.{config.name}")
        
        # 指标
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0
        }
    
    async def initialize(self):
        """初始化 Agent"""
        self.logger.info(f"Initializing agent {self.config.name}")
        
        # 加载工具
        await self._load_tools()
        
        # 预热缓存
        if self.config.enable_caching:
            await self._warmup_cache()
        
        self.logger.info(f"Agent {self.config.name} initialized")
    
    async def _load_tools(self):
        """加载工具"""
        # 实现工具加载逻辑
        pass
    
    async def _warmup_cache(self):
        """预热缓存"""
        # 实现缓存预热
        pass
    
    async def run(self, task: Dict, context: AgentContext) -> Dict:
        """运行 Agent"""
        self.context = context
        context.state = AgentState.RUNNING
        
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # 验证输入
            self._validate_input(task)
            
            # 检查缓存
            if self.config.enable_caching:
                cached_result = await self._check_cache(task)
                if cached_result:
                    self.metrics["successful_requests"] += 1
                    return cached_result
            
            # 执行主要逻辑
            result = await self._execute(task)
            
            # 缓存结果
            if self.config.enable_caching:
                await self._cache_result(task, result)
            
            # 更新指标
            duration = time.time() - start_time
            self.metrics["successful_requests"] += 1
            self.metrics["total_time"] += duration
            
            context.state = AgentState.IDLE
            return result
        
        except Exception as e:
            self.logger.error(f"Agent error: {e}", exc_info=True)
            self.metrics["failed_requests"] += 1
            context.state = AgentState.ERROR
            
            # 重试逻辑
            if self.metrics["failed_requests"] <= self.config.max_retries:
                self.logger.info(f"Retrying... ({self.metrics['failed_requests']}/{self.config.max_retries})")
                return await self.run(task, context)
            
            raise
    
    def _validate_input(self, task: Dict):
        """验证输入"""
        if not task:
            raise ValueError("Task cannot be empty")
        
        # 实现具体的验证逻辑
        pass
    
    async def _check_cache(self, task: Dict) -> Optional[Dict]:
        """检查缓存"""
        cache_key = self._generate_cache_key(task)
        return self.cache.get(cache_key)
    
    async def _cache_result(self, task: Dict, result: Dict):
        """缓存结果"""
        cache_key = self._generate_cache_key(task)
        self.cache[cache_key] = {
            "result": result,
            "timestamp": time.time(),
            "ttl": 3600  # 1小时
        }
    
    def _generate_cache_key(self, task: Dict) -> str:
        """生成缓存键"""
        import hashlib
        import json
        
        task_str = json.dumps(task, sort_keys=True)
        return hashlib.md5(task_str.encode()).hexdigest()
    
    async def _execute(self, task: Dict) -> Dict:
        """执行主要逻辑（子类实现）"""
        raise NotImplementedError
    
    def get_metrics(self) -> Dict:
        """获取指标"""
        return {
            **self.metrics,
            "success_rate": self.metrics["successful_requests"] / max(self.metrics["total_requests"], 1) * 100,
            "avg_response_time": self.metrics["total_time"] / max(self.metrics["successful_requests"], 1)
        }
```

### 1.2 配置管理

```python
import yaml
import os
from pathlib import Path

class ConfigurationManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Dict] = {}
    
    def load_config(self, config_name: str) -> Dict:
        """加载配置"""
        if config_name in self.configs:
            return self.configs[config_name]
        
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # 合并环境变量
        config = self._merge_env_vars(config)
        
        # 验证配置
        self._validate_config(config)
        
        # 缓存
        self.configs[config_name] = config
        
        return config
    
    def _merge_env_vars(self, config: Dict) -> Dict:
        """合并环境变量"""
        merged = config.copy()
        
        for key, value in merged.items():
            if isinstance(value, str) and value.startswith("${"):
                env_var = value[2:-1]
                env_value = os.environ.get(env_var)
                if env_value:
                    merged[key] = env_value
            elif isinstance(value, dict):
                merged[key] = self._merge_env_vars(value)
        
        return merged
    
    def _validate_config(self, config: Dict):
        """验证配置"""
        required_fields = ["name", "model"]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")

# 示例配置文件
"""
# config/agent.yaml
name: research_agent
model: gpt-5.2
max_tokens: 4096
temperature: 0.7
max_retries: 3
timeout: 60.0

tools:
  - name: web_search
    enabled: true
    config:
      api_key: ${SEARCH_API_KEY}
      max_results: 10
  
  - name: calculator
    enabled: true

caching:
  enabled: true
  ttl: 3600
  max_size: 1000

logging:
  level: INFO
  format: json
  output: file

monitoring:
  enabled: true
  metrics_interval: 60
  alerting:
    enabled: true
    channels:
      - email
      - slack
"""
```

## 2. 错误处理与恢复

### 2.1 全面错误处理

```python
from functools import wraps
from typing import Type, Tuple

class AgentError(Exception):
    """Agent 基础错误"""
    def __init__(self, message: str, error_code: str = None, metadata: Dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.metadata = metadata or {}

class ToolExecutionError(AgentError):
    """工具执行错误"""
    pass

class ModelAPIError(AgentError):
    """模型 API 错误"""
    pass

class ValidationError(AgentError):
    """验证错误"""
    pass

class TimeoutError(AgentError):
    """超时错误"""
    pass

def retry_on_failure(
    max_retries: int = 3,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0
):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        wait_time = min(backoff_factor ** attempt, max_backoff)
                        logging.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator

def handle_errors(error_handler: Callable = None):
    """错误处理装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except AgentError as e:
                # 已知的 Agent 错误
                if error_handler:
                    await error_handler(e)
                raise
            except Exception as e:
                # 未知错误
                agent_error = AgentError(
                    message=str(e),
                    error_code="unexpected_error",
                    metadata={"original_exception": type(e).__name__}
                )
                
                if error_handler:
                    await error_handler(agent_error)
                
                raise agent_error
        
        return wrapper
    return decorator

# 使用示例
class RobustAgent(ProductionAgent):
    """健壮的 Agent"""
    
    @retry_on_failure(max_retries=3, retry_exceptions=(ModelAPIError, TimeoutError))
    @handle_errors()
    async def _execute(self, task: Dict) -> Dict:
        """执行任务（带错误处理）"""
        try:
            # 调用模型
            result = await self._call_model(task)
            
            # 验证结果
            self._validate_output(result)
            
            return result
        
        except ModelAPIError as e:
            self.logger.error(f"Model API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise ModelAPIError(str(e))
    
    @retry_on_failure(max_retries=2, backoff_factor=1.5)
    async def _call_model(self, task: Dict) -> Dict:
        """调用模型"""
        # 实现模型调用
        pass
```

### 2.2 优雅降级

```python
class GracefulDegradation:
    """优雅降级"""
    
    def __init__(self, agent: ProductionAgent):
        self.agent = agent
        self.fallback_strategies = []
    
    def add_fallback(self, condition: Callable, fallback_func: Callable):
        """添加降级策略"""
        self.fallback_strategies.append({
            "condition": condition,
            "fallback": fallback_func
        })
    
    async def execute_with_fallback(self, task: Dict, context: AgentContext) -> Dict:
        """执行带降级"""
        try:
            return await self.agent.run(task, context)
        except Exception as e:
            self.agent.logger.warning(f"Primary execution failed: {e}")
            
            # 尝试降级策略
            for strategy in self.fallback_strategies:
                if strategy["condition"](e):
                    try:
                        self.agent.logger.info(f"Using fallback strategy")
                        return await strategy["fallback"](task, context)
                    except Exception as fallback_error:
                        self.agent.logger.error(f"Fallback also failed: {fallback_error}")
            
            # 所有策略都失败
            raise

# 使用示例
async def setup_degradation():
    """设置降级策略"""
    agent = ProductionAgent(AgentConfig(name="test", model="gpt-5.2"))
    degradation = GracefulDegradation(agent)
    
    # 添加降级策略
    degradation.add_fallback(
        condition=lambda e: isinstance(e, ModelAPIError),
        fallback_func=use_smaller_model
    )
    
    degradation.add_fallback(
        condition=lambda e: isinstance(e, TimeoutError),
        fallback_func=use_cached_response
    )
    
    degradation.add_fallback(
        condition=lambda e: True,  # 捕获所有错误
        fallback_func=return_error_message
    )
    
    return degradation

async def use_smaller_model(task: Dict, context: AgentContext) -> Dict:
    """使用小模型降级"""
    # 切换到小模型
    return {"result": "Using smaller model", "degraded": True}

async def use_cached_response(task: Dict, context: AgentContext) -> Dict:
    """使用缓存响应"""
    return {"result": "Using cached response", "degraded": True}

async def return_error_message(task: Dict, context: AgentContext) -> Dict:
    """返回错误消息"""
    return {
        "error": "Service temporarily unavailable",
        "suggestion": "Please try again later"
    }
```

## 3. 性能优化

### 3.1 缓存策略

```python
import hashlib
import json
from functools import lru_cache
from typing import Any

class MultiLevelCache:
    """多级缓存"""
    
    def __init__(
        self,
        memory_size: int = 1000,
        redis_url: str = None,
        default_ttl: int = 3600
    ):
        # L1: 内存缓存
        self.memory_cache = {}
        self.memory_size = memory_size
        
        # L2: Redis 缓存
        self.redis_client = None
        if redis_url:
            import redis
            self.redis_client = redis.from_url(redis_url)
        
        self.default_ttl = default_ttl
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        # 检查 L1
        if key in self.memory_cache:
            cached = self.memory_cache[key]
            if time.time() - cached["timestamp"] < cached["ttl"]:
                return cached["value"]
            else:
                del self.memory_cache[key]
        
        # 检查 L2
        if self.redis_client:
            cached = self.redis_client.get(key)
            if cached:
                value = json.loads(cached)
                # 回填 L1
                self._set_memory(key, value)
                return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None):
        """设置缓存"""
        if ttl is None:
            ttl = self.default_ttl
        
        # 设置 L1
        self._set_memory(key, value, ttl)
        
        # 设置 L2
        if self.redis_client:
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(value)
            )
    
    def _set_memory(self, key: str, value: Any, ttl: int = None):
        """设置内存缓存"""
        if len(self.memory_cache) >= self.memory_size:
            # 清理过期条目
            self._evict_expired()
            
            # 如果仍然满了，删除最旧的
            if len(self.memory_cache) >= self.memory_size:
                oldest_key = min(
                    self.memory_cache.keys(),
                    key=lambda k: self.memory_cache[k]["timestamp"]
                )
                del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl or self.default_ttl
        }
    
    def _evict_expired(self):
        """清理过期条目"""
        current_time = time.time()
        expired_keys = [
            key for key, cached in self.memory_cache.items()
            if current_time - cached["timestamp"] >= cached["ttl"]
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]

# 使用示例
cache = MultiLevelCache(
    memory_size=1000,
    redis_url="redis://localhost:6379",
    default_ttl=3600
)

async def cached_agent_execution(agent, task):
    """带缓存的 Agent 执行"""
    cache_key = generate_cache_key(task)
    
    # 检查缓存
    cached = await cache.get(cache_key)
    if cached:
        return cached
    
    # 执行
    result = await agent.run(task)
    
    # 缓存
    await cache.set(cache_key, result, ttl=1800)
    
    return result
```

### 3.2 批量处理优化

```python
class BatchProcessor:
    """批处理器"""
    
    def __init__(
        self,
        batch_size: int = 32,
        max_wait_time: float = 0.1,
        max_concurrent_batches: int = 5
    ):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_concurrent_batches = max_concurrent_batches
        
        self.task_queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
    
    async def submit(self, task: Dict) -> asyncio.Future:
        """提交任务"""
        future = asyncio.get_event_loop().create_future()
        await self.task_queue.put((task, future))
        return future
    
    async def process_loop(self, process_func: Callable):
        """处理循环"""
        while True:
            # 收集批次
            batch = []
            futures = []
            
            # 等待第一个任务
            task, future = await self.task_queue.get()
            batch.append(task)
            futures.append(future)
            
            # 收集更多任务
            start_time = time.time()
            
            while len(batch) < self.batch_size:
                time_elapsed = time.time() - start_time
                
                if time_elapsed >= self.max_wait_time:
                    break
                
                try:
                    task, future = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=self.max_wait_time - time_elapsed
                    )
                    batch.append(task)
                    futures.append(future)
                except asyncio.TimeoutError:
                    break
            
            # 处理批次
            async with self.semaphore:
                try:
                    results = await process_func(batch)
                    
                    for future, result in zip(futures, results):
                        if not future.done():
                            future.set_result(result)
                
                except Exception as e:
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)

# 使用示例
async def batch_processing_example():
    """批量处理示例"""
    processor = BatchProcessor(batch_size=10, max_wait_time=0.5)
    
    # 启动处理循环
    asyncio.create_task(processor.process_loop(mock_batch_process))
    
    # 提交任务
    futures = []
    for i in range(100):
        future = await processor.submit({"task": f"task_{i}"})
        futures.append(future)
    
    # 等待结果
    results = await asyncio.gather(*futures)
    
    return results
```

## 4. 安全最佳实践

### 4.1 输入验证与清理

```python
import re
from html import escape

class InputValidator:
    """输入验证器"""
    
    @staticmethod
    def validate_and_clean(text: str, max_length: int = 10000) -> str:
        """验证和清理文本输入"""
        if not text:
            raise ValueError("Input cannot be empty")
        
        # 长度限制
        if len(text) > max_length:
            text = text[:max_length]
        
        # 移除危险字符
        text = InputValidator._remove_dangerous_chars(text)
        
        # HTML 转义
        text = escape(text)
        
        # 规范化空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def _remove_dangerous_chars(text: str) -> str:
        """移除危险字符"""
        # 移除控制字符（保留常见空白）
        dangerous_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        return dangerous_pattern.sub('', text)
    
    @staticmethod
    def validate_tool_name(name: str) -> bool:
        """验证工具名称"""
        # 只允许字母、数字和下划线
        pattern = re.compile(r'^[a-zA-Z0-9_]+$')
        return bool(pattern.match(name))
    
    @staticmethod
    def validate_api_key(key: str) -> bool:
        """验证 API Key 格式"""
        # 示例：至少 32 个字符
        return len(key) >= 32

class PromptSecurity:
    """提示词安全"""
    
    @staticmethod
    def detect_injection_attempt(prompt: str) -> bool:
        """检测注入尝试"""
        injection_patterns = [
            r'ignore\s+previous',
            r'system\s*:',
            r'override\s+safety',
            r'debug\s+mode',
            r'print\s+training\s+data',
            r'drop\s+table',
            r'<script>',
            r'javascript:'
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        
        return False
    
    @staticmethod
    def sanitize_prompt(prompt: str) -> str:
        """清理提示词"""
        # 移除系统提示词覆盖尝试
        prompt = re.sub(r'system\s*:', '[USER]:', prompt, flags=re.IGNORECASE)
        
        # 移除 HTML/JS
        prompt = re.sub(r'<[^>]+>', '', prompt)
        prompt = re.sub(r'javascript:', '', prompt, flags=re.IGNORECASE)
        
        return prompt

# 使用示例
def secure_input_handling(user_input: str) -> str:
    """安全输入处理"""
    # 验证和清理
    cleaned = InputValidator.validate_and_clean(user_input)
    
    # 检查注入
    if PromptSecurity.detect_injection_attempt(cleaned):
        raise SecurityError("Potential injection attempt detected")
    
    # 清理提示词
    cleaned = PromptSecurity.sanitize_prompt(cleaned)
    
    return cleaned
```

### 4.2 速率限制

```python
from collections import defaultdict
import time

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """检查是否允许请求"""
        current_time = time.time()
        
        # 清理过期记录
        self.requests[client_id] = [
            timestamp for timestamp in self.requests[client_id]
            if current_time - timestamp < self.time_window
        ]
        
        # 检查限制
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # 记录请求
        self.requests[client_id].append(current_time)
        return True
    
    def get_remaining(self, client_id: str) -> int:
        """获取剩余请求数"""
        current_time = time.time()
        
        active_requests = len([
            timestamp for timestamp in self.requests[client_id]
            if current_time - timestamp < self.time_window
        ])
        
        return max(0, self.max_requests - active_requests)

# Token 桶算法
class TokenBucketRateLimiter:
    """Token 桶速率限制器"""
    
    def __init__(
        self,
        rate: float = 10.0,  # tokens per second
        capacity: int = 100
    ):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """消费 token"""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def _refill(self):
        """补充 token"""
        current_time = time.time()
        time_passed = current_time - self.last_refill
        
        new_tokens = time_passed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = current_time

# 使用示例
rate_limiter = RateLimiter(max_requests=100, time_window=60)

def check_rate_limit(client_id: str):
    """检查速率限制"""
    if not rate_limiter.is_allowed(client_id):
        remaining = rate_limiter.get_remaining(client_id)
        raise RateLimitError(
            f"Rate limit exceeded. Remaining: {remaining}"
        )
```

## 5. 部署与运维

### 5.1 Docker 部署

```dockerfile
# Dockerfile
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 创建非 root 用户
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  agent-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    depends_on:
      - redis
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  redis-data:
```

### 5.2 Kubernetes 部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-service
  labels:
    app: agent-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-service
  template:
    metadata:
      labels:
        app: agent-service
    spec:
      containers:
      - name: agent-service
        image: agent-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: openai-api-key
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: agent-service
spec:
  selector:
    app: agent-service
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 6. 监控与告警

### 6.1 关键指标

```python
from prometheus_client import Counter, Histogram, Gauge

class AgentMonitoring:
    """Agent 监控"""
    
    def __init__(self):
        # 请求指标
        self.requests_total = Counter(
            'agent_requests_total',
            'Total requests',
            ['agent_name', 'status']
        )
        
        # 延迟指标
        self.request_duration = Histogram(
            'agent_request_duration_seconds',
            'Request duration',
            ['agent_name'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # Token 指标
        self.tokens_used = Counter(
            'agent_tokens_total',
            'Total tokens used',
            ['type']
        )
        
        # 错误指标
        self.errors_total = Counter(
            'agent_errors_total',
            'Total errors',
            ['error_type']
        )
        
        # 缓存指标
        self.cache_hits = Counter(
            'agent_cache_hits_total',
            'Cache hits'
        )
        self.cache_misses = Counter(
            'agent_cache_misses_total',
            'Cache misses'
        )
        
        # 活动指标
        self.active_agents = Gauge(
            'agent_active_count',
            'Active agents'
        )
    
    def record_request(self, agent_name: str, status: str, duration: float):
        """记录请求"""
        self.requests_total.labels(agent_name, status).inc()
        self.request_duration.labels(agent_name).observe(duration)
    
    def record_tokens(self, token_count: int, token_type: str):
        """记录 Token"""
        self.tokens_used.labels(token_type).inc(token_count)
    
    def record_error(self, error_type: str):
        """记录错误"""
        self.errors_total.labels(error_type).inc()
    
    def record_cache_hit(self):
        """记录缓存命中"""
        self.cache_hits.inc()
    
    def record_cache_miss(self):
        """记录缓存未命中"""
        self.cache_misses.inc()

# 全局监控实例
monitoring = AgentMonitoring()
```

### 6.2 告警配置

```yaml
# prometheus-alerts.yaml
groups:
  - name: agent_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(agent_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate"
          description: "Error rate is {{ $value }}/s"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(agent_request_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency"
          description: "P95 latency is {{ $value }}s"
      
      - alert: LowCacheHitRate
        expr: rate(agent_cache_hits_total[5m]) / (rate(agent_cache_hits_total[5m]) + rate(agent_cache_misses_total[5m])) < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
```

## 7. 测试策略

### 7.1 测试金字塔

```python
import pytest
from unittest.mock import Mock, AsyncMock

class TestProductionAgent:
    """生产 Agent 测试"""
    
    @pytest.fixture
    def agent(self):
        """创建测试 Agent"""
        config = AgentConfig(name="test_agent", model="gpt-5.2")
        return ProductionAgent(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, agent):
        """测试初始化"""
        await agent.initialize()
        assert agent.config.name == "test_agent"
    
    @pytest.mark.asyncio
    async def test_successful_execution(self, agent):
        """测试成功执行"""
        task = {"query": "test"}
        context = AgentContext(
            agent_id="test",
            session_id="session_1",
            user_id="user_1"
        )
        
        result = await agent.run(task, context)
        
        assert result is not None
        assert context.state == AgentState.IDLE
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """测试错误处理"""
        task = {"invalid": True}
        context = AgentContext(
            agent_id="test",
            session_id="session_1",
            user_id="user_1"
        )
        
        with pytest.raises(Exception):
            await agent.run(task, context)
        
        assert context.state == AgentState.ERROR
    
    @pytest.mark.asyncio
    async def test_caching(self, agent):
        """测试缓存"""
        agent.config.enable_caching = True
        
        task = {"query": "cache_test"}
        context = AgentContext(
            agent_id="test",
            session_id="session_1",
            user_id="user_1"
        )
        
        # 第一次执行
        result1 = await agent.run(task, context)
        
        # 第二次执行（应该从缓存）
        result2 = await agent.run(task, context)
        
        assert result1 == result2
    
    def test_metrics(self, agent):
        """测试指标"""
        metrics = agent.get_metrics()
        
        assert "total_requests" in metrics
        assert "successful_requests" in metrics
        assert "failed_requests" in metrics
        assert "success_rate" in metrics
```

## 8. 参考资料

### 框架与工具

- **LangGraph**: 生产级 Agent 框架
- **CrewAI**: 多 Agent 协作框架
- **AutoGen**: Microsoft Agent 框架
- **FastAPI**: 高性能 API 框架

### 最佳实践总结

1. **错误处理**: 实施全面的错误处理和重试机制
2. **缓存**: 使用多级缓存优化性能
3. **安全**: 验证所有输入,防止注入攻击
4. **监控**: 收集关键指标,设置告警
5. **测试**: 多层次测试确保质量
6. **部署**: 使用容器化和编排工具
7. **配置**: 集中管理配置,支持环境变量
8. **日志**: 结构化日志,便于调试
9. **速率限制**: 保护服务免受过载
10. **文档**: 完善的文档和示例

---

## 8. 生产级进阶实践

> 📌 **说明**: 本章节来自原"工作流编排与LangGraph"文档的生产级最佳实践部分，已整合至本文件。
## 4. 生产级最佳实践

### 2.1 错误处理

#### 优雅降级

```python
# 优雅降级策略

from typing import Optional

class GracefulDegradation:
    """优雅降级管理器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.degradation_levels = [
            "full_functionality",  # 完整功能
            "limited_tools",       # 有限工具
            "llm_only",           # 仅 LLM
            "cached_response",    # 缓存响应
            "error_message"       # 错误消息
        ]
    
    def execute_with_degradation(self, agent_executor, user_input: str) -> dict:
        """带降级的执行"""
        # Level 0: 完整功能
        try:
            result = agent_executor.invoke({"input": user_input})
            return {
                "success": True,
                "result": result,
                "degradation_level": "full_functionality"
            }
        except Exception as e:
            logger.warning(f"完整功能失败：{e}")
        
        # Level 1: 有限工具（移除复杂工具）
        try:
            limited_executor = self._create_limited_executor()
            result = limited_executor.invoke({"input": user_input})
            return {
                "success": True,
                "result": result,
                "degradation_level": "limited_tools",
                "warning": "部分工具不可用"
            }
        except Exception as e:
            logger.warning(f"有限工具失败：{e}")
        
        # Level 2: 仅 LLM
        try:
            response = self.llm.invoke([HumanMessage(content=user_input)])
            return {
                "success": True,
                "result": {"output": response.content},
                "degradation_level": "llm_only",
                "warning": "工具不可用，仅使用 LLM"
            }
        except Exception as e:
            logger.error(f"LLM 失败：{e}")
        
        # Level 3: 缓存响应
        cached = self._get_cached_response(user_input)
        if cached:
            return {
                "success": True,
                "result": {"output": cached},
                "degradation_level": "cached_response",
                "warning": "使用缓存响应"
            }
        
        # Level 4: 错误消息
        return {
            "success": False,
            "result": {
                "output": "抱歉，服务暂时不可用，请稍后重试。"
            },
            "degradation_level": "error_message",
            "error": "所有降级策略都失败"
        }
    
    def _create_limited_executor(self):
        """创建有限工具执行器"""
        # 移除复杂工具，保留基础工具
        limited_tools = [tool for tool in tools if tool.name in ["search", "calculator"]]
        limited_agent = create_react_agent(llm, limited_tools, prompt)
        return AgentExecutor(agent=limited_agent, tools=limited_tools)
    
    def _get_cached_response(self, user_input: str) -> Optional[str]:
        """获取缓存响应"""
        # 实现缓存查找逻辑
        return None

# 使用示例
degradation_mgr = GracefulDegradation(llm)
result = degradation_mgr.execute_with_degradation(executor, "用户请求")

print(f"成功：{result['success']}")
print(f"降级级别：{result['degradation_level']}")
if result.get("warning"):
    print(f"警告：{result['warning']}")
```

#### 用户反馈

```python
# 用户反馈收集与处理

from typing import Optional

class FeedbackCollector:
    """反馈收集器"""
    
    def __init__(self):
        self.feedbacks: List[dict] = []
    
    def collect_feedback(
        self,
        task: str,
        response: str,
        rating: int,  # 1-5
        comment: str = "",
        user_id: str = ""
    ):
        """收集反馈"""
        feedback = {
            "task": task,
            "response": response,
            "rating": rating,
            "comment": comment,
            "user_id": user_id,
            "timestamp": time.time()
        }
        
        self.feedbacks.append(feedback)
        
        # 低评分触发告警
        if rating <= 2:
            self._trigger_alert(feedback)
    
    def get_feedback_stats(self) -> dict:
        """获取反馈统计"""
        if not self.feedbacks:
            return {}
        
        ratings = [f["rating"] for f in self.feedbacks]
        
        import numpy as np
        return {
            "total_feedbacks": len(self.feedbacks),
            "average_rating": float(np.mean(ratings)),
            "rating_distribution": {
                str(i): ratings.count(i) for i in range(1, 6)
            },
            "low_rating_rate": sum(1 for r in ratings if r <= 2) / len(ratings)
        }
    
    def get_improvement_suggestions(self) -> List[str]:
        """获取改进建议（从低评分反馈）"""
        low_feedbacks = [f for f in self.feedbacks if f["rating"] <= 2]
        
        if not low_feedbacks:
            return ["当前反馈都很积极"]
        
        # LLM 分析
        analysis_prompt = f"""
分析以下用户反馈，提取主要问题和改进建议：

{json.dumps(low_feedbacks[:10], ensure_ascii=False, indent=2)}

输出 JSON 数组：
["改进建议 1", "改进建议 2"]
"""
        
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except:
            return ["分析失败"]
    
    def _trigger_alert(self, feedback: dict):
        """触发告警"""
        logger.warning(
            f"收到低评分反馈：{feedback['rating']}星\n"
            f"任务：{feedback['task']}\n"
            f"评论：{feedback['comment']}"
        )

# 使用示例
feedback_collector = FeedbackCollector()

# 收集反馈
feedback_collector.collect_feedback(
    task="查询天气",
    response="北京晴天，25°C",
    rating=4,
    comment="回答简洁，但可以更多信息",
    user_id="user_001"
)

# 查看统计
stats = feedback_collector.get_feedback_stats()
print(f"平均评分：{stats.get('average_rating', 0):.2f}")

# 获取改进建议
suggestions = feedback_collector.get_improvement_suggestions()
print(f"改进建议：{suggestions}")
```

### 2.2 安全控制

#### 工具权限

```python
# 工具权限管理

from enum import Enum
from typing import Set

class PermissionLevel(Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    DANGEROUS = "dangerous"

class ToolPermissionManager:
    """工具权限管理器"""
    
    def __init__(self):
        self.tool_permissions: dict = {
            "search": PermissionLevel.READ,
            "calculator": PermissionLevel.READ,
            "weather": PermissionLevel.READ,
            "database_query": PermissionLevel.READ,
            "database_update": PermissionLevel.WRITE,
            "database_delete": PermissionLevel.ADMIN,
            "file_read": PermissionLevel.READ,
            "file_write": PermissionLevel.WRITE,
            "file_delete": PermissionLevel.DANGEROUS,
            "system_command": PermissionLevel.DANGEROUS
        }
        
        self.user_permissions: dict = {}  # 用户权限
    
    def set_user_permission(self, user_id: str, max_level: PermissionLevel):
        """设置用户权限"""
        self.user_permissions[user_id] = max_level
    
    def check_permission(self, user_id: str, tool_name: str) -> bool:
        """检查权限"""
        if tool_name not in self.tool_permissions:
            return False
        
        required_level = self.tool_permissions[tool_name]
        user_level = self.user_permissions.get(user_id, PermissionLevel.READ)
        
        # 权限等级排序
        permission_order = [
            PermissionLevel.READ,
            PermissionLevel.WRITE,
            PermissionLevel.ADMIN,
            PermissionLevel.DANGEROUS
        ]
        
        user_level_idx = permission_order.index(user_level)
        required_level_idx = permission_order.index(required_level)
        
        return user_level_idx >= required_level_idx
    
    def filter_tools_by_permission(self, user_id: str, tools: list) -> list:
        """根据权限过滤工具"""
        return [
            tool for tool in tools
            if self.check_permission(user_id, tool.name)
        ]

# 使用示例
permission_mgr = ToolPermissionManager()

# 设置用户权限
permission_mgr.set_user_permission("user_basic", PermissionLevel.READ)
permission_mgr.set_user_permission("user_admin", PermissionLevel.ADMIN)

# 检查权限
can_search = permission_mgr.check_permission("user_basic", "search")
can_delete = permission_mgr.check_permission("user_basic", "file_delete")

print(f"普通用户可搜索：{can_search}")
print(f"普通用户可删除：{can_delete}")
```

#### 输入验证

```python
# 输入验证与过滤

import re
from typing import Set

class InputValidator:
    """输入验证器"""
    
    def __init__(self):
        self.blocked_patterns = [
            r"DELETE\s+FROM",  # SQL 注入
            r"DROP\s+TABLE",
            r"<script>",  # XSS
            r"javascript:",
            r"rm\s+-rf",  # 危险命令
            r"sudo\s+"
        ]
        
        self.max_input_length = 1000
        self.blocked_keywords = [
            "password", "secret", "token",  # 敏感词
            "delete all", "drop database"
        ]
    
    def validate(self, user_input: str) -> dict:
        """验证输入"""
        issues = []
        
        # 长度检查
        if len(user_input) > self.max_input_length:
            issues.append(f"输入过长（最大 {self.max_input_length} 字符）")
        
        # 模式检查
        for pattern in self.blocked_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                issues.append(f"检测到危险模式：{pattern}")
        
        # 关键词检查
        input_lower = user_input.lower()
        for keyword in self.blocked_keywords:
            if keyword in input_lower:
                issues.append(f"检测到敏感词：{keyword}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "sanitized_input": self._sanitize(user_input)
        }
    
    def _sanitize(self, text: str) -> str:
        """清理输入"""
        # 移除危险字符
        text = re.sub(r'[<>"\']', '', text)
        # 限制长度
        return text[:self.max_input_length]

# 使用示例
validator = InputValidator()

# 正常输入
result1 = validator.validate("查询北京天气")
print(f"有效：{result1['valid']}")

# 恶意输入
result2 = validator.validate("DELETE FROM users; DROP TABLE")
print(f"有效：{result2['valid']}")
print(f"问题：{result2['issues']}")
```

#### 输出审核

```python
# 输出内容审核

class OutputModerator:
    """输出审核器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def moderate(self, output: str) -> dict:
        """审核输出"""
        prompt = f"""
请审核以下 AI 输出是否合适：

{output}

审核标准：
1. 是否包含不当内容（暴力、色情、歧视）
2. 是否泄露敏感信息
3. 是否有事实错误
4. 是否有潜在风险

输出 JSON：
```json
{{
    "safe": true/false,
    "issues": ["问题列表"],
    "confidence": 0.9,
    "suggestion": "修改建议"
}}
```
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except:
            return {
                "safe": True,
                "issues": [],
                "confidence": 0.5,
                "suggestion": ""
            }

# 使用示例
moderator = OutputModerator(llm)

output = "这是一个正常的回答"
moderation = moderator.moderate(output)

if not moderation["safe"]:
    print(f"输出不安全：{moderation['issues']}")
    print(f"建议：{moderation['suggestion']}")
```

### 2.3 性能优化

#### 缓存策略

```python
# 多级缓存策略

from typing import Optional
import time
import hashlib

class MultiLevelCache:
    """多级缓存"""
    
    def __init__(self):
        # L1：内存缓存（最快，容量小）
        self.l1_cache: dict = {}
        self.l1_max_size = 100
        self.l1_ttl = 300  # 5 分钟
        
        # L2：本地文件缓存（较快，容量中）
        self.l2_cache_dir = "./cache/l2"
        Path(self.l2_cache_dir).mkdir(parents=True, exist_ok=True)
        self.l2_ttl = 3600  # 1 小时
        
        # L3：远程缓存（慢，容量大）
        # 可以使用 Redis 等
        
        self.stats = {"l1_hits": 0, "l2_hits": 0, "misses": 0}
    
    def get(self, key: str) -> Optional[str]:
        """获取缓存"""
        # L1 查找
        if key in self.l1_cache:
            cache_data = self.l1_cache[key]
            if time.time() - cache_data["timestamp"] < self.l1_ttl:
                self.stats["l1_hits"] += 1
                return cache_data["value"]
            else:
                del self.l1_cache[key]
        
        # L2 查找
        l2_value = self._get_from_l2(key)
        if l2_value:
            self.stats["l2_hits"] += 1
            # 升级到 L1
            self._set_to_l1(key, l2_value)
            return l2_value
        
        self.stats["misses"] += 1
        return None
    
    def set(self, key: str, value: str):
        """设置缓存"""
        self._set_to_l1(key, value)
        self._set_to_l2(key, value)
    
    def _set_to_l1(self, key: str, value: str):
        """设置 L1 缓存"""
        # 检查大小限制
        if len(self.l1_cache) >= self.l1_max_size:
            # 删除最旧的
            oldest_key = min(
                self.l1_cache.keys(),
                key=lambda k: self.l1_cache[k]["timestamp"]
            )
            del self.l1_cache[oldest_key]
        
        self.l1_cache[key] = {
            "value": value,
            "timestamp": time.time()
        }
    
    def _get_from_l2(self, key: str) -> Optional[str]:
        """从 L2 获取"""
        file_path = f"{self.l2_cache_dir}/{self._hash_key(key)}.json"
        
        if not Path(file_path).exists():
            return None
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            if time.time() - data["timestamp"] < self.l2_ttl:
                return data["value"]
            else:
                Path(file_path).unlink()
                return None
        except:
            return None
    
    def _set_to_l2(self, key: str, value: str):
        """设置 L2 缓存"""
        file_path = f"{self.l2_cache_dir}/{self._hash_key(key)}.json"
        
        data = {
            "value": value,
            "timestamp": time.time()
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f)
    
    def _hash_key(self, key: str) -> str:
        """哈希键"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def get_stats(self) -> dict:
        """获取统计"""
        total = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["misses"]
        hit_rate = (self.stats["l1_hits"] + self.stats["l2_hits"]) / total if total > 0 else 0
        
        return {
            **self.stats,
            "total_requests": total,
            "hit_rate": hit_rate
        }

# 使用示例
cache = MultiLevelCache()

# 设置缓存
cache.set("query:天气", "北京晴天")

# 获取缓存
value = cache.get("query:天气")
print(f"缓存值：{value}")

# 查看统计
print(f"缓存统计：{cache.get_stats()}")
```

#### 并发控制

```python
# 并发控制与限流

import asyncio
from asyncio import Semaphore
from typing import Callable

class ConcurrencyController:
    """并发控制器"""
    
    def __init__(self, max_concurrent: int = 10, rate_limit: int = 100):
        self.semaphore = Semaphore(max_concurrent)
        self.rate_limit = rate_limit  # 每分钟请求数
        self.request_times: list = []
    
    async def execute_with_limit(self, func: Callable, *args, **kwargs):
        """限流执行"""
        # 并发控制
        async with self.semaphore:
            # 速率限制
            await self._wait_for_rate_limit()
            
            # 执行
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, *args, **kwargs)
    
    async def _wait_for_rate_limit(self):
        """等待速率限制"""
        now = time.time()
        
        # 清理 1 分钟前的记录
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # 检查是否超限
        if len(self.request_times) >= self.rate_limit:
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self.request_times.append(now)

# 使用示例
controller = ConcurrencyController(max_concurrent=5, rate_limit=60)

async def concurrent_execution():
    """并发执行示例"""
    async def task(i):
        async def mock_api():
            await asyncio.sleep(1)
            return f"结果 {i}"
        
        return await controller.execute_with_limit(mock_api)
    
    # 并发执行 20 个任务
    tasks = [task(i) for i in range(20)]
    results = await asyncio.gather(*tasks)
    
    print(f"完成 {len(results)} 个任务")

# asyncio.run(concurrent_execution())
```

### 2.4 可观测性

#### 链路追踪

```python
# 完整的链路追踪

from typing import Optional
import uuid

class TraceSpan:
    """追踪 Span"""
    def __init__(self, name: str, parent_id: Optional[str] = None):
        self.trace_id = str(uuid.uuid4())
        self.span_id = str(uuid.uuid4())
        self.parent_id = parent_id
        self.name = name
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.attributes: dict = {}
        self.status: str = "ok"
        self.error: Optional[str] = None
    
    def end(self):
        """结束 Span"""
        self.end_time = time.time()
    
    def duration(self) -> float:
        """获取持续时间"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration(),
            "attributes": self.attributes,
            "status": self.status,
            "error": self.error
        }

class Tracer:
    """链路追踪器"""
    
    def __init__(self):
        self.spans: List[TraceSpan] = []
        self.current_span: Optional[TraceSpan] = None
    
    def start_span(self, name: str) -> TraceSpan:
        """开始 Span"""
        parent_id = self.current_span.span_id if self.current_span else None
        span = TraceSpan(name, parent_id)
        self.spans.append(span)
        self.current_span = span
        return span
    
    def end_span(self):
        """结束当前 Span"""
        if self.current_span:
            self.current_span.end()
            # 回到父 Span
            parent_id = self.current_span.parent_id
            if parent_id:
                self.current_span = next(
                    (s for s in self.spans if s.span_id == parent_id),
                    None
                )
            else:
                self.current_span = None
    
    def visualize_trace(self, trace_id: str) -> str:
        """可视化链路"""
        spans = [s for s in self.spans if s.trace_id == trace_id]
        
        visualization = f"Trace: {trace_id}\n"
        visualization += "=" * 60 + "\n"
        
        for span in spans:
            indent = "  " * (self._get_depth(span))
            status_icon = "✓" if span.status == "ok" else "✗"
            
            visualization += f"{indent}{status_icon} {span.name}\n"
            visualization += f"{indent}  耗时：{span.duration()*1000:.2f}ms\n"
            
            if span.error:
                visualization += f"{indent}  错误：{span.error}\n"
        
        return visualization
    
    def _get_depth(self, span: TraceSpan) -> int:
        """获取 Span 深度"""
        depth = 0
        current = span
        while current.parent_id:
            depth += 1
            current = next(
                (s for s in self.spans if s.span_id == current.parent_id),
                None
            )
        return depth

# 使用示例
tracer = Tracer()

# 模拟链路
root_span = tracer.start_span("agent_execution")
llm_span = tracer.start_span("llm_call")
tracer.end_span()
tool_span = tracer.start_span("tool_execution")
tracer.end_span()
tracer.end_span()

# 可视化
print(tracer.visualize_trace(root_span.trace_id))
```

#### 告警机制

```python
# 智能告警系统

from typing import Callable

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alert_rules: list = []
        self.alert_history: list = []
    
    def add_rule(
        self,
        name: str,
        condition: Callable,
        severity: str = "warning",
        callback: Callable = None
    ):
        """添加告警规则"""
        self.alert_rules.append({
            "name": name,
            "condition": condition,
            "severity": severity,
            "callback": callback
        })
    
    def check_rules(self, metrics: dict):
        """检查规则"""
        for rule in self.alert_rules:
            if rule["condition"](metrics):
                self._trigger_alert(rule, metrics)
    
    def _trigger_alert(self, rule: dict, metrics: dict):
        """触发告警"""
        alert = {
            "rule": rule["name"],
            "severity": rule["severity"],
            "metrics": metrics,
            "timestamp": time.time()
        }
        
        self.alert_history.append(alert)
        
        # 记录日志
        logger.error(
            f"告警触发：{rule['name']} "
            f"（{rule['severity']}）"
        )
        
        # 执行回调
        if rule.get("callback"):
            rule["callback"](alert)
    
    def get_recent_alerts(self, hours: int = 24) -> list:
        """获取近期告警"""
        cutoff = time.time() - hours * 3600
        return [a for a in self.alert_history if a["timestamp"] > cutoff]

# 使用示例
alert_mgr = AlertManager()

# 添加规则
alert_mgr.add_rule(
    name="high_error_rate",
    condition=lambda m: m.get("error_rate", 0) > 0.1,
    severity="critical",
    callback=lambda alert: print(f"严重告警：{alert}")
)

alert_mgr.add_rule(
    name="slow_response",
    condition=lambda m: m.get("avg_response_time", 0) > 5.0,
    severity="warning"
)

# 检查规则
metrics = {
    "error_rate": 0.15,
    "avg_response_time": 6.5
}

alert_mgr.check_rules(metrics)
```

---

