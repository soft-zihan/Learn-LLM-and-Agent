# 单Agent生产级最佳实践

> 📅 **更新时间**: 2026-06-17  

---

## 目录

- [1. 生产环境架构设计](#1-生产环境架构设计)
- [2. 错误处理与恢复](#2-错误处理与恢复)
- [3. 性能优化](#3-性能优化)
- [4. 安全最佳实践](#4-安全最佳实践)
- [5. 部署与运维](#5-部署与运维)
- [6. 监控与告警](#6-监控与告警)
- [7. 测试策略](#7-测试策略)
- [8. 参考资料](#8-参考资料)

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

> 📅 **最后更新**: 2026-06
> 📊 **难度等级**: Level 4-5（高级-专家）
> 🔗 **相关文档**: 
> - [Agent 工具集成](./01-Agent工具集成实战.md)
> - [单 Agent 开发框架](../01-单Agent开发框架/README.md)
> - [生产部署指南](../../04-推理与部署/02-生产部署/README.md)
