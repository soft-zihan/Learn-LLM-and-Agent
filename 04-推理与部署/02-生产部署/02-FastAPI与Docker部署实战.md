# FastAPI 与 Docker 部署实战

> 📅 **更新时间**：2025-06-12  
> 🎯 **难度等级**：Level 3-4 (中高级到高级)  
> ⏱️ **预计阅读**：60-90 分钟

---

## 1. 引言：为什么选择 FastAPI + Docker 部署 LLM 应用

在生产环境中部署 LLM 应用时，我们需要兼顾**性能、可维护性和可扩展性**。FastAPI + Docker 的组合成为当前最流行的部署方案之一，原因如下：

### FastAPI 的优势

- **高性能**：基于 Starlette 和 Pydantic，性能与 NodeJS 和 Go 相当，是 Python 最快的框架之一
- **异步原生**：原生支持 `async/await`，完美适配 LLM 推理的 I/O 密集型场景
- **自动文档**：自动生成 OpenAPI 和 Swagger 文档，降低 API 维护成本
- **类型安全**：基于 Python 类型提示，减少运行时错误
- **生态丰富**：易于集成各种 LLM 库（Transformers、vLLM、LangChain 等）

### Docker 的优势

- **环境一致性**：开发、测试、生产环境完全一致，消除"在我机器上能运行"的问题
- **快速部署**：一键启动完整服务栈（API + 数据库 + 缓存）
- **资源隔离**：容器级别隔离，便于资源管理和限流
- **水平扩展**：配合 Kubernetes 或 Docker Swarm 实现弹性扩缩容

### 典型架构

```mermaid
graph TB
    Client[客户端/前端] --> Gateway[API 网关/负载均衡]
    Gateway --> API1[FastAPI 实例 1]
    Gateway --> API2[FastAPI 实例 2]
    Gateway --> API3[FastAPI 实例 N]
    
    API1 --> Model[LLM 推理引擎]
    API2 --> Model
    API3 --> Model
    
    API1 --> Redis[(Redis 缓存)]
    API2 --> Redis
    API3 --> Redis
    
    API1 --> DB[(PostgreSQL)]
    API2 --> DB
    API3 --> DB
    
    API1 --> Monitor[Prometheus 监控]
    API2 --> Monitor
    API3 --> Monitor
    
    Monitor --> Grafana[Grafana 可视化]
```

---

## 2. FastAPI 基础

### 2.1 异步编程模型

FastAPI 的核心优势之一是原生支持异步编程。对于 LLM 应用来说，模型推理通常是 CPU/GPU 密集型，但网络请求、数据库查询、缓存操作等是 I/O 密集型，异步编程可以显著提升并发性能。

#### 2.1.1 同步 vs 异步

```python
# 传统同步方式（阻塞）
from fastapi import FastAPI
import time

app = FastAPI()

@app.get("/slow-sync")
def slow_endpoint():
    time.sleep(3)  # 阻塞整个线程
    return {"message": "完成"}

# 异步方式（非阻塞）
from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.get("/fast-async")
async def fast_endpoint():
    await asyncio.sleep(3)  # 释放控制权，处理其他请求
    return {"message": "完成"}
```

#### 2.1.2 异步最佳实践

```python
from fastapi import FastAPI, HTTPException
import asyncio
import httpx

app = FastAPI()

# ✅ 正确：使用异步 HTTP 客户端
@app.get("/fetch-data")
async def fetch_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()

# ✅ 正确：并发多个 I/O 操作
@app.get("/fetch-multiple")
async def fetch_multiple():
    async with httpx.AsyncClient() as client:
        # 并发请求，而非顺序请求
        responses = await asyncio.gather(
            client.get("https://api1.example.com/data"),
            client.get("https://api2.example.com/data"),
            client.get("https://api3.example.com/data"),
        )
        return [r.json() for r in responses]

# ⚠️ 注意：CPU 密集型任务应该在线程池中执行
from concurrent.futures import ThreadPoolExecutor
import time

executor = ThreadPoolExecutor(max_workers=4)

@app.get("/cpu-heavy")
async def cpu_heavy():
    # 使用 run_in_executor 避免阻塞事件循环
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, heavy_computation)
    return {"result": result}

def heavy_computation():
    time.sleep(2)  # 模拟 CPU 密集计算
    return 42
```

### 2.2 路由和请求处理

FastAPI 的路由系统简洁而强大，支持路径参数、查询参数、请求体等多种方式。

#### 2.2.1 路由基础

```python
from fastapi import FastAPI, Path, Query, Body
from typing import Optional

app = FastAPI()

# 路径参数
@app.get("/models/{model_id}")
async def get_model(
    model_id: str = Path(..., description="模型 ID", min_length=1),
    version: Optional[str] = Query(None, description="模型版本")
):
    return {"model_id": model_id, "version": version or "latest"}

# 请求体
@app.post("/models/{model_id}/generate")
async def generate_text(
    model_id: str,
    prompt: str = Body(..., embed=True, description="输入提示词"),
    max_tokens: int = Body(100, ge=1, le=4096, description="最大生成 token 数"),
    temperature: float = Body(0.7, ge=0.0, le=2.0, description="温度参数")
):
    return {
        "model_id": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

# HTTP 方法
@app.get("/models")          # 获取模型列表
@app.post("/models")         # 创建新模型
@app.put("/models/{id}")     # 更新模型
@app.delete("/models/{id}")  # 删除模型
```

#### 2.2.2 路由分组与前缀

```python
from fastapi import APIRouter

# 创建路由分组
chat_router = APIRouter(prefix="/chat", tags=["聊天"])
completion_router = APIRouter(prefix="/completions", tags=["补全"])

@chat_router.post("/messages")
async def send_message(message: str):
    return {"response": f"Echo: {message}"}

@completion_router.post("/generate")
async def generate(prompt: str):
    return {"text": f"Completed: {prompt}"}

# 注册路由
app.include_router(chat_router, prefix="/v1")
app.include_router(completion_router, prefix="/v1")
```

### 2.3 Pydantic 数据验证

Pydantic 是 FastAPI 的数据验证核心，通过类型提示自动验证请求数据。

#### 2.3.1 基础模型定义

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Message(BaseModel):
    role: Role = Field(..., description="消息角色")
    content: str = Field(..., min_length=1, max_length=10000, description="消息内容")
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError('消息内容不能为空')
        return v

class ChatRequest(BaseModel):
    model: str = Field(..., description="模型名称", example="gpt-5.2")
    messages: List[Message] = Field(..., min_length=1, description="对话历史")
    temperature: float = Field(
        0.7, 
        ge=0.0, 
        le=2.0, 
        description="温度参数，控制随机性"
    )
    max_tokens: int = Field(
        1024, 
        ge=1, 
        le=8192, 
        description="最大生成 token 数"
    )
    stream: bool = Field(False, description="是否启用流式输出")
    stop: Optional[List[str]] = Field(None, description="停止词序列")

class ChatResponse(BaseModel):
    id: str
    model: str
    choices: List[dict]
    usage: dict
```

#### 2.3.2 模型配置与示例

```python
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    name: str = Field(..., example="llama-3-70b")
    max_batch_size: int = Field(32, ge=1, le=128)
    timeout_seconds: float = Field(30.0, gt=0)
    enable_cache: bool = Field(True)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "llama-3-70b",
                "max_batch_size": 32,
                "timeout_seconds": 30.0,
                "enable_cache": True
            }
        }

@app.post("/models/configure")
async def configure_model(config: ModelConfig):
    return {"status": "configured", "config": config}
```

### 2.4 依赖注入

依赖注入是 FastAPI 的强大特性，用于处理认证、数据库连接、日志记录等跨切面关注点。

#### 2.4.1 基础依赖

```python
from fastapi import Depends, HTTPException, Header
from typing import Optional

# 依赖函数
async def verify_token(x_token: str = Header(...)):
    """验证 API Token"""
    if x_token != "secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return x_token

async def get_db():
    """获取数据库连接"""
    db = {"connected": True}
    try:
        yield db
    finally:
        db["connected"] = False

# 使用依赖
@app.get("/protected", dependencies=[Depends(verify_token)])
async def protected_route():
    return {"message": "访问成功"}

@app.get("/db-query")
async def db_query(db = Depends(get_db)):
    return {"db_status": db}
```

#### 2.4.2 依赖类

```python
from fastapi import Request

class ModelLoader:
    """模型加载器依赖类"""
    
    def __init__(self):
        self.model = None
    
    async def load_model(self, model_name: str):
        """加载模型（伪代码）"""
        print(f"Loading model: {model_name}")
        self.model = {"name": model_name, "loaded": True}
        return self.model
    
    def predict(self, input_text: str):
        """模型推理"""
        if not self.model:
            raise HTTPException(status_code=500, detail="Model not loaded")
        return f"Prediction for: {input_text}"

# 创建全局依赖实例
model_loader = ModelLoader()

@app.on_event("startup")
async def startup_event():
    await model_loader.load_model("llama-3-70b")

@app.post("/predict")
async def predict(text: str, loader: ModelLoader = Depends(lambda: model_loader)):
    return {"prediction": loader.predict(text)}
```

---

## 3. LLM 应用架构设计

### 3.1 模型加载方案

部署 LLM 应用时，模型加载是最关键的环节。我们有两种主流方案：

#### 3.1.1 方案一：HuggingFace Transformers

适合中小规模模型，灵活性高，支持各种模型架构。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class HFModelService:
    """HuggingFace 模型服务"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3-8b"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
    
    def load_model(self, device: str = "auto"):
        """加载模型和分词器"""
        print(f"Loading model: {self.model_name}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 加载模型（自动选择设备）
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        # 创建推理管道
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=device
        )
        
        print(f"Model loaded successfully on {device}")
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """生成文本"""
        if not self.pipeline:
            raise RuntimeError("Model not loaded")
        
        result = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return result[0]['generated_text']

# 全局服务实例
model_service = HFModelService()
```

#### 3.1.2 方案二：vLLM 集成（推荐生产环境）

vLLM 是专为生产环境设计的高性能推理引擎，支持 PagedAttention 和连续批处理。

```python
from vllm import LLM, SamplingParams
from typing import List, AsyncGenerator
import asyncio

class VLLMService:
    """vLLM 模型服务"""
    
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-3-8b",
        tensor_parallel_size: int = 1
    ):
        self.model_name = model_name
        self.llm = None
        self.tensor_parallel_size = tensor_parallel_size
    
    def load_model(self):
        """加载 vLLM 模型"""
        print(f"Loading vLLM model: {self.model_name}")
        
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
        )
        
        print("vLLM model loaded successfully")
    
    def generate(
        self, 
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[str]:
        """批量生成文本"""
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    async def generate_stream(
        self, 
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        
        # 注意：实际使用时需要单独初始化 AsyncLLMEngine
        # 这里仅展示接口设计
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # 模拟流式输出
        result = "这是生成的文本..."
        for i in range(0, len(result), 5):
            yield result[i:i+5]
            await asyncio.sleep(0.1)

# 全局服务实例
vllm_service = VLLMService()
```

#### 3.1.3 模型服务依赖注入

```python
from fastapi import Depends
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    print("Loading model...")
    vllm_service.load_model()
    yield
    # 关闭时清理资源
    print("Shutting down...")
    del vllm_service

app = FastAPI(lifespan=lifespan)
```

### 3.2 推理 API 设计

#### 3.2.1 基础补全 API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import uuid

app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="输入提示词")
    max_tokens: int = Field(100, ge=1, le=4096, description="最大生成 token 数")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="核采样参数")
    stop: Optional[List[str]] = Field(None, description="停止词")

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    text: str
    finish_reason: str

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """创建文本补全"""
    try:
        start_time = time.time()
        
        # 调用模型
        text = vllm_service.generate(
            prompts=[request.prompt],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )[0]
        
        latency = time.time() - start_time
        
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=vllm_service.model_name,
            text=text,
            finish_reason="stop"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 3.2.2 聊天 API（支持对话历史）

```python
class Message(BaseModel):
    role: str = Field(..., description="消息角色", enum=["system", "user", "assistant"])
    content: str = Field(..., min_length=1, description="消息内容")

class ChatCompletionRequest(BaseModel):
    model: str = Field("llama-3-8b", description="模型名称")
    messages: List[Message] = Field(..., min_length=1, description="对话历史")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=4096)
    stream: bool = Field(False, description="是否启用流式输出")

class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: dict

def format_messages_to_prompt(messages: List[Message]) -> str:
    """将消息列表转换为提示词格式"""
    prompt = ""
    for msg in messages:
        prompt += f"<|{msg.role}|>\n{msg.content}\n"
    prompt += "<|assistant|>\n"
    return prompt

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """创建聊天补全"""
    prompt = format_messages_to_prompt(request.messages)
    
    text = vllm_service.generate(
        prompts=[prompt],
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )[0]
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content=text),
                finish_reason="stop"
            )
        ],
        usage={
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(text.split()),
            "total_tokens": len(prompt.split()) + len(text.split())
        }
    )
```

#### 3.2.3 流式输出 API

```python
from fastapi.responses import StreamingResponse
import json

async def generate_stream_response(
    prompt: str, 
    model: str,
    max_tokens: int,
    temperature: float
):
    """生成流式响应"""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    
    async for token in vllm_service.generate_stream(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    ):
        data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(data)}\n\n"
    
    # 发送结束标记
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions/stream")
async def create_chat_completion_stream(request: ChatCompletionRequest):
    """创建流式聊天补全"""
    if not request.stream:
        # 非流式调用普通接口
        return await create_chat_completion(request)
    
    prompt = format_messages_to_prompt(request.messages)
    
    return StreamingResponse(
        generate_stream_response(
            prompt=prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        ),
        media_type="text/event-stream"
    )
```

### 3.3 请求队列和并发控制

生产环境需要处理大量并发请求，必须实现请求队列和限流机制。

#### 3.3.1 异步请求队列

```python
import asyncio
from collections import deque
from typing import Dict, Any

class RequestQueue:
    """异步请求队列"""
    
    def __init__(self, max_size: int = 100):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.processing = False
    
    async def add_request(self, request_data: Dict[str, Any]) -> bool:
        """添加请求到队列"""
        try:
            await asyncio.wait_for(self.queue.put(request_data), timeout=5.0)
            return True
        except asyncio.TimeoutError:
            return False
    
    async def get_next_request(self) -> Dict[str, Any]:
        """获取下一个请求"""
        return await self.queue.get()
    
    @property
    def size(self) -> int:
        return self.queue.qsize()

# 全局请求队列
request_queue = RequestQueue(max_size=100)
```

#### 3.3.2 信号量控制并发

```python
from asyncio import Semaphore

class ConcurrencyController:
    """并发控制器"""
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = Semaphore(max_concurrent)
        self.active_requests = 0
    
    async def acquire(self):
        """获取执行权限"""
        await self.semaphore.acquire()
        self.active_requests += 1
    
    def release(self):
        """释放执行权限"""
        self.semaphore.release()
        self.active_requests -= 1
    
    @property
    def status(self) -> Dict[str, int]:
        return {
            "active_requests": self.active_requests,
            "max_concurrent": self.semaphore._value
        }

# 全局并发控制器
concurrency_ctrl = ConcurrencyController(max_concurrent=5)

@app.post("/v1/completions/queued")
async def create_completion_queued(request: CompletionRequest):
    """带队列和并发控制的补全 API"""
    
    # 检查队列是否已满
    if request_queue.size >= 100:
        raise HTTPException(status_code=503, detail="Service busy, try later")
    
    # 添加到队列
    await request_queue.add_request({
        "request": request,
        "timestamp": time.time()
    })
    
    # 等待处理
    concurrency_ctrl.acquire()
    try:
        text = vllm_service.generate(
            prompts=[request.prompt],
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )[0]
        
        return {"text": text}
    finally:
        concurrency_ctrl.release()
```

#### 3.3.3 速率限制（Token Bucket 算法）

```python
import time
from fastapi import Request

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, rate: int = 10, capacity: int = 20):
        self.rate = rate  # 每秒令牌数
        self.capacity = capacity  # 最大令牌数
        self.tokens = capacity
        self.last_refill = time.time()
        self.clients: Dict[str, dict] = {}
    
    def _refill(self, client_id: str):
        """补充令牌"""
        now = time.time()
        client = self.clients[client_id]
        elapsed = now - client["last_refill"]
        
        client["tokens"] = min(
            self.capacity,
            client["tokens"] + elapsed * self.rate
        )
        client["last_refill"] = now
    
    def is_allowed(self, client_id: str) -> bool:
        """检查是否允许请求"""
        if client_id not in self.clients:
            self.clients[client_id] = {
                "tokens": self.capacity,
                "last_refill": time.time()
            }
        
        self._refill(client_id)
        
        if self.clients[client_id]["tokens"] >= 1:
            self.clients[client_id]["tokens"] -= 1
            return True
        return False

# 全局速率限制器
rate_limiter = RateLimiter(rate=10, capacity=20)

async def check_rate_limit(request: Request):
    """速率限制依赖"""
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

@app.get("/v1/status", dependencies=[Depends(check_rate_limit)])
async def get_status():
    return {"status": "ok"}
```

---

## 4. Docker 容器化

### 4.1 Dockerfile 编写

#### 4.1.1 基础 Dockerfile

```dockerfile
# 使用 Python 3.11 slim 镜像（体积小，安全性高）
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 4.1.2 多阶段构建（优化镜像大小）

```dockerfile
# ===== 第一阶段：构建依赖 =====
FROM python:3.11-slim AS builder

WORKDIR /app

# 安装构建工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 安装依赖到指定目录
RUN pip install --prefix=/install -r requirements.txt

# ===== 第二阶段：运行环境 =====
FROM python:3.11-slim AS runner

WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:$PATH" \
    PYTHONPATH="/install/lib/python3.11/site-packages"

# 从构建阶段复制依赖
COPY --from=builder /install /install

# 复制应用代码
COPY . .

# 创建非 root 用户
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 4.1.3 .dockerignore 文件

```dockerignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Git
.git/
.gitignore

# Docker
Dockerfile
docker-compose.yml
.dockerignore

# 文档
*.md
docs/

# 测试
tests/
.pytest_cache/
.coverage

# 其他
.env
*.log
.DS_Store
```

### 4.2 docker-compose 编排

#### 4.2.1 完整服务栈

```yaml
version: '3.8'

services:
  # FastAPI 应用
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: llm-api
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=${MODEL_NAME:-meta-llama/Llama-3-8b}
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/llm_db
      - LOG_LEVEL=${LOG_LEVEL:-info}
      - MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-10}
    volumes:
      - ./logs:/app/logs
      - model-cache:/root/.cache/huggingface
    depends_on:
      redis:
        condition: service_healthy
      db:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
    networks:
      - llm-network

  # Redis 缓存
  redis:
    image: redis:7-alpine
    container_name: llm-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: >
      redis-server 
      --maxmemory 2gb 
      --maxmemory-policy allkeys-lru 
      --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - llm-network

  # PostgreSQL 数据库
  db:
    image: postgres:15-alpine
    container_name: llm-db
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=llm_db
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - llm-network

  # Prometheus 监控
  prometheus:
    image: prom/prometheus:latest
    container_name: llm-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped
    networks:
      - llm-network

  # Grafana 可视化
  grafana:
    image: grafana/grafana:latest
    container_name: llm-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - llm-network

volumes:
  redis-data:
  postgres-data:
  prometheus-data:
  grafana-data:
  model-cache:

networks:
  llm-network:
    driver: bridge
```

#### 4.2.2 环境变量配置

创建 `.env` 文件：

```bash
# 模型配置
MODEL_NAME=meta-llama/Llama-3-8b
MAX_CONCURRENT_REQUESTS=10

# 日志配置
LOG_LEVEL=info

# Grafana 密码
GRAFANA_PASSWORD=admin123

# 数据库配置（生产环境请使用强密码）
POSTGRES_PASSWORD=postgres
```

### 4.3 环境变量和配置管理

#### 4.3.1 Pydantic Settings

```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """应用配置"""
    
    # 应用配置
    APP_NAME: str = "LLM API Service"
    DEBUG: bool = False
    LOG_LEVEL: str = "info"
    
    # 模型配置
    MODEL_NAME: str = "meta-llama/Llama-3-8b"
    MODEL_CACHE_DIR: str = "/tmp/models"
    MAX_MODEL_LENGTH: int = 4096
    GPU_MEMORY_UTILIZATION: float = 0.9
    
    # 并发控制
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_QUEUE_SIZE: int = 100
    REQUEST_TIMEOUT: int = 60
    
    # Redis 配置
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 3600  # 1小时
    
    # 数据库配置
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/llm_db"
    
    # 安全配置
    API_KEYS: list = ["secret-key-1", "secret-key-2"]
    CORS_ORIGINS: list = ["*"]
    RATE_LIMIT_PER_SECOND: int = 10
    
    # 监控配置
    ENABLE_METRICS: bool = True
    PROMETHEUS_PORT: int = 9090
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# 全局配置实例
settings = Settings()
```

---

## 5. 生产级最佳实践

### 5.1 健康检查和自动重启

#### 5.1.1 健康检查端点

```python
from fastapi import FastAPI
from pydantic import BaseModel
import psutil
import time

app = FastAPI()

class HealthStatus(BaseModel):
    status: str
    timestamp: float
    uptime: float
    model_loaded: bool
    memory_usage: dict
    active_requests: int

start_time = time.time()

@app.get("/health")
async def health_check():
    """基础健康检查"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - start_time
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """详细健康检查"""
    process = psutil.Process()
    
    return HealthStatus(
        status="healthy" if model_service.model else "degraded",
        timestamp=time.time(),
        uptime=time.time() - start_time,
        model_loaded=model_service.model is not None,
        memory_usage={
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
            "percent": process.memory_percent()
        },
        active_requests=concurrency_ctrl.active_requests
    )

@app.get("/health/ready")
async def readiness_check():
    """就绪检查（Kubernetes）"""
    if not model_service.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}
```

#### 5.1.2 Docker 健康检查

在 Dockerfile 中添加：

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; r = requests.get('http://localhost:8000/health'); exit(0 if r.status_code == 200 else 1)" || exit 1
```

在 docker-compose.yml 中添加：

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

### 5.2 日志和监控

#### 5.2.1 结构化日志

```python
import logging
import json
from pythonjsonlogger import jsonlogger

# 配置 JSON 日志
logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    '%(asctime)s %(levelname)s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# 使用日志
@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    logger.info("Processing completion request", extra={
        "prompt_length": len(request.prompt),
        "max_tokens": request.max_tokens,
        "temperature": request.temperature
    })
    
    # ... 处理逻辑
    
    logger.info("Completion request processed", extra={
        "response_length": len(text),
        "latency_ms": latency * 1000
    })
```

#### 5.2.2 Prometheus 指标

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response

# 定义指标
REQUEST_COUNT = Counter(
    'llm_requests_total',
    'Total number of LLM requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'llm_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'llm_active_requests',
    'Number of active requests'
)

MODEL_CACHE_HITS = Counter(
    'llm_cache_hits_total',
    'Total number of cache hits',
    ['type']
)

# 中间件记录指标
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    try:
        response = await call_next(request)
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(time.time() - start_time)
        
        return response
    finally:
        ACTIVE_REQUESTS.dec()

# Prometheus 指标端点
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

#### 5.2.3 Prometheus 配置文件

创建 `prometheus.yml`：

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'llm-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
```

### 5.3 性能优化

#### 5.3.1 Gunicorn + Uvicorn Workers

```dockerfile
# Dockerfile 中使用 gunicorn 启动
CMD ["gunicorn", "main:app", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
```

或者在 docker-compose.yml 中配置：

```yaml
command: >
  gunicorn main:app
  -w 4
  -k uvicorn.workers.UvicornWorker
  --bind 0.0.0.0:8000
  --timeout 120
  --keep-alive 5
  --access-logfile -
  --error-logfile -
```

#### 5.3.2 Worker 数量计算公式

```
workers = (2 × CPU_cores) + 1
```

例如：
- 4 核 CPU：9 个 workers
- 8 核 CPU：17 个 workers

#### 5.3.3 响应压缩

```python
from fastapi.middleware.gzip import GZipMiddleware

# 添加 Gzip 压缩中间件
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

#### 5.3.4 Redis 缓存

```python
import redis.asyncio as redis
import json

redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

async def get_cached_response(cache_key: str) -> Optional[str]:
    """从 Redis 获取缓存"""
    try:
        cached = await redis_client.get(cache_key)
        if cached:
            MODEL_CACHE_HITS.labels(type="redis").inc()
            return cached
    except Exception as e:
        logger.error(f"Redis get error: {e}")
    return None

async def cache_response(cache_key: str, response: str, ttl: int = 3600):
    """缓存响应到 Redis"""
    try:
        await redis_client.setex(cache_key, ttl, response)
    except Exception as e:
        logger.error(f"Redis set error: {e}")

@app.post("/v1/completions/cached")
async def create_completion_cached(request: CompletionRequest):
    """带缓存的补全 API"""
    # 生成缓存键
    cache_key = f"completion:{hash(request.prompt)}:{request.max_tokens}:{request.temperature}"
    
    # 检查缓存
    cached = await get_cached_response(cache_key)
    if cached:
        return json.loads(cached)
    
    # 生成响应
    text = vllm_service.generate(
        prompts=[request.prompt],
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )[0]
    
    response = {"text": text}
    
    # 缓存响应
    await cache_response(cache_key, json.dumps(response))
    
    return response
```

### 5.4 安全配置

#### 5.4.1 CORS 配置

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### 5.4.2 API 密钥认证

```python
from fastapi import Security
from fastapi.security.api_key import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """验证 API 密钥"""
    if api_key not in settings.API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

@app.get("/v1/protected", dependencies=[Depends(verify_api_key)])
async def protected_endpoint():
    return {"message": "Access granted"}
```

#### 5.4.3 请求验证和清理

```python
import re
from fastapi import Request

# 恶意内容检测
MALICIOUS_PATTERNS = [
    r"<script>.*?</script>",  # XSS
    r"UNION\s+SELECT",        # SQL 注入
    r"DROP\s+TABLE",          # SQL 注入
]

def sanitize_input(text: str) -> str:
    """清理输入文本"""
    for pattern in MALICIOUS_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """安全中间件"""
    # 检查请求大小
    if request.headers.get("content-length"):
        content_length = int(request.headers["content-length"])
        if content_length > 10 * 1024 * 1024:  # 10MB
            return JSONResponse(
                status_code=413,
                content={"detail": "Request too large"}
            )
    
    response = await call_next(request)
    return response
```

---

## 6. 完整实战案例

### 6.1 项目结构

```
llm-api-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用入口
│   ├── config.py            # 配置管理
│   ├── models.py            # Pydantic 模型
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py   # LLM 推理服务
│   │   └── cache_service.py # 缓存服务
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── chat.py          # 聊天路由
│   │   └── completions.py   # 补全路由
│   └── middleware/
│       ├── __init__.py
│       ├── auth.py          # 认证中间件
│       └── metrics.py       # 监控中间件
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── prometheus.yml
├── .env
├── .env.example
├── .dockerignore
├── requirements.txt
└── README.md
```

### 6.2 核心代码实现

#### 6.2.1 requirements.txt

```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
gunicorn==21.2.0
pydantic==2.5.3
pydantic-settings==2.1.0
vllm==0.3.3
redis==5.0.1
psycopg2-binary==2.9.9
prometheus-client==0.19.0
python-json-logger==2.0.7
psutil==5.9.7
httpx==0.26.0
python-multipart==0.0.6
```

#### 6.2.2 app/config.py

```python
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    APP_NAME: str = "LLM API Service"
    DEBUG: bool = False
    LOG_LEVEL: str = "info"
    
    MODEL_NAME: str = "meta-llama/Llama-3-8b"
    MAX_CONCURRENT_REQUESTS: int = 10
    REDIS_URL: str = "redis://localhost:6379/0"
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/llm_db"
    
    API_KEYS: List[str] = ["test-key-123"]
    CORS_ORIGINS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

#### 6.2.3 app/models.py

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Message(BaseModel):
    role: Role
    content: str = Field(..., min_length=1, max_length=10000)

class ChatRequest(BaseModel):
    model: str = Field("llama-3-8b", description="模型名称")
    messages: List[Message] = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=4096)
    stream: bool = False
    stop: Optional[List[str]] = None

class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: dict
```

#### 6.2.4 app/services/llm_service.py

```python
from typing import List, Optional
import time

class MockLLMService:
    """模拟 LLM 服务（用于演示，实际使用 vLLM 或 Transformers）"""
    
    def __init__(self):
        self.model_name = "mock-llm-3b"
        self.loaded = False
    
    def load_model(self):
        print(f"Loading model: {self.model_name}")
        self.loaded = True
        print("Model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> str:
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        # 模拟推理延迟
        time.sleep(0.5)
        
        # 模拟生成结果
        return f"这是基于提示词生成的回复：{prompt[:50]}..."

llm_service = MockLLMService()
```

#### 6.2.5 app/routers/chat.py

```python
from fastapi import APIRouter, HTTPException, Depends
from app.models import ChatRequest, ChatResponse, Message
from app.services.llm_service import llm_service
import time
import uuid

router = APIRouter(prefix="/v1/chat", tags=["chat"])

def format_messages(messages: List[Message]) -> str:
    prompt = ""
    for msg in messages:
        prompt += f"{msg.role}: {msg.content}\n"
    prompt += "assistant: "
    return prompt

@router.post("/completions", response_model=ChatResponse)
async def create_chat_completion(request: ChatRequest):
    """创建聊天补全"""
    try:
        start_time = time.time()
        
        prompt = format_messages(request.messages)
        
        # 调用 LLM 服务
        generated_text = llm_service.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        latency = time.time() - start_time
        
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=Message(role="assistant", content=generated_text),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(prompt.split()) + len(generated_text.split()),
                "latency_ms": round(latency * 1000, 2)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 6.2.6 app/main.py

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.routers import chat
from app.services.llm_service import llm_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    print("Starting LLM API Service...")
    llm_service.load_model()
    yield
    # 关闭时清理
    print("Shutting down LLM API Service...")

app = FastAPI(
    title=settings.APP_NAME,
    description="LLM API Service with FastAPI and Docker",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": llm_service.loaded
    }

@app.get("/")
async def root():
    return {
        "message": "Welcome to LLM API Service",
        "docs": "/docs",
        "health": "/health"
    }
```

### 6.3 本地测试

#### 6.3.1 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 6.3.2 运行服务

```bash
# 开发模式
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
gunicorn app.main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

#### 6.3.3 测试 API

使用 curl 测试：

```bash
# 健康检查
curl http://localhost:8000/health

# 聊天补全
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is FastAPI?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

使用 Python 测试：

```python
import httpx
import asyncio

async def test_api():
    async with httpx.AsyncClient() as client:
        # 健康检查
        response = await client.get("http://localhost:8000/health")
        print("Health:", response.json())
        
        # 聊天补全
        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "llama-3-8b",
                "messages": [
                    {"role": "user", "content": "Explain Docker in one sentence."}
                ],
                "max_tokens": 50
            }
        )
        print("Chat:", response.json())

asyncio.run(test_api())
```

### 6.4 Docker 打包和部署

#### 6.4.1 创建 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "app.main:app", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
```

#### 6.4.2 创建 docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    container_name: llm-api
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=info
    depends_on:
      - redis
    restart: unless-stopped
    networks:
      - llm-network

  redis:
    image: redis:7-alpine
    container_name: llm-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped
    networks:
      - llm-network

volumes:
  redis-data:

networks:
  llm-network:
    driver: bridge
```

#### 6.4.3 构建和运行

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f api

# 停止服务
docker-compose down

# 完全清理（包括卷）
docker-compose down -v
```

#### 6.4.4 验证部署

```bash
# 检查容器状态
docker-compose ps

# 测试 API
curl http://localhost:8000/health

# 查看日志
docker logs llm-api
```

---

## 7. 部署到云服务器（可选）

### 7.1 AWS EC2 部署

#### 7.1.1 步骤 1：创建 EC2 实例

```bash
# 选择 Ubuntu 22.04 LTS
# 实例类型：g5.xlarge（GPU）或 t3.large（CPU）
# 安全组：开放端口 8000、22
```

#### 7.1.2 步骤 2：安装 Docker

```bash
# SSH 到服务器
ssh -i your-key.pem ubuntu@your-ec2-ip

# 安装 Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu

# 安装 Docker Compose
sudo apt-get install docker-compose-plugin
```

#### 7.1.3 步骤 3：部署应用

```bash
# 上传代码
scp -r llm-api-service/ ubuntu@your-ec2-ip:~/

# SSH 并部署
ssh ubuntu@your-ec2-ip
cd llm-api-service
docker-compose up -d
```

### 7.2 使用 NGINX 反向代理

创建 `nginx.conf`：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        proxy_pass http://localhost:8000/health;
    }
}
```

更新 docker-compose.yml：

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped
```

### 7.3 HTTPS 配置（Let's Encrypt）

```bash
# 安装 Certbot
sudo apt-get install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d your-domain.com

# 自动续期
sudo certbot renew --dry-run
```

---

## 8. 参考资料

### 官方文档

- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [Docker 官方文档](https://docs.docker.com/)
- [vLLM 文档](https://docs.vllm.ai/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [Prometheus 文档](https://prometheus.io/docs/)
- [Grafana 文档](https://grafana.com/docs/)

### 相关资源

- [OpenAI API 兼容规范](https://platform.openai.com/docs/api-reference)
- [Gunicorn 文档](https://docs.gunicorn.org/)
- [Uvicorn 文档](https://www.uvicorn.org/)
- [Redis 文档](https://redis.io/docs/)

### 示例项目

- [FastAPI Full Stack](https://github.com/tiangolo/full-stack-fastapi-template)
- [vLLM OpenAI Compatible Server](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai)
- [LangServe](https://github.com/langchain-ai/langserve)

---

## 9. 总结

本文档详细介绍了如何使用 FastAPI + Docker 部署 LLM 应用，涵盖了从基础到生产的完整流程：

1. **FastAPI 基础**：异步编程、路由、数据验证、依赖注入
2. **LLM 架构设计**：模型加载、API 设计、并发控制
3. **Docker 容器化**：Dockerfile、docker-compose、配置管理
4. **生产最佳实践**：健康检查、监控、性能优化、安全配置
5. **完整实战案例**：从零构建、测试、部署
6. **云服务器部署**：AWS EC2、NGINX、HTTPS

通过本指南，你应该能够：
- ✅ 构建高性能的 LLM API 服务
- ✅ 使用 Docker 实现环境一致性和快速部署
- ✅ 实施生产级监控和安全措施
- ✅ 将服务部署到云服务器

祝你部署顺利！🚀