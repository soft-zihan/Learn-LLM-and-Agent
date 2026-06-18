# 高级编排与 Agent 通信协议

> 📅 **更新时间**: 2026-06-17

---

## 目录

- [1. Agent 编排模式](#1-agent-编排模式)
- [2. Agent 通信协议](#2-agent-通信协议)
- [3. 高级通信模式](#3-高级通信模式)
- [4. 容错与恢复](#4-容错与恢复)
- [5. 监控与可观测性](#5-监控与可观测性)
- [6. 性能优化](#6-性能优化)
- [7. 参考资料](#7-参考资料)

---

## 1. Agent 编排模式

### 1.1 流水线编排

流水线编排是最简单的多 Agent 协作模式,每个 Agent 负责一个阶段。

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import time

@dataclass
class AgentMessage:
    """Agent 消息"""
    content: str
    sender: str
    recipient: str
    metadata: Dict[str, Any] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}

class AgentPipeline:
    """Agent 流水线"""
    
    def __init__(self):
        self.agents: List[Dict] = []
        self.message_history: List[AgentMessage] = []
    
    def add_agent(
        self,
        name: str,
        agent_func: callable,
        input_transform: callable = None,
        output_transform: callable = None
    ):
        """添加 Agent 到流水线"""
        self.agents.append({
            "name": name,
            "func": agent_func,
            "input_transform": input_transform,
            "output_transform": output_transform
        })
    
    async def execute(self, initial_input: Any) -> Any:
        """执行流水线"""
        current_output = initial_input
        
        for agent_config in self.agents:
            # 输入转换
            if agent_config["input_transform"]:
                input_data = agent_config["input_transform"](current_output)
            else:
                input_data = current_output
            
            # 记录消息
            message = AgentMessage(
                content=str(input_data),
                sender="pipeline",
                recipient=agent_config["name"]
            )
            self.message_history.append(message)
            
            # 执行 Agent
            output = await agent_config["func"](input_data)
            
            # 输出转换
            if agent_config["output_transform"]:
                current_output = agent_config["output_transform"](output)
            else:
                current_output = output
            
            # 记录结果
            result_message = AgentMessage(
                content=str(current_output),
                sender=agent_config["name"],
                recipient="pipeline"
            )
            self.message_history.append(result_message)
        
        return current_output
    
    def get_execution_trace(self) -> List[Dict]:
        """获取执行轨迹"""
        return [
            {
                "sender": msg.sender,
                "recipient": msg.recipient,
                "content_preview": msg.content[:100],
                "timestamp": msg.timestamp
            }
            for msg in self.message_history
        ]

# 示例：文档处理流水线
async def research_agent(input_data: Dict) -> Dict:
    """研究 Agent"""
    # 模拟研究工作
    return {
        "topic": input_data["topic"],
        "research_findings": ["Finding 1", "Finding 2"],
        "sources": ["Source 1", "Source 2"]
    }

async def writing_agent(input_data: Dict) -> Dict:
    """写作 Agent"""
    return {
        "topic": input_data["topic"],
        "draft": f"Article about {input_data['topic']}",
        "findings_used": input_data["research_findings"]
    }

async def editing_agent(input_data: Dict) -> Dict:
    """编辑 Agent"""
    return {
        "topic": input_data["topic"],
        "final_article": f"Polished: {input_data['draft']}",
        "quality_score": 0.95
    }

# 构建流水线
pipeline = AgentPipeline()

pipeline.add_agent("researcher", research_agent)
pipeline.add_agent("writer", writing_agent)
pipeline.add_agent("editor", editing_agent)

# 执行
async def run_pipeline_example():
    """流水线示例"""
    result = await pipeline.execute({
        "topic": "AI in Healthcare"
    })
    print(f"Final result: {result}")
    
    # 查看执行轨迹
    trace = pipeline.get_execution_trace()
    for step in trace:
        print(f"{step['sender']} -> {step['recipient']}: {step['content_preview']}")
```

### 1.2 层次化编排

```python
class HierarchicalAgent:
    """层次化 Agent"""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.subordinates: List['HierarchicalAgent'] = []
        self.supervisor: Optional['HierarchicalAgent'] = None
    
    def add_subordinate(self, agent: 'HierarchicalAgent'):
        """添加下属"""
        agent.supervisor = self
        self.subordinates.append(agent)
    
    async def execute_task(self, task: Dict) -> Dict:
        """执行任务"""
        if self.subordinates:
            # 分解任务并分配给下属
            subtasks = self.decompose_task(task)
            results = await self.delegate_tasks(subtasks)
            
            # 整合结果
            return self.integrate_results(results)
        else:
            # 执行者直接处理
            return await self.process_task(task)
    
    def decompose_task(self, task: Dict) -> List[Dict]:
        """分解任务"""
        # 简化实现
        return [
            {**task, "subtask_id": i, "assignee": sub.name}
            for i, sub in enumerate(self.subordinates)
        ]
    
    async def delegate_tasks(self, subtasks: List[Dict]) -> List[Dict]:
        """分配任务"""
        tasks = []
        for subtask in subtasks:
            assignee = next(
                sub for sub in self.subordinates
                if sub.name == subtask["assignee"]
            )
            task = asyncio.create_task(assignee.execute_task(subtask))
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def integrate_results(self, results: List[Dict]) -> Dict:
        """整合结果"""
        return {
            "status": "completed",
            "sub_results": results,
            "summary": f"Integrated {len(results)} results"
        }
    
    async def process_task(self, task: Dict) -> Dict:
        """处理任务（由子类实现）"""
        raise NotImplementedError

# 示例：项目管理层次结构
class ProjectManager(HierarchicalAgent):
    def __init__(self):
        super().__init__("project_manager", "manager")

class Developer(HierarchicalAgent):
    async def process_task(self, task: Dict) -> Dict:
        return {
            "task": task["description"],
            "status": "completed",
            "code": "function implementation"
        }

class Tester(HierarchicalAgent):
    async def process_task(self, task: Dict) -> Dict:
        return {
            "task": task["description"],
            "status": "completed",
            "test_results": "all tests passed"
        }

# 构建组织结构
async def hierarchical_example():
    """层次化示例"""
    pm = ProjectManager()
    
    dev1 = Developer()
    dev1.name = "developer_1"
    dev2 = Developer()
    dev2.name = "developer_2"
    tester = Tester()
    tester.name = "tester"
    
    pm.add_subordinate(dev1)
    pm.add_subordinate(dev2)
    pm.add_subordinate(tester)
    
    # 执行任务
    result = await pm.execute_task({
        "description": "Build a web application",
        "deadline": "2026-12-31"
    })
    print(f"Project result: {result}")
```

### 1.3 图编排

```python
class AgentGraph:
    """Agent 图"""
    
    def __init__(self):
        self.nodes: Dict[str, callable] = {}
        self.edges: List[tuple] = []
        self.state: Dict[str, Any] = {}
    
    def add_node(self, name: str, agent_func: callable):
        """添加节点"""
        self.nodes[name] = agent_func
    
    def add_edge(self, from_node: str, to_node: str, condition: callable = None):
        """添加边"""
        self.edges.append((from_node, to_node, condition))
    
    async def execute(self, start_node: str, initial_state: Dict) -> Dict:
        """执行图"""
        self.state = initial_state
        visited = set()
        queue = [start_node]
        
        while queue:
            current_node = queue.pop(0)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # 执行当前节点
            if current_node in self.nodes:
                result = await self.nodes[current_node](self.state)
                self.state[f"{current_node}_output"] = result
            
            # 获取下一个节点
            next_nodes = self.get_next_nodes(current_node)
            queue.extend(next_nodes)
        
        return self.state
    
    def get_next_nodes(self, current_node: str) -> List[str]:
        """获取下一个节点"""
        next_nodes = []
        
        for from_node, to_node, condition in self.edges:
            if from_node == current_node:
                if condition is None or condition(self.state):
                    next_nodes.append(to_node)
        
        return next_nodes

# 示例：决策图
async def decision_graph_example():
    """决策图示例"""
    graph = AgentGraph()
    
    # 添加节点
    async def analyze_data(state):
        return {"analysis": "data analyzed"}
    
    async def simple_solution(state):
        return {"solution": "simple"}
    
    async def complex_solution(state):
        return {"solution": "complex"}
    
    async def implement(state):
        return {"implementation": "done"}
    
    graph.add_node("analyze", analyze_data)
    graph.add_node("simple", simple_solution)
    graph.add_node("complex", complex_solution)
    graph.add_node("implement", implement)
    
    # 添加边
    graph.add_edge("analyze", "simple", 
                   lambda s: s.get("complexity", "low") == "low")
    graph.add_edge("analyze", "complex",
                   lambda s: s.get("complexity", "low") == "high")
    graph.add_edge("simple", "implement")
    graph.add_edge("complex", "implement")
    
    # 执行
    result = await graph.execute("analyze", {"complexity": "high"})
    print(f"Graph result: {result}")
```

## 2. Agent 通信协议

### 2.1 消息格式标准

```python
from enum import Enum
from pydantic import BaseModel, Field
from typing import Union

class MessageType(str, Enum):
    """消息类型"""
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    QUERY = "query"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"

class Priority(str, Enum):
    """优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class AgentMessageV2(BaseModel):
    """Agent 消息 V2"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    sender: str
    recipient: str
    message_type: MessageType
    priority: Priority = Priority.NORMAL
    
    # 内容
    content: Union[str, Dict, List]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # 路由
    correlation_id: Optional[str] = None  # 关联 ID
    reply_to: Optional[str] = None  # 回复地址
    ttl: Optional[int] = 3600  # 生存时间（秒）
    
    # 重试
    retry_count: int = 0
    max_retries: int = 3
    
    class Config:
        json_schema_extra = {
            "example": {
                "sender": "agent_a",
                "recipient": "agent_b",
                "message_type": "task",
                "content": {"task": "analyze data"},
                "priority": "high"
            }
        }

class TaskMessage(AgentMessageV2):
    """任务消息"""
    message_type: MessageType = MessageType.TASK
    task_id: str
    task_description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    deadline: Optional[float] = None

class ResultMessage(AgentMessageV2):
    """结果消息"""
    message_type: MessageType = MessageType.RESULT
    task_id: str
    result: Any
    success: bool
    error_message: Optional[str] = None

# 消息序列化
class MessageSerializer:
    """消息序列化器"""
    
    @staticmethod
    def serialize(message: AgentMessageV2) -> str:
        """序列化为 JSON"""
        return message.json()
    
    @staticmethod
    def deserialize(data: str) -> AgentMessageV2:
        """从 JSON 反序列化"""
        return AgentMessageV2.parse_raw(data)
    
    @staticmethod
    def to_bytes(message: AgentMessageV2) -> bytes:
        """序列化为字节"""
        return MessageSerializer.serialize(message).encode('utf-8')
    
    @staticmethod
    def from_bytes(data: bytes) -> AgentMessageV2:
        """从字节反序列化"""
        return MessageSerializer.deserialize(data.decode('utf-8'))
```

### 2.2 消息队列

```python
import aio_pika
import asyncio
from typing import Callable

class MessageQueue:
    """消息队列"""
    
    def __init__(self, rabbitmq_url: str = "amqp://guest:guest@localhost/"):
        self.rabbitmq_url = rabbitmq_url
        self.connection: aio_pika.Connection = None
        self.channel: aio_pika.Channel = None
        self.queues: Dict[str, aio_pika.Queue] = {}
        self.exchanges: Dict[str, aio_pika.Exchange] = {}
    
    async def connect(self):
        """连接到 RabbitMQ"""
        self.connection = await aio_pika.connect_robust(self.rabbitmq_url)
        self.channel = await self.connection.channel()
        await self.channel.set_qos(prefetch_count=100)
    
    async def declare_queue(self, queue_name: str, durable: bool = True):
        """声明队列"""
        queue = await self.channel.declare_queue(
            queue_name,
            durable=durable
        )
        self.queues[queue_name] = queue
        return queue
    
    async def declare_exchange(self, exchange_name: str, exchange_type: str = "direct"):
        """声明交换机"""
        exchange = await self.channel.declare_exchange(
            exchange_name,
            aio_pika.ExchangeType(exchange_type),
            durable=True
        )
        self.exchanges[exchange_name] = exchange
        return exchange
    
    async def publish(
        self,
        message: AgentMessageV2,
        exchange_name: str,
        routing_key: str
    ):
        """发布消息"""
        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            raise ValueError(f"Exchange {exchange_name} not found")
        
        # 创建 RabbitMQ 消息
        rabbitmq_message = aio_pika.Message(
            MessageSerializer.to_bytes(message),
            content_type='application/json',
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            headers={
                'message_type': message.message_type,
                'priority': message.priority
            }
        )
        
        # 发布
        await exchange.publish(
            rabbitmq_message,
            routing_key=routing_key
        )
    
    async def consume(
        self,
        queue_name: str,
        callback: Callable,
        prefetch_count: int = 10
    ):
        """消费消息"""
        queue = self.queues.get(queue_name)
        if not queue:
            raise ValueError(f"Queue {queue_name} not found")
        
        await queue.set_qos(prefetch_count=prefetch_count)
        
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        # 反序列化
                        agent_message = MessageSerializer.from_bytes(message.body)
                        
                        # 调用回调
                        await callback(agent_message)
                    except Exception as e:
                        print(f"Error processing message: {e}")
                        # 消息会被重新入队
    
    async def close(self):
        """关闭连接"""
        if self.connection:
            await self.connection.close()

# 使用示例
async def message_queue_example():
    """消息队列示例"""
    mq = MessageQueue("amqp://guest:guest@localhost/")
    await mq.connect()
    
    # 声明队列和交换机
    await mq.declare_queue("agent_a_queue")
    await mq.declare_exchange("agent_exchange", "direct")
    
    # 绑定队列到交换机
    queue = mq.queues["agent_a_queue"]
    exchange = mq.exchanges["agent_exchange"]
    await queue.bind(exchange, routing_key="agent_a")
    
    # 发布消息
    message = AgentMessageV2(
        sender="agent_b",
        recipient="agent_a",
        message_type=MessageType.TASK,
        content={"task": "process data"}
    )
    
    await mq.publish(message, "agent_exchange", "agent_a")
    
    # 消费消息
    async def handle_message(msg: AgentMessageV2):
        print(f"Received: {msg.content}")
    
    await mq.consume("agent_a_queue", handle_message)
```

### 2.3 事件总线

```python
from typing import Callable, Dict, List
import asyncio

class EventBus:
    """事件总线"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[Dict] = []
    
    def subscribe(self, event_type: str, callback: Callable):
        """订阅事件"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """取消订阅"""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(callback)
    
    async def publish(self, event_type: str, data: Dict):
        """发布事件"""
        event = {
            "event_type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        
        # 记录事件
        self.event_history.append(event)
        
        # 通知订阅者
        if event_type in self.subscribers:
            tasks = [
                callback(event)
                for callback in self.subscribers[event_type]
            ]
            await asyncio.gather(*tasks)
    
    def get_event_history(self, event_type: str = None) -> List[Dict]:
        """获取事件历史"""
        if event_type:
            return [e for e in self.event_history if e["event_type"] == event_type]
        return self.event_history

# 使用示例
async def event_bus_example():
    """事件总线示例"""
    bus = EventBus()
    
    # 订阅事件
    def on_task_completed(event):
        print(f"Task completed: {event['data']}")
    
    def on_error(event):
        print(f"Error occurred: {event['data']}")
    
    bus.subscribe("task.completed", on_task_completed)
    bus.subscribe("task.error", on_error)
    
    # 发布事件
    await bus.publish("task.completed", {
        "task_id": "123",
        "result": "success"
    })
    
    await bus.publish("task.error", {
        "task_id": "456",
        "error": "timeout"
    })
    
    # 查看历史
    history = bus.get_event_history()
    print(f"Event history: {history}")
```

## 3. 高级通信模式

### 3.1 请求-响应模式

```python
class RequestResponsePattern:
    """请求-响应模式"""
    
    def __init__(self, message_queue: MessageQueue):
        self.mq = message_queue
        self.pending_requests: Dict[str, asyncio.Future] = {}
    
    async def send_request(
        self,
        request: AgentMessageV2,
        timeout: float = 30.0
    ) -> AgentMessageV2:
        """发送请求并等待响应"""
        # 创建 Future
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[request.message_id] = future
        
        # 设置回复地址
        request.reply_to = f"response_queue_{request.message_id}"
        
        # 发送请求
        await self.mq.publish(
            request,
            exchange_name="request_exchange",
            routing_key=request.recipient
        )
        
        # 等待响应
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            del self.pending_requests[request.message_id]
            raise TimeoutError(f"Request {request.message_id} timed out")
    
    async def handle_response(self, response: AgentMessageV2):
        """处理响应"""
        correlation_id = response.correlation_id
        if correlation_id in self.pending_requests:
            future = self.pending_requests[correlation_id]
            future.set_result(response)
            del self.pending_requests[correlation_id]
    
    async def send_response(
        self,
        request: AgentMessageV2,
        response_data: Any,
        success: bool = True
    ):
        """发送响应"""
        response = ResultMessage(
            sender=request.recipient,
            recipient=request.sender,
            task_id=request.message_id,
            result=response_data,
            success=success,
            correlation_id=request.message_id
        )
        
        await self.mq.publish(
            response,
            exchange_name="response_exchange",
            routing_key=request.reply_to or request.sender
        )

# 使用示例
async def request_response_example():
    """请求-响应示例"""
    mq = MessageQueue()
    await mq.connect()
    
    pattern = RequestResponsePattern(mq)
    
    # 发送请求
    request = AgentMessageV2(
        sender="client",
        recipient="server",
        message_type=MessageType.QUERY,
        content={"query": "What is the temperature?"}
    )
    
    try:
        response = await pattern.send_request(request, timeout=10.0)
        print(f"Response: {response.content}")
    except TimeoutError:
        print("Request timed out")
```

### 3.2 发布-订阅模式

```python
class PubSubPattern:
    """发布-订阅模式"""
    
    def __init__(self, message_queue: MessageQueue):
        self.mq = message_queue
    
    async def publish_event(self, event_type: str, data: Dict):
        """发布事件到所有订阅者"""
        message = AgentMessageV2(
            sender="publisher",
            recipient="*",  # 广播
            message_type=MessageType.NOTIFICATION,
            content={
                "event_type": event_type,
                "data": data
            }
        )
        
        await self.mq.publish(
            message,
            exchange_name="pubsub_exchange",
            routing_key=event_type  # 路由键作为事件类型
        )
    
    async def subscribe_to_events(
        self,
        event_types: List[str],
        callback: Callable
    ):
        """订阅多个事件类型"""
        for event_type in event_types:
            queue_name = f"subscriber_{event_type}"
            await self.mq.declare_queue(queue_name)
            
            # 绑定到交换机
            queue = self.mq.queues[queue_name]
            exchange = self.mq.exchanges["pubsub_exchange"]
            await queue.bind(exchange, routing_key=event_type)
            
            # 消费
            await self.mq.consume(queue_name, callback)

# 使用示例
async def pubsub_example():
    """发布-订阅示例"""
    mq = MessageQueue()
    await mq.connect()
    
    pattern = PubSubPattern(mq)
    
    # 订阅
    async def handle_event(event: AgentMessageV2):
        print(f"Received event: {event.content}")
    
    await pattern.subscribe_to_events(
        ["task.completed", "task.failed"],
        handle_event
    )
    
    # 发布
    await pattern.publish_event("task.completed", {
        "task_id": "123",
        "agent": "worker_1"
    })
```

### 3.3 广播模式

```python
class BroadcastPattern:
    """广播模式"""
    
    def __init__(self, message_queue: MessageQueue):
        self.mq = message_queue
    
    async def broadcast(
        self,
        sender: str,
        message: Dict,
        priority: Priority = Priority.NORMAL
    ):
        """广播消息到所有 Agent"""
        broadcast_message = AgentMessageV2(
            sender=sender,
            recipient="*",
            message_type=MessageType.NOTIFICATION,
            priority=priority,
            content=message
        )
        
        await self.mq.publish(
            broadcast_message,
            exchange_name="broadcast_exchange",
            routing_key=""  # 空路由键表示广播
        )
    
    async def receive_broadcasts(
        self,
        agent_id: str,
        callback: Callable
    ):
        """接收广播"""
        queue_name = f"broadcast_{agent_id}"
        await self.mq.declare_queue(queue_name)
        
        # 绑定到广播交换机
        queue = self.mq.queues[queue_name]
        exchange = self.mq.exchanges["broadcast_exchange"]
        await queue.bind(exchange, routing_key="")
        
        await self.mq.consume(queue_name, callback)
```

## 4. 容错与恢复

### 4.1 消息重试

```python
class RetryHandler:
    """重试处理器"""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    async def execute_with_retry(
        self,
        message: AgentMessageV2,
        handler: Callable
    ) -> bool:
        """带重试执行"""
        for attempt in range(message.max_retries):
            try:
                await handler(message)
                return True
            except Exception as e:
                message.retry_count += 1
                
                if message.retry_count >= message.max_retries:
                    print(f"Max retries reached for message {message.message_id}")
                    return False
                
                # 指数退避
                wait_time = self.backoff_factor ** attempt
                print(f"Retry {message.retry_count} after {wait_time}s")
                await asyncio.sleep(wait_time)
        
        return False
```

### 4.2 死信队列

```python
class DeadLetterQueue:
    """死信队列"""
    
    def __init__(self, message_queue: MessageQueue):
        self.mq = message_queue
    
    async def setup_dead_letter_queue(self):
        """设置死信队列"""
        # 主队列
        await self.mq.declare_queue("main_queue")
        
        # 死信队列
        await self.mq.declare_queue("dead_letter_queue")
        
        # 绑定死信交换机
        queue = self.mq.queues["main_queue"]
        await queue.bind(
            self.mq.exchanges["dead_letter_exchange"],
            routing_key="dead_letter"
        )
    
    async def send_to_dead_letter(self, message: AgentMessageV2, error: str):
        """发送到死信队列"""
        dead_letter_message = AgentMessageV2(
            sender="system",
            recipient="dead_letter_handler",
            message_type=MessageType.ERROR,
            content={
                "original_message": message.dict(),
                "error": error,
                "failed_at": time.time()
            }
        )
        
        await self.mq.publish(
            dead_letter_message,
            exchange_name="dead_letter_exchange",
            routing_key="dead_letter"
        )
    
    async def process_dead_letters(self, callback: Callable):
        """处理死信"""
        await self.mq.consume("dead_letter_queue", callback)
```

## 5. 监控与可观测性

### 5.1 消息追踪

```python
class MessageTracer:
    """消息追踪器"""
    
    def __init__(self):
        self.traces: Dict[str, List[Dict]] = {}
    
    def trace_message(self, message: AgentMessageV2, action: str):
        """追踪消息"""
        if message.correlation_id not in self.traces:
            self.traces[message.correlation_id] = []
        
        self.traces[message.correlation_id].append({
            "message_id": message.message_id,
            "sender": message.sender,
            "recipient": message.recipient,
            "action": action,
            "timestamp": message.timestamp
        })
    
    def get_trace(self, correlation_id: str) -> List[Dict]:
        """获取追踪记录"""
        return self.traces.get(correlation_id, [])
    
    def visualize_trace(self, correlation_id: str):
        """可视化追踪"""
        trace = self.get_trace(correlation_id)
        
        print(f"Trace for {correlation_id}:")
        for step in trace:
            print(f"  {step['timestamp']}: {step['sender']} -> {step['recipient']} ({step['action']})")

# Mermaid 图
trace_mermaid = """
graph TD
    A[Agent A] -->|Request| B[Agent B]
    B -->|Process| B
    B -->|Response| A
    A -->|Task| C[Agent C]
    C -->|Result| A
    
    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style C fill:#bfb,stroke:#333
"""
```

## 6. 性能优化

### 6.1 批量处理

```python
class BatchMessageProcessor:
    """批量消息处理器"""
    
    def __init__(self, batch_size: int = 100, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.message_buffer = []
        self.last_flush = time.time()
    
    async def add_message(self, message: AgentMessageV2):
        """添加消息到缓冲"""
        self.message_buffer.append(message)
        
        # 检查是否需要刷新
        if (len(self.message_buffer) >= self.batch_size or
            time.time() - self.last_flush >= self.flush_interval):
            await self.flush()
    
    async def flush(self):
        """刷新缓冲"""
        if not self.message_buffer:
            return
        
        # 批量处理
        await self.process_batch(self.message_buffer)
        
        self.message_buffer = []
        self.last_flush = time.time()
    
    async def process_batch(self, messages: List[AgentMessageV2]):
        """处理批次"""
        # 实现批量处理逻辑
        pass
```

## 7. 参考资料

### 协议与标准

- **FIPA ACL**: Foundation for Intelligent Physical Agents
- **KQML**: Knowledge Query and Manipulation Language
- **AMQP**: Advanced Message Queuing Protocol
- **MQTT**: Message Queuing Telemetry Transport

### 最佳实践

1. 使用标准化的消息格式
2. 实现完善的错误处理和重试
3. 监控消息队列深度和延迟
4. 使用死信队列处理失败消息
5. 实施消息追踪和调试
6. 定期清理过期消息
7. 进行负载测试确保性能

---

> 📅 **最后更新**: 2026-06
> 📊 **难度等级**: Level 4-5（高级-专家）
> 🔗 **相关文档**: 
> - [多 Agent 系统](../01-多Agent系统/README.md)
> - [MCP 协议](../03-MCP协议与Skills/README.md)
> - [Agent 工程化框架](../04-Agent工程化框架/README.md)
