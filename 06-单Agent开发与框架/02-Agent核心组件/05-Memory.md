# 记忆系统设计

> 📅 **更新时间**: 2026-06-17  

---

## 目录

- [1. 为什么 Agent 需要记忆](#1-为什么-agent-需要记忆)
- [2. 从人类记忆到 Agent 记忆](#2-从人类记忆到-agent-记忆)
- [3. 记忆系统架构设计](#3-记忆系统架构设计)
- [4. 2026年主流记忆方案对比](#4-2026年主流记忆方案对比)
- [5. 实战：构建记忆系统](#5-实战构建记忆系统)
- [6. 记忆与 MCP 集成](#6-记忆与-mcp-集成)
- [7. 最佳实践与常见陷阱](#7-最佳实践与常见陷阱)
- [8. 总结](#8-总结)

---

## 1. 为什么 Agent 需要记忆

### 1.1 LLM 的两大核心限制

当前基于 LLM 的 Agent 面临两个根本性挑战：

**限制 1：无状态导致的对话遗忘**

LLM 本质上是**无状态**的，每次请求都是独立计算：

```python
# ❌ 问题示例
from openai import OpenAI

client = OpenAI()

# 第一次对话
response1 = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "我叫张三，正在学习Python"}]
)

# 第二次对话（新会话）
response2 = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "你还记得我叫什么吗？"}]
)
# 输出：抱歉，我不知道您的名字...
```

**核心问题**：
- ❌ 上下文窗口有限，长对话会丢失早期信息
- ❌ 无法记住用户偏好和习惯
- ❌ 不能从历史经验中学习
- ❌ 多轮对话可能出现矛盾

**限制 2：内置知识的局限性**

LLM 的知识来自训练数据，存在明显边界：

```
训练数据截止: 2024年
↓
无法访问最新信息
↓
领域知识深度不足
↓
缺乏私有数据
```

### 1.2 记忆系统的价值

根据 MongoDB 和 The New Stack 的 2026 年最新报告，引入记忆系统后：

| 指标 | 无记忆 | 有记忆 | 提升 |
|------|--------|--------|------|
| 用户满意度 | 62% | 89% | +43% |
| 任务完成率 | 58% | 85% | +47% |
| 对话轮次 | 3.2轮 | 12.5轮 | +290% |
| 个性化评分 | 2.1/5 | 4.3/5 | +105% |

**记忆系统让 Agent 从"工具"进化为"伙伴"**。

---

## 2. 从人类记忆到 Agent 记忆

### 2.1 认知科学的启示

认知心理学将人类记忆分为四个层次：

```
人类记忆系统
├── 感官记忆 (Sensory Memory)
│   ├── 持续时间: 0.5-3秒
│   ├── 容量: 巨大
│   └── 作用: 临时存储所有感官信息
│
├── 工作记忆 (Working Memory)
│   ├── 持续时间: 15-30秒
│   ├── 容量: 7±2 个项目
│   └── 作用: 当前任务的信息处理
│
├── 长期记忆 (Long-term Memory)
│   ├── 程序性记忆 (Procedural)
│   │   └── 技能和习惯（如骑自行车）
│   │
│   └── 陈述性记忆 (Declarative)
│       ├── 语义记忆 (Semantic)
│       │   └── 通用知识（如"巴黎是法国首都"）
│       │
│       └── 情景记忆 (Episodic)
│           └── 个人经历（如"昨天的会议内容"）
```

### 2.2 Agent 记忆的映射

借鉴人类记忆，Agent 记忆系统设计为：

```python
Agent 记忆系统
├── 感知记忆 (Perceptual Memory)
│   ├── 对应: 感官记忆
│   ├── 存储: 多模态数据（图像、音频、视频）
│   └── 示例: 用户上传的图片、语音记录
│
├── 工作记忆 (Working Memory)
│   ├── 对应: 工作记忆
│   ├── 存储: 当前对话上下文
│   ├── TTL: 会话级别（会话结束即清除）
│   └── 示例: "用户正在询问Python安装问题"
│
├── 情景记忆 (Episodic Memory)
│   ├── 对应: 情景记忆
│   ├── 存储: 特定事件和交互
│   ├── 特征: 时间序列、可追溯
│   └── 示例: "2026-06-15 用户学习了Python基础语法"
│
├── 语义记忆 (Semantic Memory)
│   ├── 对应: 语义记忆
│   ├── 存储: 抽象知识和概念
│   ├── 特征: 结构化、可推理
│   └── 示例: "Python是一种解释型语言"
│
└── 程序记忆 (Procedural Memory)
    ├── 对应: 程序性记忆
    ├── 存储: 技能和偏好
    ├── 特征: 模式化、可复用
    └── 示例: "用户偏好使用VS Code编辑器"
```

---

## 3. 记忆系统架构设计

### 3.1 四层架构

根据业界最佳实践，推荐四层架构：

```
┌─────────────────────────────────────────────────┐
│           应用层 (Application Layer)              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ MemoryTool  │  │ RAGTool     │  │ NoteTool │ │
│  └──────┬──────┘  └──────┬──────┘  └────┬─────┘ │
└─────────┼────────────────┼──────────────┼───────┘
          │                │              │
┌─────────┼────────────────┼──────────────┼───────┐
│      记忆类型层 (Memory Types Layer)       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Working  │ │Episodic  │ │ Semantic │  │
│  │ Memory   │ │ Memory   │ │ Memory   │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘  │
└───────┼────────────┼────────────┼─────────┘
        │            │            │
┌───────┼────────────┼────────────┼─────────┐
│     存储后端层 (Storage Backend Layer)      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Vector   │ │  Graph   │ │ Document │  │
│  │ Store    │ │  Store   │ │  Store   │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘  │
└───────┼────────────┼────────────┼─────────┘
        │            │            │
┌───────┼────────────┼────────────┼─────────┐
│    基础设施层 (Infrastructure Layer)        │
│  ┌─────────────┐  ┌──────────────┐        │
│  │MemoryManager│  │EmbeddingSvc  │        │
│  └─────────────┘  └──────────────┘        │
└───────────────────────────────────────────┘
```

### 3.2 核心组件实现

#### MemoryItem - 记忆数据结构

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

class MemoryType(Enum):
    WORKING = "working"      # 工作记忆
    EPISODIC = "episodic"    # 情景记忆
    SEMANTIC = "semantic"    # 语义记忆
    PROCEDURAL = "procedural" # 程序记忆
    PERCEPTUAL = "perceptual" # 感知记忆

class MemoryScope(Enum):
    SESSION = "session"   # 会话级
    USER = "user"         # 用户级
    GLOBAL = "global"     # 全局级

@dataclass
class MemoryItem:
    """记忆项数据结构"""
    id: str
    type: MemoryType
    content: str
    metadata: dict = field(default_factory=dict)
    scope: MemoryScope = MemoryScope.SESSION
    importance: float = 0.5  # 重要性评分 0-1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    ttl: Optional[int] = None  # 生存时间（秒）
    embedding: Optional[list[float]] = None  # 向量嵌入
    
    def is_expired(self) -> bool:
        """检查记忆是否过期"""
        if self.ttl is None:
            return False
        elapsed = (datetime.utcnow() - self.updated_at).total_seconds()
        return elapsed > self.ttl
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "scope": self.scope.value,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
```

#### MemoryManager - 记忆管理器

```python
import uuid
from typing import List, Optional
from datetime import datetime

class MemoryManager:
    """记忆系统管理器"""
    
    def __init__(self):
        self.memories: dict[str, MemoryItem] = {}
        self.working_memory: list[MemoryItem] = []
        self.episodic_memory: list[MemoryItem] = []
        self.semantic_memory: list[MemoryItem] = []
    
    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: dict = None,
        importance: float = 0.5,
        ttl: int = None
    ) -> MemoryItem:
        """添加记忆"""
        memory = MemoryItem(
            id=str(uuid.uuid4()),
            type=memory_type,
            content=content,
            metadata=metadata or {},
            importance=importance,
            ttl=ttl
        )
        
        # 存储记忆
        self.memories[memory.id] = memory
        
        # 根据类型分类存储
        if memory_type == MemoryType.WORKING:
            self.working_memory.append(memory)
        elif memory_type == MemoryType.EPISODIC:
            self.episodic_memory.append(memory)
        elif memory_type == MemoryType.SEMANTIC:
            self.semantic_memory.append(memory)
        
        return memory
    
    async def retrieve_memories(
        self,
        query: str,
        memory_types: List[MemoryType] = None,
        limit: int = 5,
        min_importance: float = 0.3
    ) -> List[MemoryItem]:
        """检索记忆"""
        # 过滤过期记忆
        active_memories = [
            m for m in self.memories.values()
            if not m.is_expired() and m.importance >= min_importance
        ]
        
        # 按类型过滤
        if memory_types:
            active_memories = [
                m for m in active_memories
                if m.type in memory_types
            ]
        
        # 按重要性排序
        active_memories.sort(key=lambda m: m.importance, reverse=True)
        
        return active_memories[:limit]
    
    async def consolidate_memories(self):
        """记忆巩固 - 将工作记忆转为长期记忆"""
        for memory in self.working_memory:
            if memory.importance > 0.7:
                # 重要记忆转为情景记忆
                memory.type = MemoryType.EPISODIC
                memory.scope = MemoryScope.USER
                self.episodic_memory.append(memory)
        
        # 清理工作记忆
        self.working_memory.clear()
    
    async def forget_unimportant(self, threshold: float = 0.2):
        """遗忘不重要的记忆"""
        to_remove = [
            mid for mid, memory in self.memories.items()
            if memory.importance < threshold and memory.is_expired()
        ]
        
        for mid in to_remove:
            del self.memories[mid]
        
        return len(to_remove)
```

---

## 4. 2026年主流记忆方案对比

根据腾讯云开发者社区和 CSDN 的最新评测：

### 4.1 五大方案横评

| 方案 | 准确率 | 响应时间 | 部署难度 | 适用场景 |
|------|--------|----------|----------|----------|
| **OpenClaw** | 72.3% | 45ms | 中 | 通用Agent |
| **Hermes** | 68.5% | 38ms | 低 | 轻量级应用 |
| **Memori** | 74.1% | 52ms | 高 | 企业级 |
| **OpenViking** | 70.8% | 41ms | 中 | 多模态 |
| **腾讯云AgentMemory** | **76.1%** | **35ms** | **低** | **全场景** |

### 4.2 三代技术演进

```
第一代：向量记忆 (2023-2024)
├── 代表: LangChain Memory
├── 特点: 简单向量检索
├── 优点: 实现简单
└── 缺点: 缺乏结构化、无法推理

第二代：结构化记忆 (2024-2025)
├── 代表: MemGPT/Letta, Graphiti
├── 特点: 分层存储、图结构
├── 优点: 支持复杂查询
└── 缺点: 维护成本高

第三代：认知架构 (2025-2026) ← 当前主流
├── 代表: OpenClaw, Claude Code, Hermes
├── 特点: 融合情景+语义+动态调度
├── 优点: 接近人类记忆机制
└── 缺点: 实现复杂
```

### 4.3 技术选型指南

```python
# 选型决策树
def select_memory_solution(requirements: dict) -> str:
    """根据需求选择记忆方案"""
    
    if requirements.get("budget") == "low":
        return "Hermes"  # 轻量、免费
    
    if requirements.get("multimodal"):
        return "OpenViking"  # 多模态支持
    
    if requirements.get("enterprise"):
        return "腾讯云AgentMemory"  # 企业级、高准确率
    
    if requirements.get("customization"):
        return "OpenClaw"  # 高度可定制
    
    return "Memori"  # 默认选择
```

---

## 5. 实战：构建记忆系统

### 5.1 完整示例：智能助手记忆

```python
from datetime import datetime
from typing import List

class IntelligentAssistant:
    """带记忆系统的智能助手"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.user_profile = {}
    
    async def chat(self, user_input: str) -> str:
        """对话处理"""
        # 1. 检索相关记忆
        relevant_memories = await self.memory_manager.retrieve_memories(
            query=user_input,
            memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
            limit=3
        )
        
        # 2. 构建上下文
        context = self._build_context(user_input, relevant_memories)
        
        # 3. 调用 LLM
        response = await self._call_llm(context)
        
        # 4. 存储对话记忆
        await self.memory_manager.add_memory(
            content=f"User: {user_input}\nAssistant: {response}",
            memory_type=MemoryType.WORKING,
            importance=self._calculate_importance(user_input, response)
        )
        
        return response
    
    def _build_context(self, query: str, memories: List[MemoryItem]) -> str:
        """构建上下文"""
        context = f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        if memories:
            context += "相关记忆:\n"
            for memory in memories:
                context += f"- [{memory.type.value}] {memory.content}\n"
        
        context += f"\n用户问题: {query}\n"
        return context
    
    def _calculate_importance(self, user_input: str, response: str) -> float:
        """计算记忆重要性"""
        importance = 0.5
        
        # 包含个人信息，重要性高
        if any(keyword in user_input.lower() 
               for keyword in ["我叫", "我喜欢", "我的工作", "我的"]):
            importance = 0.8
        
        # 包含技术问题，重要性中
        if any(keyword in user_input.lower()
               for keyword in ["如何", "怎么", "为什么", "什么是"]):
            importance = 0.6
        
        return importance
    
    async def end_session(self):
        """会话结束 - 记忆巩固"""
        await self.memory_manager.consolidate_memories()
        await self.memory_manager.forget_unimportant(threshold=0.2)
```

### 5.2 记忆工具 MCP 实现

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("memory-tool")

@mcp.tool()
async def add_memory(
    content: str,
    type: str = "episodic",
    importance: float = 0.5
) -> str:
    """添加记忆"""
    memory_type = MemoryType(type)
    memory = await memory_manager.add_memory(
        content=content,
        memory_type=memory_type,
        importance=importance
    )
    return f"✅ 记忆已添加: {memory.id}"

@mcp.tool()
async def search_memories(
    query: str,
    limit: int = 5
) -> str:
    """搜索记忆"""
    memories = await memory_manager.retrieve_memories(
        query=query,
        limit=limit
    )
    
    if not memories:
        return "未找到相关记忆"
    
    result = "相关记忆:\n"
    for m in memories:
        result += f"- [{m.type.value}] {m.content} (重要性: {m.importance})\n"
    
    return result

@mcp.tool()
async def clear_working_memory() -> str:
    """清理工作记忆"""
    count = len(memory_manager.working_memory)
    memory_manager.working_memory.clear()
    return f"✅ 已清理 {count} 条工作记忆"
```

---

## 6. 记忆与 MCP 集成

### 6.1 MCP Memory Server

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["/path/to/memory_server.py"]
    }
  }
}
```

### 6.2 使用示例

```python
# Claude Desktop 配置
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}

# 在对话中使用
"""
用户: 记住我叫张三，是一名Python开发者

AI: [调用 memory/add_memory]
    - 类型: episodic
    - 内容: "用户叫张三，是Python开发者"
    - 重要性: 0.8

用户: 你还记得我做什么的吗？

AI: [调用 memory/search_memories]
    - 查询: "职业 工作"
    - 返回: "张三是一名Python开发者"
    
AI: 记得！你是一名Python开发者。
"""
```

---

## 7. 最佳实践与常见陷阱

### 7.1 最佳实践

#### ✅ 1. 分层记忆管理

```python
# ✅ 正确：按重要性分层
async def smart_memory_management(content: str):
    # 即时信息 → 工作记忆
    if is_conversation_context(content):
        await add_memory(content, MemoryType.WORKING, ttl=3600)
    
    # 重要事实 → 语义记忆
    elif is_factual_information(content):
        await add_memory(content, MemoryType.SEMANTIC, importance=0.8)
    
    # 个人经历 → 情景记忆
    elif is_personal_experience(content):
        await add_memory(content, MemoryType.EPISODIC, importance=0.9)
```

#### ✅ 2. 定期记忆巩固

```python
# ✅ 正确：会话结束时巩固
async def on_session_end():
    # 工作记忆 → 情景记忆
    await consolidate_working_to_episodic()
    
    # 清理过期记忆
    await forget_unimportant_memories()
    
    # 更新用户画像
    await update_user_profile()
```

#### ✅ 3. 重要性评分

```python
# ✅ 正确：动态计算重要性
def calculate_importance(content: str, context: dict) -> float:
    score = 0.5
    
    # 包含个人信息 +0.3
    if has_personal_info(content):
        score += 0.3
    
    # 用户明确要求记住 +0.2
    if "记住" in content or "remember" in content.lower():
        score += 0.2
    
    # 重复出现的信息 +0.1
    if is_repeated_information(content):
        score += 0.1
    
    return min(score, 1.0)
```

### 7.2 常见陷阱

#### ❌ 陷阱 1：记忆无限增长

```python
# ❌ 错误：只添加不清理
async def bad_memory_management(content: str):
    await add_memory(content)  # 没有TTL、没有清理
    # 结果：内存爆炸、检索变慢

# ✅ 正确：设置TTL和清理策略
async def good_memory_management(content: str):
    await add_memory(
        content,
        ttl=86400,  # 24小时过期
        importance=0.5
    )
    
    # 定期清理
    if memory_count > 1000:
        await forget_unimportant(threshold=0.3)
```

#### ❌ 陷阱 2：检索所有记忆

```python
# ❌ 错误：返回所有记忆
async def bad_retrieval(query: str):
    all_memories = await get_all_memories()  # 可能上千条
    return all_memories  # 上下文爆炸

# ✅ 正确：限制数量、按相关性排序
async def good_retrieval(query: str):
    memories = await search_memories(
        query=query,
        limit=5,  # 只返回Top 5
        min_importance=0.4  # 最低重要性
    )
    return memories
```

#### ❌ 陷阱 3：忽视隐私

```python
# ❌ 错误：存储敏感信息
await add_memory("用户密码是123456")  # 危险！

# ✅ 正确：过滤敏感信息
async def safe_add_memory(content: str):
    # 过滤敏感信息
    if contains_sensitive_info(content):
        content = mask_sensitive_info(content)
    
    await add_memory(content)
```

---

## 8. 总结

记忆系统是 Agent 从"工具"进化为"伙伴"的关键。2026 年的最佳实践：

1. **采用认知架构** - 借鉴人类记忆的分层设计
2. **定期记忆巩固** - 工作记忆 → 长期记忆
3. **动态重要性评分** - 智能管理记忆价值
4. **MCP 集成** - 标准化记忆工具接口
5. **隐私保护** - 过滤敏感信息

**下一步**：
- 学习 [RAG系统集成](./04-RAG系统集成.md)
- 了解 [上下文工程](./05-上下文工程.md)
- 实践 [MCP协议与工具调用](./02-MCP协议与工具调用.md)

---

> 📚 **参考资源**：
> - Hello-Agents Chapter 8: Memory and Retrieval
> - MongoDB & The New Stack: 2026 Agent Memory Report
> - 腾讯云: 2026年Agent记忆系统方案横评
> - CSDN: 2026年版AI Agent记忆技术演进全解析
>
> 🤝 **贡献**：欢迎提交 PR 补充更多实践案例！
