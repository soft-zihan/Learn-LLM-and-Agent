# Agent 学习路线图

> 📅 **更新时间**: 2026-06-17

---

## 目录

- [1. 📋 目录](#1-目录)
- [2. 学习路径总览](#2-学习路径总览)
- [3. 阶段 1：基础入门](#3-阶段-1基础入门)
- [4. 阶段 2：单 Agent 开发](#4-阶段-2单-agent-开发)
- [5. 阶段 3：高级 Agent 技术](#5-阶段-3高级-agent-技术)
- [6. 阶段 4：多 Agent 协作](#6-阶段-4多-agent-协作)

---

## 1. 📋 目录

- [1. 学习路径总览](#1-学习路径总览)
- [2. 阶段 1：基础入门](#2-阶段-1基础入门)
- [3. 阶段 2：单 Agent 开发](#3-阶段-2单-agent-开发)
- [4. 阶段 3：高级 Agent 技术](#4-阶段-3高级-agent-技术)
- [5. 阶段 4：多 Agent 协作](#5-阶段-4多-agent-协作)

---

## 2. 学习路径总览

### 1.1 学习路径全景图

```
阶段 0: 编程基础 ──────────────────────────────────────▶ 预备条件
         Python/TypeScript、API 基础、Git

阶段 1: 基础入门 (4-6 周) ───────────────────────────────▶ 初学者
         LLM 基础、Agent 概念、Prompt Engineering、工具调用

阶段 2: 单 Agent 开发 (6-8 周) ──────────────────────────▶ 初级开发者
         LangChain、LangGraph、记忆系统、RAG 基础

阶段 3: 高级 Agent 技术 (8-10 周) ───────────────────────▶ 中级开发者
         Harness Engineering、Nanobot、Skills 开发、MCP 协议

阶段 4: 多 Agent 协作 (6-8 周) ──────────────────────────▶ 中高级开发者
         CrewAI、AutoGen、OpenAI Agents、Agent 编排

阶段 5: 生产部署 (8-10 周) ─────────────────────────────▶ 高级工程师
         推理优化、服务化、监控、安全与对齐

阶段 6: 前沿技术 (持续) ────────────────────────────────▶ 专家
         多模态、MoE、推理模型、MCP 生态、Agent 工程化
```

### 1.2 学习时间估算

| 阶段 | 预计时间 | 目标水平 | 关键产出 |
|------|----------|----------|----------|
| 阶段 1 | 4-6 周 | 理解 Agent 基础概念 | 能使用 API 构建简单 Agent |
| 阶段 2 | 6-8 周 | 掌握单 Agent 开发 | 能构建具备记忆和工具的 Agent |
| 阶段 3 | 8-10 周 | 掌握高级 Agent 技术 | 能开发复杂 Skills 和 MCP 服务 |
| 阶段 4 | 6-8 周 | 掌握多 Agent 协作 | 能设计和实现多 Agent 系统 |
| 阶段 5 | 8-10 周 | 掌握生产部署 | 能将 Agent 系统部署到生产环境 |
| 阶段 6 | 持续学习 | 前沿技术探索 | 能跟进最新研究并应用到实际项目 |

**总学习时间**: 约 32-42 周（8-10 个月）达到高级工程师水平

### 1.3 学习路径决策树

```
你是否已掌握 Python/TypeScript 编程？
├─ 否 → 先学习编程基础（2-4 周）
└─ 是 ↓

你是否了解 LLM 基础概念？
├─ 否 → 从阶段 1 开始
└─ 是 ↓

你是否使用过 LangChain 或类似框架？
├─ 否 → 从阶段 2 开始
└─ 是 ↓

你是否构建过多 Agent 系统？
├─ 否 → 从阶段 3 或 4 开始
└─ 是 ↓

你是否将 Agent 系统部署到生产环境？
├─ 否 → 从阶段 5 开始
└─ 是 → 从阶段 6 开始，探索前沿技术
```

### 1.4 学习原则

1. **循序渐进**: 不要跳过基础阶段，每个阶段都是后续阶段的基础
2. **实践为主**: 每个知识点都要通过代码实践来巩固
3. **项目驱动**: 通过实际项目来整合所学知识
4. **持续更新**: Agent 技术发展迅速，需要持续学习新技术
5. **社区参与**: 加入学习社区，与他人交流经验和解决问题

---

## 3. 阶段 1：基础入门

> **目标**: 理解 LLM 和 Agent 的基础概念，能够使用 API 构建简单的 Agent 应用  
> **时间**: 4-6 周  
> **前置知识**: Python 或 TypeScript 编程基础

### 2.1 LLM 基础

#### 2.1.1 核心概念

**学习内容**:
- Transformer 架构基础
  - Self-Attention 机制
  - Multi-Head Attention
  - Position Encoding
  - Encoder-Decoder 架构
  
- LLM 工作原理
  - Tokenization（分词）
  - Next Token Prediction
  - Context Window（上下文窗口）
  - Temperature 和 Top-p 采样
  
- 主流模型系列
  - OpenAI GPT 系列（GPT-3.5/4/4o）
  - Anthropic Claude 系列
  - Meta Llama 系列（Llama 2/3/3.1）
  - Google Gemini 系列
  - 中国模型：通义千问、文心一言、智谱 GLM

**关键知识点**:
```python
# Tokenization 示例
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "Hello, Agent World!"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")  # [15496, 11, 3135, 21873, 0]

# 理解上下文窗口限制
# GPT-3.5: 4K/16K tokens
# GPT-4: 8K/32K tokens  
# GPT-4 Turbo: 128K tokens
# Claude 3: 200K tokens
```

**延伸阅读**:
- 📖 LLM 预训练基础.md
- 📖 深入了解 Transformer 架构

#### 2.1.2 Prompt Engineering 基础

**学习内容**:
- 基础 Prompt 设计原则
  - 清晰明确的指令
  - 提供上下文和示例
  - 指定输出格式
  
- 常用 Prompt 技巧
  - Zero-shot Prompting
  - Few-shot Prompting
  - Chain-of-Thought (CoT)
  - Role Prompting
  
- Prompt 模板
  - 变量替换
  - 条件逻辑
  - 结构化输出

**实践示例**:
```python
# Zero-shot Prompting
prompt = """
请将以下文本翻译为中文：
Text: "The future of AI is promising."
"""

# Few-shot Prompting
prompt = """
根据以下示例，判断情感倾向：

文本: "我喜欢这个产品" -> 情感: 正面
文本: "这个服务很差" -> 情感: 负面
文本: "天气还不错" -> 情感: ?
"""

# Chain-of-Thought
prompt = """
请逐步思考，然后给出答案：

问题: 如果一家公司年收入 100 万，成本 80 万，税率 20%，净利润是多少？

思考步骤:
1. 计算利润: 100 - 80 = 20 万
2. 计算税收: 20 * 0.2 = 4 万
3. 计算净利润: 20 - 4 = 16 万

答案: 16 万
"""
```

**延伸阅读**:
- 📖 工具调用与函数调用实战.md - Prompt Engineering 进阶

### 2.2 Agent 概念与基础

#### 2.2.1 什么是 AI Agent

**核心定义**:
> AI Agent 是能够感知环境、做出决策并采取行动以实现目标的智能系统。

**Agent 的核心组件**:
```
┌─────────────────────────────────────┐
│           AI Agent                  │
├─────────────────────────────────────┤
│  感知 (Perception)                  │
│    ├─ 接收用户输入                   │
│    ├─ 理解上下文                     │
│    └─ 解析环境状态                   │
│                                     │
│  规划 (Planning)                    │
│    ├─ 任务分解                       │
│    ├─ 策略制定                       │
│    └─ 步骤排序                       │
│                                     │
│  记忆 (Memory)                      │
│    ├─ 短期记忆（上下文）              │
│    ├─ 长期记忆（向量数据库）          │
│    └─ 程序化记忆（技能）              │
│                                     │
│  工具使用 (Tool Use)                │
│    ├─ API 调用                       │
│    ├─ 代码执行                       │
│    └─ 外部系统集成                   │
│                                     │
│  行动 (Action)                      │
│    ├─ 生成响应                       │
│    ├─ 调用工具                       │
│    └─ 更新状态                       │
└─────────────────────────────────────┘
```

#### 2.2.2 Agent 架构模式

**1. ReAct 模式 (Reasoning + Acting)**
```
Thought → Action → Observation → Thought → ... → Final Answer
```

**2. Plan-and-Execute 模式**
```
Plan → [Execute Step 1, Execute Step 2, ...] → Review → Final Answer
```

**3. Reflection 模式**
```
Generate → Reflect → Refine → Output
```

**实践：简单的 ReAct Agent**
```python
import openai

class SimpleReActAgent:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.memory = []
    
    def run(self, task: str, tools: list, max_steps: int = 5):
        """运行 ReAct Agent"""
        messages = [
            {"role": "system", "content": "你是一个智能助手。请使用 ReAct 模式：Thought → Action → Observation"},
            {"role": "user", "content": task}
        ]
        
        for step in range(max_steps):
            # Thought + Action
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0
            )
            
            thought_action = response.choices[0].message.content
            messages.append({"role": "assistant", "content": thought_action})
            
            # 解析动作
            if "Action:" in thought_action:
                action = self._parse_action(thought_action)
                observation = self._execute_action(action, tools)
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                # Final answer
                return thought_action
        
        return "达到最大步骤限制"
    
    def _parse_action(self, text: str):
        """解析动作"""
        # 简单解析逻辑
        pass
    
    def _execute_action(self, action: str, tools: list):
        """执行动作"""
        # 工具执行逻辑
        pass
```

#### 2.2.3 工具调用基础

**学习内容**:
- Function Calling 原理
- OpenAI Function Calling
- Anthropic Tool Use
- 工具定义与注册
- 工具执行流程

**OpenAI Function Calling 示例**:
```python
from openai import OpenAI

client = OpenAI()

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如 '北京'、'上海'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 调用 LLM
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "北京今天天气怎么样？"}
    ],
    tools=tools,
    tool_choice="auto"
)

# 处理工具调用
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    # 执行工具
    if function_name == "get_weather":
        result = get_weather(arguments["city"], arguments.get("unit", "celsius"))
        
        # 将结果返回给 LLM
        messages = [
            {"role": "user", "content": "北京今天天气怎么样？"},
            response.choices[0].message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            }
        ]
        
        # 获取最终响应
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        print(final_response.choices[0].message.content)
```

**延伸阅读**:
- 📖 工具调用与函数调用实战.md - 完整的工具调用实战指南

### 2.3 LLM API 使用

#### 2.3.1 OpenAI API

**基础使用**:
```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# 聊天补全
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "什么是 AI Agent？"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

#### 2.3.2 Anthropic Claude API

```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

message = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "解释一下 ReAct 模式"}
    ]
)

print(message.content[0].text)
```

#### 2.3.3 多模型适配

```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list, **kwargs) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def chat(self, messages: list, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=kwargs.get("model", "gpt-4"),
            messages=messages
        )
        return response.choices[0].message.content

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def chat(self, messages: list, **kwargs) -> str:
        # 转换消息格式
        system_msg = ""
        user_msgs = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_msgs.append(msg)
        
        response = self.client.messages.create(
            model=kwargs.get("model", "claude-3-sonnet-20240229"),
            system=system_msg,
            messages=user_msgs,
            max_tokens=kwargs.get("max_tokens", 1000)
        )
        return response.content[0].text
```

### 2.4 阶段 1 实战项目

#### 项目 1.1：智能问答助手
- **目标**: 构建一个能回答用户问题的简单助手
- **技术**: OpenAI API、基础 Prompt Engineering
- **功能**:
  - 单轮对话
  - 角色设定
  - 温度控制

#### 项目 1.2：多轮对话系统
- **目标**: 实现具有记忆功能的多轮对话
- **技术**: 消息历史管理、上下文维护
- **功能**:
  - 对话历史维护
  - 上下文长度管理
  - 对话摘要

#### 项目 1.3：简单工具调用 Agent
- **目标**: 构建能调用外部工具的 Agent
- **技术**: Function Calling、工具定义
- **功能**:
  - 天气查询
  - 时间查询
  - 简单计算

### 2.5 阶段 1 自检清单

- [ ] 理解 Transformer 和 LLM 基础原理
- [ ] 掌握 Prompt Engineering 基础技巧
- [ ] 能够使用 OpenAI/Claude API
- [ ] 理解 Agent 的核心组件和工作流程
- [ ] 能够实现简单的 ReAct Agent
- [ ] 掌握 Function Calling 基础
- [ ] 完成 3 个实战项目

---

## 4. 阶段 2：单 Agent 开发

> **目标**: 掌握使用框架开发单 Agent 的能力，能够实现记忆、RAG、复杂工具调用  
> **时间**: 6-8 周  
> **前置知识**: 阶段 1 完成

### 3.1 LangChain 框架

#### 3.1.1 LangChain 核心概念

**学习内容**:
- LangChain 架构
  - Models（模型层）
  - Prompts（提示层）
  - Chains（链）
  - Agents（智能体）
  - Memory（记忆）
  - Indexes（索引）

- 核心组件
  - LLM Wrappers
  - Prompt Templates
  - Output Parsers
  - Chains
  - Tools

**LangChain 基础示例**:
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 初始化 LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 创建 Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{profession}助手"),
    ("user", "请解释{concept}")
])

# 创建输出解析器
parser = StrOutputParser()

# 构建链
chain = prompt | llm | parser

# 执行
result = chain.invoke({
    "profession": "AI",
    "concept": "Transformer 架构"
})
print(result)
```

#### 3.1.2 LangChain Tools 与 Agent

**工具定义**:
```python
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

@tool
def search_knowledge(query: str) -> str:
    """搜索知识库"""
    # 实现搜索逻辑
    return f"搜索结果: {query}"

# 注册工具
tools = [calculate, search_knowledge]

# 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 运行 Agent
result = agent.run("计算 2^10 并搜索相关知识")
```

**延伸阅读**:
- 📖 LangChain 与 LangGraph 最新实践.md

#### 3.1.3 LangChain Expression Language (LCEL)

**LCEL 基础**:
```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# LCEL 链式调用
chain = (
    RunnablePassthrough.assign(
        context=lambda x: retrieve_context(x["question"])
    )
    | prompt
    | llm
    | parser
)

# 并行执行
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    summary=summary_chain,
    analysis=analysis_chain,
    recommendations=recommendation_chain
)

result = parallel.invoke({"input": "某篇文章"})
```

### 3.2 LangGraph 工作流

#### 3.2.1 LangGraph 基础

**学习内容**:
- StateGraph 概念
- Node（节点）
- Edge（边）
- Conditional Edge（条件边）
- State（状态）

**基础工作流示例**:
```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

# 定义状态
class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    steps: Annotated[list, operator.add]

# 定义节点
def retrieve_node(state: AgentState):
    """检索节点"""
    context = retrieve_documents(state["question"])
    return {"context": context, "steps": ["retrieve"]}

def generate_node(state: AgentState):
    """生成节点"""
    answer = generate_answer(state["question"], state["context"])
    return {"answer": answer, "steps": ["generate"]}

def review_node(state: AgentState):
    """审核节点"""
    needs_revision = check_quality(state["answer"])
    return {"steps": ["review"], "needs_revision": needs_revision}

# 构建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("review", review_node)

# 添加边
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "review")

# 条件边
def decide_route(state: AgentState):
    if state.get("needs_revision", False):
        return "generate"
    return END

workflow.add_conditional_edges(
    "review",
    decide_route,
    {"generate": "generate", END: END}
)

# 编译
app = workflow.compile()

# 执行
result = app.invoke({
    "question": "什么是 LangGraph?",
    "context": "",
    "answer": "",
    "steps": []
})
```

#### 3.2.2 复杂工作流模式

**1. 并行处理**:
```python
from langgraph.graph import StateGraph

class ParallelState(TypedDict):
    query: str
    results: dict

def search_web(state: ParallelState):
    return {"results": {"web": web_search(state["query"])}}

def search_database(state: ParallelState):
    return {"results": {"db": database_search(state["query"])}}

workflow = StateGraph(ParallelState)
workflow.add_node("web_search", search_web)
workflow.add_node("db_search", search_database)

# 并行执行
workflow.add_edge("__start__", "web_search")
workflow.add_edge("__start__", "db_search")
workflow.add_edge("web_search", END)
workflow.add_edge("db_search", END)

app = workflow.compile()
```

**2. 循环与迭代**:
```python
class IterativeState(TypedDict):
    task: str
    solution: str
    iteration: int
    max_iterations: int

def generate_solution(state: IterativeState):
    solution = generate(state["task"])
    return {
        "solution": solution,
        "iteration": state["iteration"] + 1
    }

def evaluate_solution(state: IterativeState):
    score = evaluate(state["solution"])
    return {"score": score}

def should_continue(state: IterativeState):
    if state["iteration"] >= state["max_iterations"]:
        return END
    if state["score"] >= 0.9:
        return END
    return "generate_solution"

workflow = StateGraph(IterativeState)
workflow.add_node("generate", generate_solution)
workflow.add_node("evaluate", evaluate_solution)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "evaluate")
workflow.add_conditional_edges("evaluate", should_continue)
```

**延伸阅读**:
- 📖 LangChain 与 LangGraph 最新实践.md - 深入 LangGraph 实践

### 3.3 记忆系统

#### 3.3.1 记忆类型

**学习内容**:
- 短期记忆（Short-term Memory）
  - 对话历史
  - 上下文窗口管理
  
- 长期记忆（Long-term Memory）
  - 向量数据库
  - 知识图谱
  - 摘要存储
  
- 工作记忆（Working Memory）
  - 当前任务状态
  - 中间结果
  
- 程序化记忆（Procedural Memory）
  - 技能和工具
  - 最佳实践

#### 3.3.2 向量数据库与记忆

**使用 ChromaDB 实现记忆**:
```python
import chromadb
from chromadb.utils import embedding_functions

# 初始化 ChromaDB
chroma_client = chromadb.Client()

# 创建集合
collection = chroma_client.create_collection(
    name="agent_memory",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key="your-api-key",
        model_name="text-embedding-ada-002"
    )
)

# 添加记忆
def add_memory(text: str, metadata: dict):
    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[f"mem_{collection.count()}"]
    )

# 检索记忆
def retrieve_memory(query: str, n_results: int = 5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results["documents"][0]

# 使用示例
add_memory(
    "用户喜欢 Python 编程",
    {"type": "preference", "user_id": "user_123", "timestamp": "2024-01-01"}
)

memories = retrieve_memory("用户兴趣")
print(memories)
```

#### 3.3.3 LangChain 记忆实现

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    VectorStoreRetrieverMemory,
    ConversationBufferWindowMemory
)

# 1. 对话缓冲记忆
buffer_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 2. 对话摘要记忆
summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

# 3. 滑动窗口记忆
window_memory = ConversationBufferWindowMemory(
    k=5,  # 保留最近 5 轮对话
    memory_key="chat_history",
    return_messages=True
)

# 4. 向量检索记忆
from langchain.vectorstores import Chroma

vectorstore = Chroma(embedding_function=embeddings)
retriever_memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
)

# 在 Agent 中使用记忆
agent_chain = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=buffer_memory,
    verbose=True
)
```

### 3.4 RAG 基础

#### 3.4.1 RAG 架构

**学习内容**:
- Document Loading（文档加载）
- Text Splitting（文本分割）
- Embedding（向量化）
- Vector Storage（向量存储）
- Retrieval（检索）
- Generation（生成）

**基础 RAG 实现**:
```python
from langchain.document_loaders import TextLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# 1. 加载文档
loader = TextLoader("document.txt")
documents = loader.load()

# 2. 分割文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# 3. 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    collection_name="rag_docs"
)

# 4. 创建检索 QA 链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    ),
    return_source_documents=True
)

# 5. 查询
result = qa_chain({"query": "文档中关于 AI Agent 的内容？"})
print(result["result"])
print(f"来源文档: {result['source_documents']}")
```

**延伸阅读**:
- 📖 RAG 技术进阶实战.md - RAG 高级技术

#### 3.4.2 高级 RAG 技术

**1. 混合检索**:
```python
from langchain.retrievers import (
    BM25Retriever,
    EnsembleRetriever
)

# BM25 检索（关键词匹配）
bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 3

# 向量检索（语义匹配）
vector_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 集成检索
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)
```

**2. 上下文压缩**:
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    LLMChainFilter
)

# LLM 压缩
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_retriever
)

# 使用压缩检索
compressed_docs = compression_retriever.get_relevant_documents(
    "用户查询"
)
```

### 3.5 阶段 2 实战项目

#### 项目 2.1：智能文档助手
- **目标**: 构建基于 RAG 的文档问答系统
- **技术**: LangChain、ChromaDB、RAG
- **功能**:
  - 文档上传和解析
  - 向量检索
  - 智能问答
  - 来源追溯

#### 项目 2.2：个人知识助手
- **目标**: 构建具有长期记忆的个人助手
- **技术**: LangGraph、向量记忆、摘要记忆
- **功能**:
  - 用户偏好学习
  - 对话历史管理
  - 个性化响应
  - 知识积累

#### 项目 2.3：代码助手 Agent
- **目标**: 构建能辅助编程的 Agent
- **技术**: 工具调用、代码执行、错误处理
- **功能**:
  - 代码生成
  - 代码审查
  - Bug 修复建议
  - 代码解释

### 3.6 阶段 2 自检清单

- [ ] 掌握 LangChain 核心概念和组件
- [ ] 能够使用 LangGraph 构建复杂工作流
- [ ] 理解并实现多种记忆系统
- [ ] 掌握 RAG 架构和实现
- [ ] 能够实现高级 RAG 技术（混合检索、上下文压缩）
- [ ] 完成 3 个实战项目
- [ ] 能够调试和优化 Agent 性能

---

## 5. 阶段 3：高级 Agent 技术

> **目标**: 掌握高级 Agent 开发技术，包括 Harness Engineering、Nanobot、Skills 开发和 MCP 协议  
> **时间**: 8-10 周  
> **前置知识**: 阶段 2 完成

### 4.1 Harness Engineering

#### 4.1.1 什么是 Harness Engineering

**核心概念**:
> Harness Engineering 是指设计和构建 Agent 的"控制 harness"，即管理 Agent 行为、工具访问、安全限制的框架层。

**学习内容**:
- Agent 运行时环境
- 工具权限管理
- 安全沙箱
- 资源限制
- 监控和审计

#### 4.1.2 Harness 架构设计

```python
from dataclasses import dataclass
from typing import Dict, List, Callable
import time

@dataclass
class ToolPermission:
    name: str
    allowed: bool
    rate_limit: int  # 每分钟调用次数
    timeout: int     # 超时时间（秒）

class AgentHarness:
    """Agent 控制Harness"""
    
    def __init__(self):
        self.permissions: Dict[str, ToolPermission] = {}
        self.usage_log: List[Dict] = []
        self.is_running = False
    
    def register_tool(self, name: str, permission: ToolPermission):
        """注册工具权限"""
        self.permissions[name] = permission
    
    def check_permission(self, tool_name: str) -> bool:
        """检查工具权限"""
        if tool_name not in self.permissions:
            return False
        
        perm = self.permissions[tool_name]
        if not perm.allowed:
            return False
        
        # 检查速率限制
        recent_calls = [
            log for log in self.usage_log
            if log["tool"] == tool_name 
            and time.time() - log["timestamp"] < 60
        ]
        
        if len(recent_calls) >= perm.rate_limit:
            return False
        
        return True
    
    async def execute_tool(self, tool_name: str, func: Callable, *args, **kwargs):
        """安全执行工具"""
        if not self.check_permission(tool_name):
            raise PermissionError(f"Tool {tool_name} not allowed or rate limited")
        
        perm = self.permissions[tool_name]
        
        try:
            # 设置超时
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=perm.timeout
            )
            
            # 记录日志
            self.usage_log.append({
                "tool": tool_name,
                "timestamp": time.time(),
                "status": "success"
            })
            
            return result
            
        except asyncio.TimeoutError:
            self.usage_log.append({
                "tool": tool_name,
                "timestamp": time.time(),
                "status": "timeout"
            })
            raise
        except Exception as e:
            self.usage_log.append({
                "tool": tool_name,
                "timestamp": time.time(),
                "status": "error",
                "error": str(e)
            })
            raise
    
    def get_usage_stats(self) -> Dict:
        """获取使用统计"""
        stats = {}
        for log in self.usage_log:
            tool = log["tool"]
            if tool not in stats:
                stats[tool] = {"success": 0, "error": 0, "timeout": 0}
            stats[tool][log["status"]] += 1
        return stats
```

**延伸阅读**:
- 📖 Harness Engineering 实战.md - 完整的 Harness Engineering 实践

#### 4.1.3 安全沙箱实现

**代码执行沙箱**:
```python
import ast
import builtins
from typing import Any, Dict

class CodeSandbox:
    """安全的代码执行沙箱"""
    
    def __init__(self):
        # 允许的内建函数
        self.safe_builtins = {
            "print": print,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
        }
        
        # 禁止的模块
        self.banned_modules = {
            "os", "sys", "subprocess", "shutil",
            "socket", "requests", "urllib"
        }
    
    def is_safe_code(self, code: str) -> bool:
        """检查代码安全性"""
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # 检查 import
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = node.names[0].name
                    if module in self.banned_modules:
                        return False
                
                # 检查危险函数调用
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in ["__import__", "eval", "exec"]:
                            return False
            
            return True
            
        except SyntaxError:
            return False
    
    def execute(self, code: str, namespace: Dict[str, Any] = None) -> Any:
        """执行安全代码"""
        if not self.is_safe_code(code):
            raise SecurityError("Unsafe code detected")
        
        safe_globals = {"__builtins__": self.safe_builtins}
        if namespace:
            safe_globals.update(namespace)
        
        exec(code, safe_globals)
        return safe_globals.get("result")
```

### 4.2 Nanobot 开发

#### 4.2.1 Nanobot 概念

**核心定义**:
> Nanobot 是小型、专注、高效的 Agent，每个 Nanobot 专门处理特定任务，多个 Nanobot 可以协作完成复杂任务。

**Nanobot 特点**:
- 单一职责
- 轻量级
- 可组合
- 易测试
- 高性能

#### 4.2.2 Nanobot 架构

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class Nanobot(ABC):
    """Nanobot 基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.state: Dict[str, Any] = {}
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行 Nanobot 任务"""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """获取状态"""
        return self.state.copy()
    
    def set_state(self, key: str, value: Any):
        """设置状态"""
        self.state[key] = value
class SearchNanobot(Nanobot):
    """搜索 Nanobot"""
    
    def __init__(self):
        super().__init__(
            name="search_bot",
            description="搜索互联网获取信息"
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data["query"]
        # 执行搜索
        results = await self._search(query)
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    
    async def _search(self, query: str) -> list:
        # 搜索实现
        return []
class AnalyzeNanobot(Nanobot):
    """分析 Nanobot"""
    
    def __init__(self):
        super().__init__(
            name="analyze_bot",
            description="分析文本内容"
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        text = input_data["text"]
        analysis_type = input_data.get("type", "summary")
        
        if analysis_type == "summary":
            result = await self._summarize(text)
        elif analysis_type == "sentiment":
            result = await self._analyze_sentiment(text)
        else:
            result = {"error": "Unknown analysis type"}
        
        return {
            "text": text,
            "type": analysis_type,
            "result": result
        }
    
    async def _summarize(self, text: str) -> str:
        # 摘要实现
        return "摘要内容"
    
    async def _analyze_sentiment(self, text: str) -> str:
        # 情感分析实现
        return "positive"
```

#### 4.2.3 Nanobot 编排

```python
class NanobotOrchestrator:
    """Nanobot 编排器"""
    
    def __init__(self):
        self.bots: Dict[str, Nanobot] = {}
        self.workflows: Dict[str, List[str]] = {}
    
    def register_bot(self, bot: Nanobot):
        """注册 Nanobot"""
        self.bots[bot.name] = bot
    
    def define_workflow(self, name: str, bot_sequence: List[str]):
        """定义工作流"""
        self.workflows[name] = bot_sequence
    
    async def execute_workflow(self, workflow_name: str, 
                              initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """执行工作流"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} not found")
        
        sequence = self.workflows[workflow_name]
        current_data = initial_input
        
        execution_log = []
        
        for bot_name in sequence:
            if bot_name not in self.bots:
                raise ValueError(f"Nanobot {bot_name} not found")
            
            bot = self.bots[bot_name]
            
            # 执行 Nanobot
            result = await bot.execute(current_data)
            
            # 记录日志
            execution_log.append({
                "bot": bot_name,
                "input": current_data,
                "output": result
            })
            
            # 更新数据
            current_data.update(result)
        
        return {
            "final_output": current_data,
            "execution_log": execution_log
        }

# 使用示例
async def main():
    orchestrator = NanobotOrchestrator()
    
    # 注册 Nanobots
    orchestrator.register_bot(SearchNanobot())
    orchestrator.register_bot(AnalyzeNanobot())
    
    # 定义工作流
    orchestrator.define_workflow(
        "research_workflow",
        ["search_bot", "analyze_bot"]
    )
    
    # 执行工作流
    result = await orchestrator.execute_workflow(
        "research_workflow",
        {"query": "AI Agent 最新发展", "type": "summary"}
    )
    
    print(result)
```

**延伸阅读**:
- 📖 Harness Engineering 实战.md - Nanobot 实践

### 4.3 Skills 开发

#### 4.3.1 Skills 概念

**核心定义**:
> Skills 是 Agent 可以学习和使用的特定能力或知识模块，可以动态加载、组合和扩展。

**Skills 特点**:
- 模块化
- 可复用
- 可组合
- 版本管理
- 热加载

#### 4.3.2 Skill 架构

```python
from typing import Dict, List, Any, Optional
import importlib
import json

class Skill:
    """Skill 基类"""
    
    def __init__(self, name: str, version: str, description: str):
        self.name = name
        self.version = version
        self.description = description
        self.metadata: Dict[str, Any] = {}
        self.is_loaded = False
    
    async def execute(self, **kwargs) -> Any:
        """执行 Skill"""
        raise NotImplementedError
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取元数据"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "metadata": self.metadata
        }
class CodeGenerationSkill(Skill):
    """代码生成 Skill"""
    
    def __init__(self):
        super().__init__(
            name="code_generation",
            version="1.0.0",
            description="生成代码的 skill"
        )
        self.metadata = {
            "languages": ["python", "javascript", "typescript"],
            "capabilities": ["generation", "explanation", "debugging"]
        }
    
    async def execute(self, 
                     prompt: str,
                     language: str = "python",
                     **kwargs) -> Dict[str, Any]:
        """生成代码"""
        # 调用 LLM 生成代码
        code = await self._generate_code(prompt, language)
        
        return {
            "code": code,
            "language": language,
            "explanation": await self._explain_code(code, language)
        }
    
    async def _generate_code(self, prompt: str, language: str) -> str:
        # 代码生成实现
        return "# Generated code"
    
    async def _explain_code(self, code: str, language: str) -> str:
        # 代码解释实现
        return "代码解释"
class DataAnalysisSkill(Skill):
    """数据分析 Skill"""
    
    def __init__(self):
        super().__init__(
            name="data_analysis",
            version="1.0.0",
            description="数据分析 skill"
        )
        self.metadata = {
            "operations": ["statistics", "visualization", "modeling"],
            "formats": ["csv", "json", "excel"]
        }
    
    async def execute(self,
                     data: Any,
                     operation: str,
                     **kwargs) -> Dict[str, Any]:
        """执行数据分析"""
        if operation == "statistics":
            result = await self._calculate_statistics(data)
        elif operation == "visualization":
            result = await self._create_visualization(data)
        else:
            result = {"error": "Unknown operation"}
        
        return {
            "operation": operation,
            "result": result
        }
    
    async def _calculate_statistics(self, data: Any) -> Dict:
        # 统计分析实现
        return {"mean": 0, "median": 0, "std": 0}
    
    async def _create_visualization(self, data: Any) -> str:
        # 可视化实现
        return "chart_url"
```

#### 4.3.3 Skill 管理器

```python
class SkillManager:
    """Skill 管理器"""
    
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.skill_registry: List[Dict[str, str]] = []
    
    def register_skill(self, skill: Skill):
        """注册 Skill"""
        self.skills[skill.name] = skill
        self.skill_registry.append({
            "name": skill.name,
            "version": skill.version,
            "description": skill.description
        })
    
    def load_skill(self, skill_name: str, version: Optional[str] = None) -> Skill:
        """加载 Skill"""
        if skill_name not in self.skills:
            raise ValueError(f"Skill {skill_name} not found")
        
        skill = self.skills[skill_name]
        
        if version and skill.version != version:
            raise ValueError(
                f"Skill {skill_name} version {version} not found"
            )
        
        skill.is_loaded = True
        return skill
    
    def list_available_skills(self) -> List[Dict[str, str]]:
        """列出可用 Skills"""
        return self.skill_registry
    
    async def execute_skill(self, 
                           skill_name: str, 
                           **kwargs) -> Any:
        """执行 Skill"""
        skill = self.load_skill(skill_name)
        return await skill.execute(**kwargs)
    
    def compose_skills(self, skill_names: List[str]) -> 'ComposedSkill':
        """组合多个 Skills"""
        skills = [self.load_skill(name) for name in skill_names]
        return ComposedSkill(skills)
class ComposedSkill(Skill):
    """组合 Skill"""
    
    def __init__(self, skills: List[Skill]):
        super().__init__(
            name="composed_skill",
            version="1.0.0",
            description=f"Composed of {len(skills)} skills"
        )
        self.skills = skills
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行组合 Skill"""
        results = {}
        
        for skill in self.skills:
            result = await skill.execute(**kwargs)
            results[skill.name] = result
        
        return results
```

**延伸阅读**:
- 📖 MCP 协议与 Skills 开发实战.md - Skills 开发完整指南

### 4.4 MCP 协议

#### 4.4.1 MCP 基础

**学习内容**:
- MCP (Model Context Protocol) 概念
- MCP 架构
- MCP Server 开发
- MCP Client 开发
- MCP 资源管理

#### 4.4.2 MCP Server 开发

```python
from mcp.server import Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent
)

# 创建 MCP Server
server = Server("my-agent-server")

@server.list_resources()
async def list_resources() -> list[Resource]:
    """列出可用资源"""
    return [
        Resource(
            uri="knowledge://documents",
            name="Documents",
            description="访问文档知识库",
            mimeType="text/plain"
        ),
        Resource(
            uri="memory://conversation",
            name="Conversation Memory",
            description="访问对话记忆",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    """读取资源"""
    if uri.startswith("knowledge://"):
        # 读取知识库
        return "知识库内容"
    elif uri.startswith("memory://"):
        # 读取记忆
        return "{}"
    else:
        raise ValueError(f"Unknown resource: {uri}")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用工具"""
    return [
        Tool(
            name="search_documents",
            description="搜索文档",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="analyze_data",
            description="分析数据",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "JSON 格式数据"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["statistics", "visualization"]
                    }
                },
                "required": ["data", "operation"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """调用工具"""
    if name == "search_documents":
        results = await search_documents(arguments["query"])
        return [TextContent(type="text", text=json.dumps(results))]
    elif name == "analyze_data":
        result = await analyze_data(
            arguments["data"],
            arguments["operation"]
        )
        return [TextContent(type="text", text=json.dumps(result))]
    else:
        raise ValueError(f"Unknown tool: {name}")

# 运行 Server
if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server
    
    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    
    asyncio.run(main())
```

#### 4.4.3 MCP Client 集成

```python
from mcp.client import Client
import asyncio

class MCPAgentClient:
    """MCP Agent 客户端"""
    
    def __init__(self, server_url: str):
        self.client = Client(server_url)
    
    async def connect(self):
        """连接到 MCP Server"""
        await self.client.connect()
    
    async def list_resources(self) -> list:
        """列出资源"""
        return await self.client.list_resources()
    
    async def read_resource(self, uri: str) -> str:
        """读取资源"""
        return await self.client.read_resource(uri)
    
    async def list_tools(self) -> list:
        """列出工具"""
        return await self.client.list_tools()
    
    async def call_tool(self, name: str, **kwargs) -> Any:
        """调用工具"""
        return await self.client.call_tool(name, kwargs)
    
    async def close(self):
        """关闭连接"""
        await self.client.close()

# 使用示例
async def main():
    client = MCPAgentClient("http://localhost:8080")
    
    try:
        await client.connect()
        
        # 列出工具
        tools = await client.list_tools()
        print(f"可用工具: {tools}")
        
        # 调用工具
        result = await client.call_tool(
            "search_documents",
            query="AI Agent"
        )
        print(f"搜索结果: {result}")
        
    finally:
        await client.close()

asyncio.run(main())
```

**延伸阅读**:
- 📖 MCP 协议与 Skills 开发实战.md - MCP 协议完整指南

### 4.5 阶段 3 实战项目

#### 项目 3.1：安全代码执行 Agent
- **目标**: 构建能安全执行代码的 Agent
- **技术**: Harness Engineering、代码沙箱
- **功能**:
  - 代码安全检测
  - 沙箱执行
  - 资源限制
  - 结果返回

#### 项目 3.2：Nanobot 研究助手
- **目标**: 构建基于 Nanobot 的研究助手
- **技术**: Nanobot、工作流编排
- **功能**:
  - 文献搜索
  - 内容分析
  - 摘要生成
  - 报告撰写

#### 项目 3.3：Skills 市场系统
- **目标**: 构建 Skills 管理和交易系统
- **技术**: Skills 开发、动态加载
- **功能**:
  - Skill 注册
  - Skill 发现
  - Skill 组合
  - 版本管理

#### 项目 3.4：MCP 知识服务
- **目标**: 构建基于 MCP 的知识服务
- **技术**: MCP Server/Client、资源管理
- **功能**:
  - 知识检索
  - 工具调用
  - 资源访问
  - 多客户端支持

### 4.6 阶段 3 自检清单

- [ ] 理解 Harness Engineering 概念和实现
- [ ] 能够实现安全沙箱和资源控制
- [ ] 掌握 Nanobot 开发和编排
- [ ] 理解 Skills 架构和开发
- [ ] 能够实现 Skills 管理器
- [ ] 掌握 MCP 协议基础
- [ ] 能够开发 MCP Server 和 Client
- [ ] 完成 4 个实战项目

---

## 6. 阶段 4：多 Agent 协作

> **目标**: 掌握多 Agent 系统设计和开发，能够实现复杂的 Agent 协作  
> **时间**: 6-8 周  
> **前置知识**: 阶段 3 完成

### 5.1 多 Agent 基础

#### 5.1.1 为什么需要多 Agent

**单 Agent 的局限性**:
- 能力单一，难以处理复杂任务
- 容易陷入死循环
- 缺乏专业分工
- 难以扩展

**多 Agent 的优势**:
- 专业分工，各司其职
- 并行处理，提高效率
- 相互监督，减少错误
- 易于扩展和维护

#### 5.1.2 多 Agent 架构模式

**1. 顺序协作模式**:
```
Agent A → Agent B → Agent C → Output
```

**2. 层次化管理模式**:
```
         Manager Agent
        /      |      \
   Agent A  Agent B  Agent C
```

**3. 投票决策模式**:
```
   Agent A ↘
   Agent B → Voter → Decision
   Agent C ↗
```

**4. 辩论模式**:
```
Agent A (Pro) ←→ Debate ←→ Agent B (Con)
                           ↓
                      Judge Agent
```

### 5.2 CrewAI 框架

#### 5.2.1 CrewAI 核心概念

**学习内容**:
- Crew（团队）
- Agent（角色）
- Task（任务）
- Process（流程）
- Tools（工具）

#### 5.2.2 CrewAI 基础使用

```python
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# 初始化 LLM
llm = ChatOpenAI(model="gpt-4")

# 创建 Agent
researcher = Agent(
    role='高级研究分析师',
    goal='深入研究 {topic} 领域的最新进展',
    backstory="""你是 {topic} 领域的专家研究分析师，
擅长查找、分析和总结最新研究成果。""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

writer = Agent(
    role='技术作家',
    goal='基于研究成果撰写清晰的技术文章',
    backstory="""你是经验丰富的技术作家，
擅长将复杂的技术概念解释得通俗易懂。""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

editor = Agent(
    role='编辑',
    goal='审核和改进文章质量',
    backstory="""你是资深编辑，
擅长提高文章质量和可读性。""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# 创建 Task
research_task = Task(
    description="""研究 {topic} 领域的最新进展，包括：
1. 最近 3 个月的重要突破
2. 关键技术发展趋势
3. 主要研究团队和成果""",
    agent=researcher,
    expected_output="一份详细的研究分析报告"
)

writing_task = Task(
    description="""基于研究报告撰写技术文章：
1. 引人入胜的标题和开头
2. 清晰的技术解释
3. 实际应用案例""",
    agent=writer,
    expected_output="一篇 2000 字的技术文章",
    context=[research_task]  # 依赖研究任务
)

editing_task = Task(
    description="""审核文章质量：
1. 检查准确性和完整性
2. 改进语言和结构
3. 添加必要的补充说明""",
    agent=editor,
    expected_output="最终版本的文章",
    context=[writing_task]  # 依赖写作任务
)

# 创建 Crew
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,  # 顺序执行
    verbose=True
)

# 执行
result = crew.kickoff(inputs={"topic": "AI Agent"})
print(result)
```

#### 5.2.3 多 Agent 协作模式

**层次化管理**:
```python
from crewai import Agent, Task, Crew, Process

# 管理者 Agent
manager = Agent(
    role='项目经理',
    goal='协调团队完成项目开发',
    backstory="""你是经验丰富的项目经理，
擅长任务分配和团队协调。""",
    allow_delegation=True,  # 允许委派任务
    llm=llm
)

# 开发团队
developer1 = Agent(
    role='前端开发',
    goal='开发用户界面',
    backstory="资深前端开发工程师",
    llm=llm
)

developer2 = Agent(
    role='后端开发',
    goal='开发 API 和业务逻辑',
    backstory="资深后端开发工程师",
    llm=llm
)

tester = Agent(
    role='测试工程师',
    goal='确保代码质量',
    backstory="经验丰富的测试工程师",
    llm=llm
)

# 创建 Crew
dev_crew = Crew(
    agents=[manager, developer1, developer2, tester],
    tasks=[
        Task(description="制定开发计划", agent=manager),
        Task(description="开发前端界面", agent=developer1),
        Task(description="开发后端 API", agent=developer2),
        Task(description="集成测试", agent=tester)
    ],
    process=Process.hierarchical,  # 层次化流程
    manager_llm=llm,  # 管理者的 LLM
    verbose=True
)

result = dev_crew.kickoff(inputs={"project": "Agent 管理系统"})
```

**延伸阅读**:
- 📖 多 Agent 协作系统开发.md - CrewAI 完整实践

### 5.3 AutoGen 框架

#### 5.3.1 AutoGen 核心概念

**学习内容**:
- ConversableAgent
- GroupChat
- AssistantAgent
- UserProxyAgent
- 对话管理

#### 5.3.2 AutoGen 基础使用

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# 配置
config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-api-key"
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7
}

# 创建 Agent
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="""你是一个专业的 AI 研究助手。
擅长分析和总结 AI 领域的最新进展。"""
)

critic = AssistantAgent(
    name="critic",
    llm_config=llm_config,
    system_message="""你是一个严格的评论者。
负责审查和改进研究结果的质量。"""
)

user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",  # 不要求人类输入
    max_consecutive_auto_reply=10,
    code_execution_config=False  # 不执行代码
)

# 创建群聊
groupchat = GroupChat(
    agents=[user_proxy, assistant, critic],
    messages=[],
    max_round=10
)

# 创建群聊管理器
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# 开始对话
user_proxy.initiate_chat(
    manager,
    message="""请研究并分析 AI Agent 技术的最新发展趋势，
包括技术突破、应用场景和未来方向。"""
)
```

#### 5.3.3 复杂多 Agent 场景

**代码开发与审查**:
```python
from autogen import AssistantAgent, UserProxyAgent

# 配置
llm_config = {"config_list": config_list}

# 创建 Agent
coder = AssistantAgent(
    name="coder",
    llm_config=llm_config,
    system_message="你是专业的 Python 开发工程师"
)

reviewer = AssistantAgent(
    name="reviewer",
    llm_config=llm_config,
    system_message="你是严格的代码审查员"
)

tester = AssistantAgent(
    name="tester",
    llm_config=llm_config,
    system_message="你是专业的测试工程师"
)

user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False
    }
)

# 多轮协作开发
def code_development_workflow(task: str):
    # 用户提出需求
    user_proxy.initiate_chat(
        coder,
        message=f"开发需求：{task}"
    )
    
    # 代码审查
    user_proxy.send(
        "请审查以上代码，提出改进建议",
        reviewer
    )
    
    # 改进代码
    user_proxy.send(
        "根据审查建议改进代码",
        coder
    )
    
    # 测试
    user_proxy.send(
        "编写测试用例并执行测试",
        tester
    )

# 使用
code_development_workflow("实现一个快速排序算法")
```

**延伸阅读**:
- 📖 多 Agent 协作系统开发.md - AutoGen 完整实践

### 5.4 OpenAI Agents SDK

#### 5.4.1 OpenAI Agents 基础

**学习内容**:
- Agent 定义
- Handoffs（交接）
- Guards（守卫）
- Tools（工具）
- Tracing（追踪）

#### 5.4.2 Agents SDK 使用

```python
from agents import Agent, Runner, function_tool, handoff
import asyncio

# 定义工具
@function_tool
def get_weather(city: str) -> str:
    """获取天气信息"""
    # 实现天气查询
    return f"{city} 今天晴天，25°C"

@function_tool
def search_news(topic: str) -> str:
    """搜索新闻"""
    # 实现新闻搜索
    return f"关于 {topic} 的最新新闻..."

# 创建 Agent
weather_agent = Agent(
    name="Weather Agent",
    instructions="你是一个专业的天气助手",
    tools=[get_weather]
)

news_agent = Agent(
    name="News Agent",
    instructions="你是一个新闻助手",
    tools=[search_news]
)

# 创建路由 Agent
router_agent = Agent(
    name="Router",
    instructions="根据用户需求路由到合适的专业 Agent",
    handoffs=[
        handoff(weather_agent),
        handoff(news_agent)
    ]
)

# 运行 Agent
async def main():
    result = await Runner.run(
        router_agent,
        "北京今天天气怎么样？"
    )
    print(result.final_output)

asyncio.run(main())
```

#### 5.4.3 Guards 安全机制

```python
from agents import Agent, Guard, function_tool
from typing import Any

# 定义输入守卫
class InputGuard(Guard):
    """输入内容安全检查"""
    
    async def validate(self, input: Any) -> Any:
        # 检查敏感词
        sensitive_words = ["password", "secret", "token"]
        for word in sensitive_words:
            if word in str(input).lower():
                raise ValueError(
                    f"输入包含敏感词: {word}"
                )
        return input

# 定义输出守卫
class OutputGuard(Guard):
    """输出内容安全检查"""
    
    async def validate(self, output: Any) -> Any:
        # 检查输出格式
        if not isinstance(output, str):
            raise ValueError("输出必须是字符串")
        
        # 检查长度限制
        if len(output) > 1000:
            raise ValueError("输出过长")
        
        return output

# 使用 Guards
safe_agent = Agent(
    name="Safe Agent",
    instructions="安全的助手",
    input_guards=[InputGuard()],
    output_guards=[OutputGuard()]
)
```

**延伸阅读**:
- 📖 多 Agent 协作系统开发.md - OpenAI Agents 实践

### 5.5 Agent 编排与调度

#### 5.5.1 Agent 编排模式

**1. 流水线编排**:
```python
class PipelineOrchestrator:
    """流水线编排器"""
    
    def __init__(self):
        self.stages: List[Agent] = []
    
    def add_stage(self, agent: Agent):
        self.stages.append(agent)
    
    async def execute(self, input: Any) -> Any:
        current = input
        results = []
        
        for stage in self.stages:
            result = await stage.run(current)
            results.append({
                "stage": stage.name,
                "input": current,
                "output": result
            })
            current = result
        
        return {
            "final_output": current,
            "execution_trace": results
        }
```

**2. DAG 编排**:
```python
class DAGOrchestrator:
    """DAG 编排器"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.dependencies: Dict[str, List[str]] = {}
    
    def add_agent(self, name: str, agent: Agent, 
                 dependencies: List[str] = None):
        self.agents[name] = agent
        self.dependencies[name] = dependencies or []
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        completed = set()
        
        while len(completed) < len(self.agents):
            # 找到可以执行的节点
            ready = [
                name for name in self.agents
                if name not in completed
                and all(
                    dep in completed
                    for dep in self.dependencies[name]
                )
            ]
            
            # 并行执行
            for name in ready:
                agent_input = {
                    dep: results[dep]
                    for dep in self.dependencies[name]
                    if dep in results
                }
                agent_input.update(inputs.get(name, {}))
                
                results[name] = await self.agents[name].run(agent_input)
                completed.add(name)
        
        return results
```

#### 5.5.2 Agent 通信机制

**消息队列**:
```python
import asyncio
from typing import Dict, List, Callable
from dataclasses import dataclass

@dataclass
class AgentMessage:
    sender: str
    receiver: str
    content: Any
    metadata: Dict = None

class MessageBroker:
    """Agent 消息代理"""
    
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}
        self.handlers: Dict[str, List[Callable]] = {}
    
    def register_agent(self, agent_name: str):
        """注册 Agent"""
        self.queues[agent_name] = asyncio.Queue()
    
    async def send_message(self, message: AgentMessage):
        """发送消息"""
        if message.receiver in self.queues:
            await self.queues[message.receiver].put(message)
    
    async def receive_message(self, agent_name: str) -> AgentMessage:
        """接收消息"""
        if agent_name in self.queues:
            return await self.queues[agent_name].get()
    
    def subscribe(self, agent_name: str, handler: Callable):
        """订阅消息"""
        if agent_name not in self.handlers:
            self.handlers[agent_name] = []
        self.handlers[agent_name].append(handler)
```

### 5.6 阶段 4 实战项目

#### 项目 4.1：自动化研究团队
- **目标**: 构建自动化的研究团队
- **技术**: CrewAI、多 Agent 协作
- **功能**:
  - 主题研究
  - 文献综述
  - 报告撰写
  - 质量审核

#### 项目 4.2：软件开发团队
- **目标**: 构建自动化软件开发团队
- **技术**: AutoGen、代码执行
- **功能**:
  - 需求分析
  - 代码开发
  - 代码审查
  - 测试部署

#### 项目 4.3：智能客服系统
- **目标**: 构建多 Agent 客服系统
- **技术**: OpenAI Agents、路由
- **功能**:
  - 意图识别
  - 专业分流
  - 多轮对话
  - 人工接管

#### 项目 4.4：Agent 编排平台
- **目标**: 构建可视化的 Agent 编排平台
- **技术**: DAG 编排、消息队列
- **功能**:
  - 可视化编排
  - Agent 管理
  - 流程监控
  - 性能分析

### 5.7 阶段 4 自检清单

- [ ] 理解多 Agent 架构模式
- [ ] 掌握 CrewAI 框架和使用
- [ ] 掌握 AutoGen 框架和使用
- [ ] 掌握 OpenAI Agents SDK
- [ ] 能够实现 Agent 编排和调度
- [ ] 理解 Agent 通信机制
- [ ] 完成 4 个实战项目
- [ ] 能够设计复杂的多 Agent 系统

---
