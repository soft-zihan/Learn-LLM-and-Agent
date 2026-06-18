# AI Agent 基础与架构

> 📅 **更新时间**: 2026-06-18  

---

## 目录

- [1. Agent 基础概念](#1-agent-基础概念)
- [2. 单 Agent 工作流](#2-单-agent-工作流)
- [3. LangChain Agent 开发](#3-langchain-agent-开发)
- [4. LangGraph 单 Agent 状态图](#4-langgraph-单-agent-状态图)
- [5. 规划与推理](#5-规划与推理)

---

## 1. Agent 基础概念

### 1.1 什么是 AI Agent

#### Agent 定义与特征

AI Agent（智能代理）是一种能够**感知环境、自主决策、执行行动**以达成特定目标的智能系统。与普通的 LLM 对话不同，Agent 具备以下核心特征：

| 特征 | 说明 | 示例 |
|------|------|------|
| **自主性 (Autonomy)** | 能够在无需人工干预的情况下做出决策和执行行动 | 自动调用搜索工具获取信息 |
| **反应性 (Reactivity)** | 能够感知环境变化并做出响应 | 检测代码错误后自动修复 |
| **主动性 (Proactiveness)** | 能够主动采取行动以达成目标 | 主动规划多步骤任务 |
| **社交能力 (Social Ability)** | 能够与其他 Agent 或人类交互协作 | 请求人类审核关键决策 |

```python
# 普通 LLM vs Agent 的对比

# 普通 LLM：只能根据输入生成文本
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{"role": "user", "content": "北京今天天气如何？"}]
)
# 结果：基于训练数据回答，无法获取实时信息

# Agent：可以调用工具获取实时信息
from langchain.agents import create_react_agent
from langchain.tools import Tool

# 定义天气查询工具
def get_weather(city: str) -> str:
    """调用天气 API 获取实时天气"""
    import requests
    response = requests.get(f"https://api.weather.com/{city}")
    return response.json()["weather"]

weather_tool = Tool(
    name="weather_query",
    func=get_weather,
    description="查询指定城市的实时天气"
)

# Agent 会自动判断何时使用工具
agent = create_react_agent(llm, [weather_tool], prompt)
result = agent.invoke({"input": "北京今天天气如何？"})
# 结果：调用天气 API 获取真实数据后回答
```

#### Agent vs 普通 LLM

```mermaid
graph LR
    A[用户输入] --> B{普通 LLM}
    B --> C[基于训练数据生成回答]
    C --> D[输出文本]
    
    A2[用户输入] --> E{AI Agent}
    E --> F[理解意图]
    F --> G{需要工具？}
    G -->|是| H[调用工具]
    G -->|否| I[直接回答]
    H --> J[获取工具结果]
    J --> K[综合信息生成回答]
    K --> D2[输出结果]
    
    style E fill:#e1f5ff
    style G fill:#fff3cd
    style H fill:#d4edda
```

**核心差异对比表**：

| 维度 | 普通 LLM | AI Agent |
|------|---------|----------|
| **能力边界** | 仅限文本生成 | 可通过工具扩展能力 |
| **数据来源** | 训练数据（静态） | 实时数据 + 训练数据 |
| **执行能力** | 无 | 可执行代码、调用 API |
| **决策能力** | 被动响应 | 主动规划、多步推理 |
| **记忆能力** | 单次对话上下文 | 长期记忆 + 短期记忆 |
| **错误处理** | 无法自我纠正 | 可反思、重试、调整策略 |
| **适用场景** | 内容生成、问答 | 复杂任务自动化、问题解决 |

#### Agent 能力层级

根据自主性和能力范围，Agent 可分为四个层级：

```mermaid
graph TD
    A[L1: 对话式 Agent] --> B[L2: 工具使用 Agent]
    B --> C[L3: 规划式 Agent]
    C --> D[L4: 自主式 Agent]
    
    A1[简单问答<br/>无工具调用] -.-> A
    B1[单步工具调用<br/>ReAct 模式] -.-> B
    C1[多步任务规划<br/>Plan-and-Execute] -.-> C
    D1[完全自主执行<br/>自我反思优化] -.-> D
    
    style A fill:#cce5ff
    style B fill:#d4edda
    style C fill:#fff3cd
    style D fill:#f8d7da
```

**各层级详细说明**：

| 层级 | 名称 | 特征 | 典型应用 | 技术实现 |
|------|------|------|----------|----------|
| **L1** | 对话式 Agent | 基于 LLM 的对话，无外部工具 | 聊天机器人、问答系统 | 纯 LLM API |
| **L2** | 工具使用 Agent | 可调用单一或多个工具 | 搜索引擎助手、代码执行器 | ReAct、Toolformer |
| **L3** | 规划式 Agent | 能分解复杂任务、制定计划 | 项目管理助手、数据分析 | Plan-and-Execute、Tree of Thoughts |
| **L4** | 自主式 Agent | 完全自主执行、自我优化 | 自动化运维、智能客服 | Reflection、AutoGPT |

#### 自主性程度

Agent 的自主性是一个**连续谱**，而非二元状态：

```
完全人工控制 ◄━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━► 完全自主
     │              │              │              │
   手动执行     人工审批+自动执行   自动执行+人工审核   完全自主
   (Manual)    (Human-in-loop)   (Human-on-loop)  (Autonomous)
```

**自主性级别选择指南**：

| 场景 | 推荐自主性 | 原因 | 示例 |
|------|-----------|------|------|
| 金融交易 | 人工审批 + 自动执行 | 高风险，需人工审核 | 股票交易 Agent |
| 代码生成 | 自动执行 + 人工审核 | 中等风险，需代码审查 | GitHub Copilot |
| 数据查询 | 自动执行 | 低风险，只读操作 | SQL 查询助手 |
| 内容创作 | 完全自主 | 无安全风险 | 文章生成 Agent |

---

### 1.2 Agent 核心组件

一个完整的 AI Agent 由五大核心组件构成，它们协同工作实现智能行为：

```mermaid
graph TB
    subgraph Agent核心架构
        A[LLM 大脑] --> B[记忆系统]
        A --> C[工具集]
        A --> D[规划模块]
        A --> E[反思机制]
        
        B --> B1[短期记忆]
        B --> B2[长期记忆]
        
        C --> C1[API 工具]
        C --> C2[代码执行]
        C --> C3[数据库查询]
        
        D --> D1[任务分解]
        D --> D2[步骤规划]
        D --> D3[资源分配]
        
        E --> E1[错误检测]
        E --> E2[策略调整]
        E --> E3[经验积累]
    end
    
    F[用户输入] --> A
    A --> G[输出结果]
    
    style A fill:#cce5ff
    style B fill:#d4edda
    style C fill:#fff3cd
    style D fill:#e2e3ff
    style E fill:#f8d7da
```

#### 1. LLM（大脑）

LLM 是 Agent 的核心决策引擎，负责：
- **理解输入**：解析用户意图、提取关键信息
- **生成决策**：选择使用哪些工具、如何执行任务
- **综合输出**：整合工具结果，生成最终回答

**主流 LLM 对比（2025-2026）**：

| 模型 | 提供商 | 工具调用能力 | 上下文窗口 | 适用场景 | 成本 |
|------|--------|-------------|-----------|----------|------|
| GPT-5.5 | OpenAI | ★★★★★ | 200K | 通用Agent | 中高 |
| Claude 4.8 | Anthropic | ★★★★★ | 200K | 复杂推理 | 中高 |
| Gemini 3.1 Pro | Google | ★★★★★ | 1M | 大规模上下文 | 中 |
| Qwen 3 | 阿里云 | ★★★★☆ | 256K | 中文场景 | 低 |
| DeepSeek V3.5 | DeepSeek | ★★★★☆ | 256K | 代码生成 | 低 |

```python
# 不同 LLM 的 Agent 初始化示例

# OpenAI GPT-5.2
from langchain_openai import ChatOpenAI

gpt52 = ChatOpenAI(
    model="gpt-5.2",
    temperature=0,  # Agent 通常需要确定性输出
    api_key="your-api-key"
)

# Anthropic Claude
from langchain_anthropic import ChatAnthropic

claude = ChatAnthropic(
    model="claude-4-8-20251101",
    temperature=0,
    max_tokens=4096
)

# 国产模型 - 通义千问
from langchain_openai import ChatOpenAI

qwen = ChatOpenAI(
    model="qwen2.5-72b-instruct",
    temperature=0,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="your-dashscope-api-key"
)
```

#### 2. 记忆系统（短期/长期）

记忆系统是 Agent 的"经验库"，分为两类：

**短期记忆（Short-term Memory）**：
- **作用**：维护当前对话上下文
- **实现**：对话历史列表、滑动窗口
- **容量**：受 LLM 上下文窗口限制
- **特点**：快速访问、自动管理

```python
# 短期记忆实现 - 对话历史
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class ShortTermMemory:
    def __init__(self, max_messages: int = 20):
        self.messages = []
        self.max_messages = max_messages
    
    def add_message(self, message):
        self.messages.append(message)
        # 滑动窗口：保留最近 N 条消息
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self):
        return self.messages
    
    def summarize(self, llm):
        """当消息过多时，压缩早期对话"""
        if len(self.messages) > self.max_messages * 0.8:
            # 保留最近的几条，其余进行摘要
            recent = self.messages[-5:]
            old_messages = self.messages[:-5]
            
            # 使用 LLM 生成摘要
            summary_prompt = f"请总结以下对话的关键信息：\n{old_messages}"
            summary = llm.invoke(summary_prompt)
            
            # 用摘要替换旧消息
            self.messages = [SystemMessage(content=summary)] + recent
```

**长期记忆（Long-term Memory）**：
- **作用**：跨会话保存重要信息
- **实现**：向量数据库、知识图谱、文件存储
- **容量**：理论上无限
- **特点**：需要主动检索、可持久化

```python
# 长期记忆实现 - 向量数据库
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class LongTermMemory:
    def __init__(self, persist_directory: str = "./memory"):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
    
    def store(self, key: str, value: str, metadata: dict = None):
        """存储记忆"""
        doc = Document(
            page_content=value,
            metadata={"key": key, **(metadata or {})}
        )
        self.vectorstore.add_documents([doc])
    
    def retrieve(self, query: str, k: int = 5) -> list:
        """检索相关记忆"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs
    
    def update(self, key: str, new_value: str):
        """更新记忆（先删除旧记忆，再添加新记忆）"""
        # 删除旧记忆
        self.vectorstore.delete(where={"key": key})
        # 添加新记忆
        self.store(key, new_value)
```

#### 3. 工具使用（手脚）

工具是 Agent 与外部世界交互的桥梁，使 Agent 能够：
- 获取实时信息（搜索、API 调用）
- 执行操作（文件操作、数据库查询）
- 调用专业服务（代码执行、图像生成）

```python
# 工具定义示例
from langchain.tools import Tool
from pydantic import BaseModel, Field

# 方式 1：简单函数工具
def search_knowledge_base(query: str) -> str:
    """搜索知识库并返回相关文档"""
    # 实现搜索逻辑
    return f"搜索结果: {query}"

search_tool = Tool(
    name="knowledge_search",
    func=search_knowledge_base,
    description="当需要查找知识库中的信息时使用此工具"
)

# 方式 2：带参数验证的工具（推荐）
from langchain.tools import StructuredTool

class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    category: str = Field(description="搜索类别", default="all")

def structured_search(query: str, category: str = "all") -> str:
    """结构化搜索工具"""
    return f"在 {category} 类别中搜索: {query}"

structured_tool = StructuredTool.from_function(
    func=structured_search,
    name="structured_search",
    description="支持分类的结构化搜索",
    args_schema=SearchInput
)
```

**工具设计最佳实践**：

| 原则 | 说明 | 示例 |
|------|------|------|
| **单一职责** | 每个工具只做一件事 | `search_db` vs `search_db_and_analyze` |
| **清晰描述** | 描述要详细，包含使用场景 | "当需要查询用户信息时使用" |
| **参数简洁** | 参数数量 ≤ 5，类型明确 | 使用 Pydantic 定义 schema |
| **错误处理** | 返回友好错误信息 | 返回 "错误：用户不存在" 而非抛出异常 |
| **结果格式化** | 输出结构化、易解析 | JSON 格式优于自由文本 |

#### 4. 规划能力（思维）

规划能力使 Agent 能够处理复杂任务，将大目标分解为可执行的小步骤：

```mermaid
graph LR
    A[复杂任务] --> B[目标理解]
    B --> C[任务分解]
    C --> D[依赖分析]
    D --> E[步骤排序]
    E --> F[资源分配]
    F --> G[执行计划]
    
    style A fill:#cce5ff
    style G fill:#d4edda
```

```python
# 简单规划示例
from langchain_core.prompts import PromptTemplate

planning_prompt = PromptTemplate.from_template("""
你是一个任务规划专家。请将以下复杂任务分解为可执行的步骤。

任务：{task}

要求：
1. 每个步骤应该是原子操作
2. 标注步骤间的依赖关系
3. 估计每个步骤的复杂度（1-5）
4. 识别潜在风险

输出格式（JSON）：
{{
    "steps": [
        {{
            "id": 1,
            "action": "步骤描述",
            "tool": "使用的工具",
            "depends_on": [],
            "complexity": 2,
            "risks": ["潜在风险"]
        }}
    ]
}}

请输出规划结果：
""")

def plan_task(task: str, llm) -> dict:
    """任务规划函数"""
    import json
    
    response = llm.invoke(planning_prompt.format(task=task))
    
    # 解析 JSON 输出
    try:
        # 提取 JSON 部分
        json_str = response.content
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        
        plan = json.loads(json_str.strip())
        return plan
    except Exception as e:
        return {"error": f"规划解析失败: {e}", "raw": response.content}

# 使用示例
task = "分析公司 2024 年 Q1-Q4 的销售数据，生成可视化报告"
plan = plan_task(task, gpt4o)
print(plan)
```

#### 5. 反思机制（元认知）

反思机制使 Agent 能够自我评估、发现错误并改进策略：

```python
# 反思机制实现
from langchain_core.messages import SystemMessage

class ReflectionMechanism:
    def __init__(self, llm):
        self.llm = llm
        self.reflection_prompt = """
你是一个自我反思助手。请分析以下执行过程，识别问题并提出改进方案。

原始任务：{task}
执行步骤：{steps}
最终结果：{result}

请从以下维度进行反思：
1. **正确性**：结果是否满足任务要求？
2. **效率**：是否有更优的执行路径？
3. **工具使用**：工具选择是否合适？
4. **错误处理**：是否有未处理的异常？
5. **改进建议**：下次如何做得更好？

输出格式：
{{
    "is_success": true/false,
    "issues": ["问题列表"],
    "suggestions": ["改进建议"],
    "confidence": 0.0-1.0
}}
"""
    
    def reflect(self, task: str, steps: list, result: str) -> dict:
        """执行反思"""
        import json
        
        prompt = self.reflection_prompt.format(
            task=task,
            steps="\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)]),
            result=result
        )
        
        response = self.llm.invoke(prompt)
        
        # 解析反思结果
        try:
            json_str = response.content
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except:
            return {"is_success": False, "issues": ["反思结果解析失败"]}
    
    def should_retry(self, reflection_result: dict, max_retries: int = 3, current_retry: int = 0) -> bool:
        """判断是否应该重试"""
        if current_retry >= max_retries:
            return False
        if not reflection_result.get("is_success", False):
            return True
        return False

# 使用示例
reflector = ReflectionMechanism(gpt4o)

task = "查询北京天气并生成报告"
steps = ["调用天气 API", "解析结果", "生成报告"]
result = "报告生成成功"

reflection = reflector.reflect(task, steps, result)
print(f"任务成功: {reflection['is_success']}")
print(f"置信度: {reflection['confidence']}")
```

---

### 1.3 Agent 架构模式

#### ReAct (Reason + Act)

**ReAct** 是最经典的 Agent 架构模式，核心思想是**交替进行推理和行动**：

```mermaid
sequenceDiagram
    participant U as 用户
    participant A as Agent
    participant L as LLM
    participant T as 工具
    
    U->>A: 输入问题
    A->>L: 思考 (Thought)
    L-->>A: 决定行动
    A->>T: 执行行动 (Action)
    T-->>A: 返回结果 (Observation)
    A->>L: 基于结果继续思考
    Note over A,L: 循环直到得出答案
    A-->>U: 输出最终答案 (Final Answer)
```

**ReAct 工作循环**：

```
Thought → Action → Observation → Thought → Action → Observation → ... → Final Answer
```

```python
# ReAct 完整实现示例
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# 1. 定义工具
def calculate(expression: str) -> str:
    """安全计算数学表达式"""
    try:
        # 使用 eval 时需注意安全，生产环境应使用安全的数学解析器
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含非法字符"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"

def search_wikipedia(query: str) -> str:
    """搜索维基百科"""
    import wikipedia
    try:
        results = wikipedia.search(query, results=2)
        if results:
            page = wikipedia.page(results[0])
            return page.summary[:500]  # 返回前 500 字符
        return "未找到相关条目"
    except Exception as e:
        return f"搜索失败: {str(e)}"

tools = [
    Tool(
        name="calculator",
        func=calculate,
        description="用于数学计算，输入应为数学表达式如 '2 + 3 * 4'"
    ),
    Tool(
        name="wikipedia_search",
        func=search_wikipedia,
        description="搜索维基百科获取信息，输入为搜索关键词"
    )
]

# 2. 初始化 LLM
llm = ChatOpenAI(model="gpt-5.2", temperature=0)

# 3. 获取 ReAct Prompt
prompt = hub.pull("hwchase17/react")

# 4. 创建 Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# 5. 创建执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 打印执行过程
    max_iterations=10,  # 最大循环次数
    handle_parsing_errors=True  # 处理解析错误
)

# 6. 执行
result = agent_executor.invoke({
    "input": "爱因斯坦获得诺贝尔奖的那一年，他多少岁？先搜索获奖年份，再计算年龄。"
})

print(f"答案: {result['output']}")
```

**ReAct Prompt 模板详解**：

```
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
```

**关键要素说明**：

| 要素 | 作用 | 示例 |
|------|------|------|
| **Thought** | 推理过程，解释为什么选择某个行动 | "我需要先搜索爱因斯坦获奖年份" |
| **Action** | 选择使用的工具 | "wikipedia_search" |
| **Action Input** | 工具的输入参数 | "爱因斯坦 诺贝尔奖" |
| **Observation** | 工具返回的结果 | "爱因斯坦 1921 年获得诺贝尔物理学奖" |
| **Final Answer** | 最终答案 | "爱因斯坦 1921 年获奖，当时 42 岁" |

#### Plan-and-Execute

**Plan-and-Execute** 模式将任务分为两个阶段：

1. **Plan（规划）**：LLM 生成完整的执行计划
2. **Execute（执行）**：按顺序执行计划中的步骤

```mermaid
graph TB
    A[用户输入] --> B[Planner]
    B --> C[生成计划]
    C --> D[执行器]
    D --> E[步骤 1]
    E --> F[步骤 2]
    F --> G[步骤 N]
    G --> H[整合结果]
    H --> I[最终输出]
    
    style B fill:#cce5ff
    style D fill:#d4edda
    style H fill:#fff3cd
```

```python
# Plan-and-Execute 实现
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5.2", temperature=0)

# 规划器
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个任务规划专家。请将复杂任务分解为可执行的步骤计划。

输出要求：
1. 每个步骤应该清晰、可执行
2. 标注步骤间的依赖关系
3. 使用 JSON 格式输出

示例输出：
```json
{{
    "plan": [
        {{"step": 1, "action": "搜索爱因斯坦基本信息", "depends_on": []}},
        {{"step": 2, "action": "查找诺贝尔奖获奖年份", "depends_on": [1]}},
        {{"step": 3, "action": "计算获奖时年龄", "depends_on": [1, 2]}}
    ]
}}
```"""),
    ("human", "{input}")
])

planner = planner_prompt | llm

# 执行器（简化版）
def execute_plan(plan_json: str, tools: list) -> str:
    """执行计划"""
    import json
    
    plan = json.loads(plan_json)
    results = {}
    
    for step in plan["plan"]:
        step_num = step["step"]
        action = step["action"]
        
        # 检查依赖
        deps_met = all(dep in results for dep in step.get("depends_on", []))
        if not deps_met:
            return f"错误：步骤 {step_num} 的依赖未满足"
        
        # 执行步骤（简化：实际应调用相应工具）
        print(f"执行步骤 {step_num}: {action}")
        results[step_num] = f"步骤 {step_num} 完成"
    
    return "计划执行完成"

# 使用示例
import asyncio

async def plan_and_execute(task: str):
    # 生成计划
    plan_response = await planner.ainvoke({"input": task})
    print(f"生成的计划:\n{plan_response.content}")
    
    # 执行计划
    result = execute_plan(plan_response.content, [])
    print(f"执行结果: {result}")

# 运行
# asyncio.run(plan_and_execute("分析苹果公司 2024 年财报"))
```

#### Reflection

**Reflection** 模式通过自我评估持续改进输出质量：

```mermaid
graph LR
    A[初始输入] --> B[生成初稿]
    B --> C[自我评估]
    C --> D{质量达标？}
    D -->|否| E[生成改进建议]
    E --> F[修订输出]
    F --> C
    D -->|是| G[输出最终结果]
    
    style C fill:#fff3cd
    style D fill:#f8d7da
```

```python
# Reflection 模式实现
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-5.2", temperature=0)

# 生成器
generator_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的内容创作者。请根据用户要求生成高质量内容。"),
    ("human", "{input}")
])

generator = generator_prompt | llm

# 反思器
reflector_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个严格的内容评审专家。请从以下维度评估内容质量：

1. **准确性**：信息是否准确无误？
2. **完整性**：是否覆盖所有要求？
3. **逻辑性**：论述是否清晰、逻辑是否严密？
4. **可读性**：语言是否流畅、易读？
5. **专业性**：是否体现专业水准？

评分标准：1-10 分

输出格式：
```json
{{
    "scores": {{
        "accuracy": 8,
        "completeness": 7,
        "logic": 9,
        "readability": 8,
        "professionalism": 7
    }},
    "average": 7.8,
    "issues": ["问题 1", "问题 2"],
    "suggestions": ["改进建议 1", "改进建议 2"],
    "pass": true/false  // 平均分 >= 8 为通过
}}
```"""),
    ("human", "请评估以下内容：\n{content}")
])

reflector = reflector_prompt | llm

# 修订器
reviser_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个内容修订专家。请根据评审意见改进内容。

原始内容：
{content}

评审意见：
{feedback}

请针对每个问题进行修订，输出改进后的完整内容。"""),
    ("human", "请开始修订：")
])

reviser = reviser_prompt | llm

# 完整 Reflection 流程
async def reflection_workflow(input_text: str, max_iterations: int = 3):
    """Reflection 工作流"""
    import json
    
    # 生成初稿
    content = (await generator.ainvoke({"input": input_text})).content
    print(f"初稿生成完成")
    
    for i in range(max_iterations):
        print(f"\n--- 第 {i+1} 轮评审 ---")
        
        # 评审
        feedback_raw = (await reflector.ainvoke({"content": content})).content
        
        # 解析评审结果
        try:
            json_str = feedback_raw.split("```json")[1].split("```")[0]
            feedback = json.loads(json_str.strip())
        except:
            print("评审结果解析失败，结束循环")
            break
        
        print(f"平均分: {feedback['average']}")
        print(f"问题: {feedback['issues']}")
        
        # 检查是否通过
        if feedback.get("pass", False):
            print("✓ 质量达标！")
            break
        
        # 修订
        print("进行修订...")
        content = (await reviser.ainvoke({
            "content": content,
            "feedback": feedback_raw
        })).content
    
    return content

# 使用示例
# result = asyncio.run(reflection_workflow("写一篇关于 AI 发展趋势的 1000 字文章"))
```

#### Toolformer

**Toolformer** 模式让 LLM 学习何时、如何使用工具，通过微调或 prompt 工程实现：

```python
# Toolformer 风格的工具选择
from langchain_core.prompts import ChatPromptTemplate

tool_selector_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个工具选择专家。根据用户请求，选择最合适的工具。

可用工具：
- calculator: 数学计算
- search: 网络搜索
- code_executor: 代码执行
- database_query: 数据库查询
- file_reader: 文件读取

输出格式（仅输出工具名称）：
calculator / search / code_executor / database_query / file_reader"""),
    ("human", "{input}")
])

tool_selector = tool_selector_prompt | llm

def toolformer_dispatch(user_input: str, tools: dict):
    """Toolformer 风格的工具分发"""
    # 选择工具
    tool_name = tool_selector.invoke({"input": user_input}).content.strip()
    
    # 分发到对应工具
    if tool_name in tools:
        return tools[tool_name](user_input)
    else:
        return f"错误：未知工具 {tool_name}"

# 使用示例
tools_dict = {
    "calculator": lambda x: f"计算结果: {eval(x.split(':')[-1])}",
    "search": lambda x: f"搜索结果: {x}",
    # ...
}

# result = toolformer_dispatch("请计算: 2 + 3 * 4", tools_dict)
```

---

## 2. 单 Agent 工作流

### 2.1 感知-思考-行动循环

Agent 的核心工作流是 **Perceive-Think-Act（感知-思考-行动）循环**，也称为 **PTA 循环**：

```mermaid
graph LR
    A[感知 Perceive] --> B[思考 Think]
    B --> C[决策 Decide]
    C --> D[行动 Act]
    D --> E[观察 Observe]
    E --> A
    
    style A fill:#cce5ff
    style B fill:#d4edda
    style C fill:#fff3cd
    style D fill:#f8d7da
    style E fill:#e2e3ff
```

**循环各阶段详解**：

| 阶段 | 输入 | 处理 | 输出 | 示例 |
|------|------|------|------|------|
| **感知** | 用户输入、环境状态 | 信息提取、意图识别 | 结构化理解 | "用户想查询北京天气" |
| **思考** | 感知结果、历史记忆 | 推理分析、知识检索 | 决策依据 | "需要调用天气 API" |
| **决策** | 思考结果、可用工具 | 工具选择、参数准备 | 行动计划 | "调用 weather_api，参数 city=北京" |
| **行动** | 行动计划 | 工具执行、API 调用 | 执行结果 | "API 返回：晴，25°C" |
| **观察** | 执行结果 | 结果验证、错误检测 | 反馈信息 | "结果正常，可以生成回答" |

```python
# 完整的 PTA 循环实现
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from typing import List, Dict, Any
import json

class PTAAgent:
    """感知-思考-行动 Agent"""
    
    def __init__(self, llm, tools: List[BaseTool], max_iterations: int = 10):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.memory = []
        self.trace = []  # 执行轨迹记录
    
    def run(self, user_input: str) -> Dict[str, Any]:
        """执行 PTA 循环"""
        self.memory.append(HumanMessage(content=user_input))
        
        for iteration in range(self.max_iterations):
            # 1. 感知阶段
            perception = self._perceive()
            self._trace("perceive", perception)
            
            # 2. 思考阶段
            thought = self._think(perception)
            self._trace("think", thought)
            
            # 3. 决策阶段
            decision = self._decide(thought)
            self._trace("decide", decision)
            
            # 4. 判断是否完成
            if decision.get("type") == "final_answer":
                self.memory.append(AIMessage(content=decision["content"]))
                return {
                    "output": decision["content"],
                    "iterations": iteration + 1,
                    "trace": self.trace
                }
            
            # 5. 行动阶段
            action_result = self._act(decision)
            self._trace("act", action_result)
            
            # 6. 观察阶段
            observation = self._observe(action_result)
            self._trace("observe", observation)
            
            # 7. 更新记忆
            self.memory.append(SystemMessage(content=f"工具执行结果: {observation}"))
        
        return {
            "output": "达到最大迭代次数，任务未完成",
            "iterations": self.max_iterations,
            "trace": self.trace
        }
    
    def _perceive(self) -> Dict[str, Any]:
        """感知：理解用户输入和当前状态"""
        return {
            "user_input": self.memory[-1].content if self.memory else "",
            "context_length": len(self.memory),
            "available_tools": list(self.tools.keys())
        }
    
    def _think(self, perception: Dict) -> Dict[str, Any]:
        """思考：分析应该采取的行动"""
        prompt = f"""基于以下信息，分析下一步应该做什么：

用户请求: {perception['user_input']}
可用工具: {', '.join(perception['available_tools'])}

请分析：
1. 用户的核心需求是什么？
2. 需要使用哪些工具？
3. 工具的参数应该是什么？

输出 JSON 格式。"""
        
        response = self.llm.invoke([SystemMessage(content=prompt)])
        
        try:
            return json.loads(response.content)
        except:
            return {"analysis": "需要进一步信息", "next_step": "ask_user"}
    
    def _decide(self, thought: Dict) -> Dict[str, Any]:
        """决策：决定具体行动"""
        if "tool_name" in thought:
            return {
                "type": "tool_call",
                "tool_name": thought["tool_name"],
                "tool_input": thought.get("tool_input", {})
            }
        else:
            return {
                "type": "final_answer",
                "content": thought.get("answer", "无法回答")
            }
    
    def _act(self, decision: Dict) -> Any:
        """行动：执行工具调用"""
        if decision["type"] != "tool_call":
            return None
        
        tool_name = decision["tool_name"]
        if tool_name not in self.tools:
            return f"错误：工具 {tool_name} 不存在"
        
        tool = self.tools[tool_name]
        try:
            if isinstance(decision["tool_input"], dict):
                return tool.run(tool_input=decision["tool_input"])
            else:
                return tool.run(tool_input=decision["tool_input"])
        except Exception as e:
            return f"工具执行错误: {str(e)}"
    
    def _observe(self, action_result: Any) -> str:
        """观察：分析行动结果"""
        if action_result is None:
            return "无结果"
        if isinstance(action_result, str) and action_result.startswith("错误"):
            return f"失败: {action_result}"
        return f"成功: {str(action_result)[:200]}"
    
    def _trace(self, stage: str, data: Any):
        """记录执行轨迹"""
        self.trace.append({
            "stage": stage,
            "data": str(data)[:500],  # 限制长度
            "timestamp": self._get_timestamp()
        })
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

# 使用示例
if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import Tool
    
    llm = ChatOpenAI(model="gpt-5.2", temperature=0)
    
    def get_weather(city: str) -> str:
        return f"{city}天气：晴，25°C"
    
    tools = [
        Tool(name="weather", func=get_weather, description="查询天气")
    ]
    
    agent = PTAAgent(llm, tools)
    result = agent.run("北京今天天气如何？")
    print(f"输出: {result['output']}")
    print(f"迭代次数: {result['iterations']}")
```

### 2.2 任务分解

复杂任务需要分解为可执行的子任务，这是 Agent 的核心能力之一。

#### 目标理解

```python
# 目标理解与意图识别
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class TaskUnderstanding(BaseModel):
    """任务理解结果"""
    main_goal: str = Field(description="主要目标")
    key_entities: list[str] = Field(description="关键实体")
    required_actions: list[str] = Field(description="需要的动作")
    constraints: list[str] = Field(description="约束条件")
    complexity: int = Field(description="复杂度 1-5", ge=1, le=5)
    estimated_steps: int = Field(description="预估步骤数")

intent_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个任务分析专家。请深入理解用户请求，提取关键信息。

输出格式为 JSON，包含：
- main_goal: 主要目标
- key_entities: 关键实体（人名、地名、时间等）
- required_actions: 需要的动作
- constraints: 约束条件
- complexity: 复杂度（1-5）
- estimated_steps: 预估需要的步骤数"""),
    ("human", "{task}")
])

# 使用 Pydantic 输出
intent_parser = intent_prompt | llm.with_structured_output(TaskUnderstanding)

# 使用示例
task = "帮我分析特斯拉 2024 年 Q1 到 Q4 的销量数据，对比同比变化，生成可视化图表"
understanding = intent_parser.invoke({"task": task})

print(f"主要目标: {understanding.main_goal}")
print(f"关键实体: {understanding.key_entities}")
print(f"复杂度: {understanding.complexity}")
print(f"预估步骤: {understanding.estimated_steps}")
```

#### 子任务划分

```python
# 任务分解为子任务
from typing import List
from pydantic import BaseModel

class SubTask(BaseModel):
    """子任务定义"""
    id: int
    description: str
    tool_required: str
    dependencies: List[int] = Field(default_factory=list)
    estimated_complexity: int = Field(ge=1, le=5)
    success_criteria: str

class TaskPlan(BaseModel):
    """任务计划"""
    subtasks: List[SubTask]
    execution_order: List[int]
    parallel_groups: List[List[int]] = Field(
        default_factory=list,
        description="可并行执行的子任务组"
    )

task_decomposition_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个任务分解专家。请将复杂任务分解为可执行的子任务。

分解原则：
1. **原子性**：每个子任务应该是不可再分的原子操作
2. **独立性**：尽量减少子任务间的依赖
3. **可验证性**：每个子任务应该有明确的完成标准
4. **可并行性**：识别可以并行执行的子任务

输出格式：
```json
{{
    "subtasks": [
        {{
            "id": 1,
            "description": "获取特斯拉 2024 年各季度销量数据",
            "tool_required": "data_api",
            "dependencies": [],
            "estimated_complexity": 2,
            "success_criteria": "成功获取 Q1-Q4 销量数据"
        }}
    ],
    "execution_order": [1, 2, 3, 4],
    "parallel_groups": [[1, 2], [3], [4]]
}}
```"""),
    ("human", "任务：{task}")
])

decomposer = task_decomposition_prompt | llm.with_structured_output(TaskPlan)

# 使用示例
plan = decomposer.invoke({"task": task})
print(f"子任务数量: {len(plan.subtasks)}")
for subtask in plan.subtasks:
    print(f"  {subtask.id}. {subtask.description} [依赖: {subtask.dependencies}]")
print(f"执行顺序: {plan.execution_order}")
print(f"可并行组: {plan.parallel_groups}")
```

#### 依赖关系分析

```mermaid
graph TD
    A[1. 获取销量数据] --> C[3. 计算同比变化]
    B[2. 获取去年同期数据] --> C
    C --> D[4. 生成可视化图表]
    C --> E[5. 撰写分析报告]
    D --> E
    
    style A fill:#cce5ff
    style B fill:#cce5ff
    style C fill:#fff3cd
    style D fill:#d4edda
    style E fill:#f8d7da
```

```python
# 依赖关系解析与执行
from collections import defaultdict, deque
from typing import Dict, Set

class DependencyResolver:
    """依赖关系解析器"""
    
    def __init__(self):
        self.dependencies: Dict[int, Set[int]] = defaultdict(set)
        self.tasks: Dict[int, SubTask] = {}
    
    def add_task(self, task: SubTask):
        """添加任务"""
        self.tasks[task.id] = task
        for dep in task.dependencies:
            self.dependencies[task.id].add(dep)
    
    def get_execution_order(self) -> List[int]:
        """获取拓扑排序的执行顺序"""
        # 计算入度
        in_degree = defaultdict(int)
        for task_id in self.tasks:
            if task_id not in in_degree:
                in_degree[task_id] = 0
            for dep in self.dependencies[task_id]:
                in_degree[task_id] += 1
        
        # Kahn 算法
        queue = deque([task_id for task_id in self.tasks if in_degree[task_id] == 0])
        order = []
        
        while queue:
            task_id = queue.popleft()
            order.append(task_id)
            
            for other_id, deps in self.dependencies.items():
                if task_id in deps:
                    deps.remove(task_id)
                    if len(deps) == 0:
                        queue.append(other_id)
        
        if len(order) != len(self.tasks):
            raise ValueError("检测到循环依赖！")
        
        return order
    
    def get_parallel_groups(self) -> List[List[int]]:
        """获取可并行执行的任务组"""
        # 简化实现：基于层级分组
        levels = []
        remaining = set(self.tasks.keys())
        completed = set()
        
        while remaining:
            # 找出当前可执行的任务（依赖都已满足）
            ready = [
                task_id for task_id in remaining
                if self.dependencies[task_id].issubset(completed)
            ]
            
            if not ready:
                raise ValueError("检测到循环依赖！")
            
            levels.append(ready)
            completed.update(ready)
            remaining -= set(ready)
        
        return levels

# 使用示例
resolver = DependencyResolver()

# 添加任务
for subtask in plan.subtasks:
    resolver.add_task(subtask)

# 获取执行顺序
order = resolver.get_execution_order()
print(f"执行顺序: {order}")

# 获取并行组
parallel_groups = resolver.get_parallel_groups()
print(f"并行组: {parallel_groups}")
```

### 2.3 状态管理

Agent 在执行过程中需要维护状态，包括上下文、进度、错误信息等。

#### 上下文维护

```python
# 上下文管理器
from langchain_core.messages import BaseMessage
from typing import Optional
import json

class ContextManager:
    """上下文管理器"""
    
    def __init__(
        self,
        max_tokens: int = 4000,
        sliding_window: int = 10,
        summary_threshold: float = 0.8
    ):
        self.max_tokens = max_tokens
        self.sliding_window = sliding_window
        self.summary_threshold = summary_threshold
        self.messages: List[BaseMessage] = []
        self.summary: Optional[str] = None
    
    def add_message(self, message: BaseMessage):
        """添加消息"""
        self.messages.append(message)
        self._check_and_compress()
    
    def get_context(self) -> List[BaseMessage]:
        """获取当前上下文"""
        if self.summary:
            from langchain_core.messages import SystemMessage
            return [SystemMessage(content=self.summary)] + self.messages[-self.sliding_window:]
        return self.messages[-self.sliding_window:]
    
    def _check_and_compress(self):
        """检查是否需要压缩上下文"""
        # 估算 token 数量（简化：按字符数 / 4）
        total_chars = sum(len(msg.content) for msg in self.messages)
        estimated_tokens = total_chars / 4
        
        if estimated_tokens > self.max_tokens * self.summary_threshold:
            self._compress()
    
    def _compress(self, llm=None):
        """压缩上下文"""
        if llm is None:
            # 简单压缩：只保留最近的消息
            compress_point = len(self.messages) // 2
            old_messages = self.messages[:compress_point]
            self.messages = self.messages[compress_point:]
            
            # 更新摘要
            if self.summary:
                self.summary += "\n" + "\n".join([msg.content for msg in old_messages])
            else:
                self.summary = "\n".join([msg.content for msg in old_messages])
        else:
            # 使用 LLM 生成摘要
            old_messages = self.messages[:-self.sliding_window]
            text_to_summarize = "\n".join([msg.content for msg in old_messages])
            
            summary_prompt = f"请总结以下对话的关键信息：\n{text_to_summarize}"
            self.summary = llm.invoke(summary_prompt).content
            
            # 保留最近的消息
            self.messages = self.messages[-self.sliding_window:]
    
    def clear(self):
        """清空上下文"""
        self.messages = []
        self.summary = None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_chars = sum(len(msg.content) for msg in self.messages)
        return {
            "message_count": len(self.messages),
            "estimated_tokens": total_chars / 4,
            "has_summary": self.summary is not None,
            "summary_length": len(self.summary) if self.summary else 0
        }

# 使用示例
context = ContextManager(max_tokens=4000, sliding_window=5)

# 添加消息
context.add_message(HumanMessage(content="你好，我想查询一下订单"))
context.add_message(AIMessage(content("好的，请提供订单号"))
context.add_message(HumanMessage(content="订单号是 12345"))

print(context.get_stats())
# 输出: {'message_count': 3, 'estimated_tokens': 75.0, ...}
```

#### 进度跟踪

```python
# 进度跟踪器
from enum import Enum
from datetime import datetime

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskProgress:
    """任务进度"""
    
    def __init__(self, task_id: str, description: str, total_steps: int):
        self.task_id = task_id
        self.description = description
        self.total_steps = total_steps
        self.current_step = 0
        self.status = TaskStatus.PENDING
        self.steps: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def start(self):
        """开始任务"""
        self.status = TaskStatus.RUNNING
        self.start_time = datetime.now()
    
    def update_step(self, step_num: int, step_name: str, status: str, details: str = ""):
        """更新步骤进度"""
        self.current_step = step_num
        
        step_record = {
            "step": step_num,
            "name": step_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.steps.append(step_record)
    
    def complete(self):
        """完成任务"""
        self.status = TaskStatus.COMPLETED
        self.end_time = datetime.now()
    
    def fail(self, error: str):
        """任务失败"""
        self.status = TaskStatus.FAILED
        self.end_time = datetime.now()
        self.steps.append({
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_progress(self) -> float:
        """获取进度百分比"""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100
    
    def get_elapsed_time(self) -> Optional[float]:
        """获取已用时间（秒）"""
        if not self.start_time:
            return None
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status.value,
            "progress": self.get_progress(),
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "elapsed_time": self.get_elapsed_time(),
            "steps": self.steps
        }

# 使用示例
progress = TaskProgress(
    task_id="task_001",
    description="分析销售数据",
    total_steps=5
)

progress.start()
progress.update_step(1, "获取数据", "completed", "成功获取 1000 条记录")
progress.update_step(2, "数据清洗", "running")

print(f"进度: {progress.get_progress():.1f}%")
print(f"已用时间: {progress.get_elapsed_time():.2f}秒")
```

#### 检查点保存

```python
# 检查点保存与恢复
import pickle
import os
from pathlib import Path

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, checkpoint_id: str, state: Dict[str, Any]):
        """保存检查点"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        
        checkpoint_data = {
            "state": state,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"检查点已保存: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """加载检查点"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        
        if not checkpoint_path.exists():
            print(f"检查点不存在: {checkpoint_path}")
            return None
        
        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)
        
        print(f"检查点已加载: {checkpoint_path}")
        return checkpoint_data["state"]
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有检查点"""
        checkpoints = []
        for path in self.checkpoint_dir.glob("*.pkl"):
            with open(path, "rb") as f:
                data = pickle.load(f)
                checkpoints.append({
                    "id": path.stem,
                    "timestamp": data["timestamp"],
                    "version": data["version"],
                    "path": str(path)
                })
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            return True
        return False
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新的检查点 ID"""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        
        # 按时间排序
        latest = max(checkpoints, key=lambda x: x["timestamp"])
        return latest["id"]

# 使用示例：Agent 状态检查点
def save_agent_state(agent, progress: TaskProgress, checkpoint_mgr: CheckpointManager):
    """保存 Agent 状态"""
    state = {
        "memory": [msg.dict() for msg in agent.memory],
        "progress": progress.to_dict(),
        "metadata": {
            "agent_type": type(agent).__name__,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    checkpoint_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_mgr.save_checkpoint(checkpoint_id, state)
    
    return checkpoint_id

def restore_agent_state(agent, checkpoint_id: str, checkpoint_mgr: CheckpointManager) -> bool:
    """恢复 Agent 状态"""
    state = checkpoint_mgr.load_checkpoint(checkpoint_id)
    if state is None:
        return False
    
    # 恢复记忆
    from langchain_core.messages import messages_from_dict
    agent.memory = messages_from_dict(state["memory"])
    
    print(f"Agent 状态已恢复: {checkpoint_id}")
    return True
```

#### 错误恢复

```python
# 错误恢复策略
from enum import Enum
from typing import Callable

class ErrorRecoveryStrategy(Enum):
    RETRY = "retry"  # 重试
    SKIP = "skip"  # 跳过
    ABORT = "abort"  # 终止
    FALLBACK = "fallback"  # 降级

class ErrorRecoveryManager:
    """错误恢复管理器"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_history: List[Dict[str, Any]] = []
    
    def handle_error(
        self,
        error: Exception,
        task_name: str,
        strategy: ErrorRecoveryStrategy = ErrorRecoveryStrategy.RETRY,
        fallback_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """处理错误"""
        error_record = {
            "task": task_name,
            "error": str(error),
            "error_type": type(error).__name__,
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy.value
        }
        self.error_history.append(error_record)
        
        print(f"错误发生在 {task_name}: {error}")
        
        if strategy == ErrorRecoveryStrategy.RETRY:
            return {"action": "retry", "retries_left": self.max_retries}
        elif strategy == ErrorRecoveryStrategy.SKIP:
            return {"action": "skip"}
        elif strategy == ErrorRecoveryStrategy.ABORT:
            return {"action": "abort", "error": str(error)}
        elif strategy == ErrorRecoveryStrategy.FALLBACK and fallback_func:
            try:
                result = fallback_func()
                return {"action": "fallback", "result": result}
            except Exception as fallback_error:
                return {"action": "abort", "error": f"降级也失败: {fallback_error}"}
        
        return {"action": "abort", "error": "未知策略"}
    
    def should_retry(self, task_name: str) -> bool:
        """判断是否应该重试"""
        # 统计该任务的错误次数
        error_count = sum(
            1 for err in self.error_history
            if err["task"] == task_name and err["strategy"] == "retry"
        )
        return error_count < self.max_retries
    
    def get_error_report(self) -> Dict[str, Any]:
        """生成错误报告"""
        if not self.error_history:
            return {"total_errors": 0}
        
        # 统计错误类型
        error_types = {}
        for err in self.error_history:
            error_type = err["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "recent_errors": self.error_history[-5:]
        }

# 使用示例
recovery_mgr = ErrorRecoveryManager(max_retries=3, retry_delay=1.0)

def execute_with_recovery(task_func, task_name: str, fallback_func=None):
    """带错误恢复的执行"""
    import time
    
    for attempt in range(recovery_mgr.max_retries + 1):
        try:
            return task_func()
        except Exception as e:
            result = recovery_mgr.handle_error(
                e,
                task_name,
                strategy=ErrorRecoveryStrategy.RETRY,
                fallback_func=fallback_func
            )
            
            if result["action"] == "retry" and attempt < recovery_mgr.max_retries:
                print(f"第 {attempt + 1} 次重试...")
                time.sleep(recovery_mgr.retry_delay * (attempt + 1))  # 指数退避
                continue
            elif result["action"] == "fallback":
                print(f"使用降级方案")
                return result["result"]
            else:
                print(f"任务失败: {result.get('error', '未知错误')}")
                raise e
    
    raise Exception(f"任务 {task_name} 重试 {recovery_mgr.max_retries} 次后仍失败")
```

---

## 3. LangChain Agent 开发

### 3.1 LangChain Agent 基础

LangChain 提供了多种 Agent 类型，适用于不同的使用场景。

#### Agent 类型对比

| Agent 类型 | 适用场景 | 优点 | 缺点 | 复杂度 |
|-----------|---------|------|------|--------|
| **ReAct** | 通用任务 | 简单、灵活 | 可能陷入循环 | ★★☆☆☆ |
| **Structured Chat** | 多轮对话 | 支持上下文、结构化输出 | 较复杂 | ★★★☆☆ |
| **OpenAI Functions** | OpenAI 模型 | 高效、原生支持 | 仅限 OpenAI | ★★☆☆☆ |
| **Self-Ask** | 需要中间答案 | 自动拆分问题 | 适用场景有限 | ★★★☆☆ |
| **Plan-and-Execute** | 复杂长任务 | 有计划性 | 规划可能不准确 | ★★★★☆ |

```python
# LangChain Agent 初始化完整示例
from langchain_openai import ChatOpenAI
from langchain.agents import (
    create_react_agent,
    create_openai_functions_agent,
    create_structured_chat_agent,
    AgentExecutor
)
from langchain import hub
from langchain_core.tools import Tool

# 1. 初始化 LLM
llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0,
    streaming=True  # 支持流式输出
)

# 2. 定义工具
def calculator(expression: str) -> str:
    """计算器"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"错误: {e}"

def search(query: str) -> str:
    """搜索工具"""
    return f"搜索结果: {query}"

tools = [
    Tool(
        name="calculator",
        func=calculator,
        description="用于数学计算，输入数学表达式"
    ),
    Tool(
        name="search",
        func=search,
        description="搜索信息，输入关键词"
    )
]

# 3. 创建不同类型的 Agent

# 3.1 ReAct Agent
react_prompt = hub.pull("hwchase17/react")
react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)
react_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# 3.2 OpenAI Functions Agent (仅适用于 OpenAI 模型)
oa_functions_prompt = hub.pull("hwchase17/openai-functions-agent")
oa_functions_agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=oa_functions_prompt
)
oa_functions_executor = AgentExecutor(
    agent=oa_functions_agent,
    tools=tools,
    verbose=True
)

# 3.3 Structured Chat Agent
structured_prompt = hub.pull("langchain-ai/structured-chat-react")
structured_agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=structured_prompt
)
structured_executor = AgentExecutor(
    agent=structured_agent,
    tools=tools,
    verbose=True
)

# 4. 执行对比
task = "计算 123 * 456 的结果，然后搜索这个数字的历史意义"

print("=== ReAct Agent ===")
result1 = react_executor.invoke({"input": task})
print(f"输出: {result1['output']}\n")

print("=== OpenAI Functions Agent ===")
result2 = oa_functions_executor.invoke({"input": task})
print(f"输出: {result2['output']}\n")

print("=== Structured Chat Agent ===")
result3 = structured_executor.invoke({"input": task})
print(f"输出: {result3['output']}")
```

#### Tool 绑定

```python
# 工具绑定的多种方式

from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from typing import Optional

# 方式 1：简单函数 + Tool 包装
def simple_tool_func(query: str) -> str:
    """简单的搜索工具"""
    return f"搜索: {query}"

tool1 = Tool(
    name="simple_search",
    func=simple_tool_func,
    description="简单搜索，输入关键词"
)

# 方式 2：StructuredTool（推荐，支持参数验证）
class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    category: str = Field(description="搜索类别", default="general")
    limit: int = Field(description="结果数量限制", default=10, ge=1, le=100)

def structured_search(query: str, category: str = "general", limit: int = 10) -> str:
    """结构化搜索"""
    return f"在 {category} 中搜索 '{query}'，返回 {limit} 条结果"

tool2 = StructuredTool.from_function(
    func=structured_search,
    name="structured_search",
    description="支持分类和数量限制的结构化搜索",
    args_schema=SearchInput,
    return_direct=True  # 直接返回结果，不经过 LLM 处理
)

# 方式 3：异步工具
import asyncio

async def async_search(query: str) -> str:
    """异步搜索工具"""
    await asyncio.sleep(1)  # 模拟异步操作
    return f"异步搜索: {query}"

tool3 = StructuredTool.from_function(
    func=async_search,
    name="async_search",
    description="异步搜索工具"
)

# 方式 4：带错误处理的工具
def safe_divide(a: float, b: float) -> str:
    """安全除法"""
    try:
        if b == 0:
            return "错误：除数不能为 0"
        result = a / b
        return f"{a} / {b} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

class DivideInput(BaseModel):
    a: float = Field(description="被除数")
    b: float = Field(description="除数")

tool4 = StructuredTool.from_function(
    func=safe_divide,
    name="safe_divide",
    description="安全除法，处理除数为 0 的情况",
    args_schema=DivideInput
)

# 方式 5：工具集（ToolKit）
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine

# 数据库工具集示例
# engine = create_engine("sqlite:///example.db")
# db = SQLDatabase(engine)
# toolkit = SQLDatabaseToolkit(db=db, llm=llm)
# tools = toolkit.get_tools()

# 组合所有工具
all_tools = [tool1, tool2, tool3, tool4]

# 创建 Agent
agent = create_react_agent(llm, all_tools, react_prompt)
executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)
```

#### Prompt 设计

Agent 的 Prompt 设计至关重要，直接影响 Agent 的行为和性能。

```python
# 高质量 Agent Prompt 设计

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. 基础 ReAct Prompt 模板
basic_react_prompt = """你是一个智能助手，可以使用工具来帮助用户完成任务。

可用工具：
{tools}

工具名称和描述：
{tool_names}

使用以下格式：

Question: 用户的问题
Thought: 你应该思考做什么
Action: 要采取的行动，应该是 [{tool_names}] 之一
Action Input: 行动的输入
Observation: 行动的结果
... (这个过程可以重复 N 次)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

开始！

Question: {input}
Thought:{agent_scratchpad}"""

# 2. 带角色设定的 Prompt
role_based_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的数据分析师助手。你的职责是：
1. 帮助用户分析数据
2. 生成统计报告
3. 创建可视化图表

你可以使用以下工具：
{tools}

请用专业但易懂的语言与用户交流。在给出数据结论时，务必说明分析方法和置信度。"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 3. 带示例的 Few-shot Prompt
few_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，可以使用工具完成任务。"),
    ("human", "2024 年奥运会在哪个城市举办？"),
    ("ai", """Thought: 我需要搜索 2024 年奥运会的信息
Action: search
Action Input: 2024 年 奥运会 举办城市"""),
    ("human", "Observation: 2024 年夏季奥运会在法国巴黎举办"),
    ("ai", """Thought: 我已经获得了答案
Final Answer: 2024 年夏季奥运会在法国巴黎举办。"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 4. 带约束的 Prompt
constrained_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个谨慎的助手。请遵循以下规则：

1. **工具使用规则**：
   - 只在必要时使用工具
   - 每次只使用一个工具
   - 确保工具参数正确

2. **安全规则**：
   - 不执行危险操作
   - 不访问敏感数据
   - 不确定时询问用户

3. **输出规则**：
   - 回答要简洁准确
   - 引用数据来源
   - 标注不确定信息

可用工具：{tools}"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 5. 多语言 Prompt
multilingual_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个多语言智能助手。请根据用户的语言进行回复。

规则：
- 用户使用中文，就用中文回复
- 用户使用英文，就用英文回复
- 保持语言一致性

可用工具：{tools}"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 使用自定义 Prompt 创建 Agent
custom_agent = create_react_agent(
    llm=llm,
    tools=all_tools,
    prompt=role_based_prompt
)
custom_executor = AgentExecutor(
    agent=custom_agent,
    tools=all_tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    max_execution_time=60  # 最大执行时间（秒）
)
```

**Prompt 设计最佳实践**：

| 原则 | 说明 | 示例 |
|------|------|------|
| **明确角色** | 设定清晰的角色定位 | "你是专业客服助手" |
| **详细规则** | 列出具体的行为约束 | "不要执行删除操作" |
| **工具描述** | 工具描述要包含使用场景 | "当需要查询天气时使用" |
| **示例引导** | 提供 Few-shot 示例 | 展示正确的使用流程 |
| **输出格式** | 指定输出格式要求 | "以 JSON 格式输出" |
| **错误处理** | 说明遇到错误怎么办 | "失败时重试一次" |

#### 执行流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant E as AgentExecutor
    participant A as Agent
    participant L as LLM
    participant T as Tools
    
    U->>E: invoke(input)
    E->>A: 创建 AgentStep
    A->>L: 发送 Prompt
    L-->>A: 返回决策 (Thought/Action)
    
    alt 需要调用工具
        A->>E: 返回 Action
        E->>T: 执行工具
        T-->>E: 返回 Observation
        E->>A: 传递 Observation
        A->>L: 继续推理
        L-->>A: 返回下一步决策
    else 得出最终答案
        A->>E: 返回 Final Answer
        E-->>U: 返回结果
    end
```

```python
# 执行流程详细追踪
from langchain.callbacks import StdOutCallbackHandler
from langchain_core.tracers import LogCallbackHandler

# 1. 基础执行（带控制台输出）
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 打印执行过程
    callbacks=[StdOutCallbackHandler()]
)

# 2. 详细日志记录
log_handler = LogCallbackHandler()
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    callbacks=[log_handler]
)

result = executor.invoke({"input": "计算 2^10 并搜索这个结果"})

# 查看执行轨迹
for step in log_handler.traces:
    print(f"步骤: {step['name']}")
    print(f"输入: {step.get('input')}")
    print(f"输出: {step.get('output')}")
    print("---")

# 3. 自定义回调（监控执行）
from langchain_core.callbacks import BaseCallbackHandler

class ExecutionMonitor(BaseCallbackHandler):
    """执行监控器"""
    
    def __init__(self):
        self.tool_calls = []
        self.llm_calls = []
        self.errors = []
        self.start_time = None
        self.end_time = None
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        self.start_time = datetime.now()
        print(f"开始执行: {serialized.get('name')}")
    
    def on_chain_end(self, outputs, **kwargs):
        self.end_time = datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        print(f"执行完成，耗时: {elapsed:.2f}秒")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"调用工具: {serialized['name']}")
        self.tool_calls.append({
            "tool": serialized["name"],
            "input": input_str,
            "timestamp": datetime.now().isoformat()
        })
    
    def on_tool_end(self, output, **kwargs):
        print(f"工具返回: {output[:100]}")
        self.tool_calls[-1]["output"] = output
    
    def on_tool_error(self, error, **kwargs):
        print(f"工具错误: {error}")
        self.errors.append({
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        })
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.llm_calls.append({
            "prompt_length": len(prompts[0]),
            "timestamp": datetime.now().isoformat()
        })
    
    def on_llm_end(self, response, **kwargs):
        self.llm_calls[-1]["output_length"] = len(response.generations[0][0].text)
    
    def get_report(self) -> Dict[str, Any]:
        """生成执行报告"""
        return {
            "tool_calls": len(self.tool_calls),
            "llm_calls": len(self.llm_calls),
            "errors": len(self.errors),
            "elapsed_time": (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            "tool_calls_details": self.tool_calls,
            "errors_details": self.errors
        }

# 使用监控器
monitor = ExecutionMonitor()
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    callbacks=[monitor]
)

result = executor.invoke({"input": "复杂任务"})
report = monitor.get_report()
print(f"工具调用次数: {report['tool_calls']}")
print(f"LLM 调用次数: {report['llm_calls']}")
print(f"错误数量: {report['errors']}")
print(f"总耗时: {report['elapsed_time']:.2f}秒")
```

---

## 4. LangGraph 单 Agent 状态图

LangGraph 是 LangChain 团队推出的状态图框架，专门用于构建复杂的 Agent 工作流。相比传统的 AgentExecutor，LangGraph 提供了更强大的状态管理、条件路由和人类在环能力。

### 4.1 状态图基础

#### 核心概念

```mermaid
graph TB
    A[State 状态] --> B[Node 节点]
    B --> C[Edge 边]
    C --> D[条件路由]
    
    A1[定义数据结构] -.-> A
    A2[维护执行上下文] -.-> A
    
    B1[处理函数] -.-> B
    B2[修改状态] -.-> B
    
    C1[连接节点] -.-> C
    C2[控制流程] -.-> C
    
    D1[条件判断] -.-> D
    D2[动态路由] -.-> D
    
    style A fill:#cce5ff
    style B fill:#d4edda
    style C fill:#fff3cd
    style D fill:#f8d7da
```

**LangGraph 四大核心组件**：

| 组件 | 说明 | 类比 | 示例 |
|------|------|------|------|
| **State** | 状态数据结构 | 内存 | `{"messages": [], "iteration": 0}` |
| **Node** | 处理函数 | 处理器 | LLM 调用、工具执行 |
| **Edge** | 节点连接 | 连线 | A → B |
| **Conditional Edge** | 条件路由 | 分支 | if-else |

#### State 定义

```python
# State 定义的多种方式

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# 方式 1：使用 TypedDict + add_messages（推荐）
class AgentState(TypedDict):
    """Agent 状态"""
    # messages 会自动合并新消息
    messages: Annotated[List[BaseMessage], add_messages]
    # 自定义字段
    iteration: int
    max_iterations: int
    current_tool: str
    tool_input: str
    observation: str

# 方式 2：完全自定义的状态管理
class CustomAgentState(TypedDict):
    """自定义 Agent 状态"""
    messages: Annotated[List[BaseMessage], add_messages]
    # 任务规划
    plan: List[str]
    current_step: int
    # 工具相关
    tool_results: List[dict]
    # 用户信息
    user_profile: dict
    # 自定义 reducer
    # 默认是覆盖，可以自定义为累加、合并等
    logs: Annotated[List[str], lambda x, y: x + y]

# 方式 3：带验证的状态
from pydantic import BaseModel, Field

class ValidatedAgentState(BaseModel):
    """带验证的 Agent 状态"""
    messages: List[BaseMessage] = Field(default_factory=list)
    iteration: int = Field(default=0, ge=0, le=100)
    max_iterations: int = Field(default=10, ge=1, le=50)
    
    class Config:
        extra = "allow"  # 允许额外字段

# 自定义 reducer 示例
def custom_reducer(existing: list, update: list) -> list:
    """自定义状态合并逻辑"""
    # 只保留最新的 50 条日志
    combined = existing + update
    return combined[-50:]

class LogState(TypedDict):
    logs: Annotated[List[str], custom_reducer]
```

#### Node 设计

```python
# Node 节点设计模式

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(model="gpt-5.2", temperature=0)

# 1. 简单节点
def llm_node(state: AgentState) -> dict:
    """LLM 调用节点"""
    messages = state["messages"]
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "iteration": state["iteration"] + 1
    }

# 2. 工具调用节点
def tool_node(state: AgentState) -> dict:
    """工具执行节点"""
    # 从最后一条消息中提取工具调用
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        
        # 执行工具（简化示例）
        result = execute_tool(tool_name, tool_input)
        
        # 返回工具结果
        from langchain_core.messages import ToolMessage
        return {
            "messages": [
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            ],
            "current_tool": tool_name,
            "observation": str(result)
        }
    
    return {"messages": []}

# 3. 规划节点
def planner_node(state: AgentState) -> dict:
    """任务规划节点"""
    user_input = state["messages"][0].content
    
    planning_prompt = f"""
请将以下任务分解为可执行的步骤：

任务：{user_input}

输出 JSON 格式的步骤列表：
```json
{{
    "steps": ["步骤 1", "步骤 2", "步骤 3"]
}}
```
"""
    
    response = llm.invoke([HumanMessage(content=planning_prompt)])
    
    # 解析计划
    try:
        json_str = response.content.split("```json")[1].split("```")[0]
        plan = json.loads(json_str.strip())
        return {"plan": plan["steps"], "current_step": 0}
    except:
        return {"plan": ["直接执行"], "current_step": 0}

# 4. 反思节点
def reflection_node(state: AgentState) -> dict:
    """自我反思节点"""
    messages = state["messages"]
    
    reflection_prompt = """
请评估以下对话的质量：

{messages}

评估维度：
1. 是否完整回答了用户问题？
2. 是否有事实错误？
3. 是否需要补充信息？

输出 JSON：
```json
{{
    "is_complete": true/false,
    "needs_improvement": true/false,
    "improvement_suggestion": "建议"
}}
```
"""
    
    messages_text = "\n".join([m.content for m in messages])
    response = llm.invoke([HumanMessage(content=reflection_prompt.format(messages=messages_text))])
    
    try:
        json_str = response.content.split("```json")[1].split("```")[0]
        reflection = json.loads(json_str.strip())
        
        if reflection.get("needs_improvement"):
            # 需要改进，添加系统提示
            return {
                "messages": [
                    SystemMessage(content=f"请改进你的回答：{reflection['improvement_suggestion']}")
                ]
            }
        else:
            # 无需改进，返回空更新
            return {}
    except:
        return {}

# 5. 人类审核节点
def human_review_node(state: AgentState) -> dict:
    """人类审核节点"""
    # 实际应用中，这里会暂停等待人类输入
    # LangGraph 支持 interrupt_before 机制
    
    last_message = state["messages"][-1]
    print(f"等待人类审核：{last_message.content}")
    
    # 模拟人类反馈（实际应从外部输入）
    approved = True
    feedback = "很好，继续"
    
    if approved:
        return {"messages": [SystemMessage(content=f"审核通过：{feedback}")]}
    else:
        return {"messages": [SystemMessage(content=f"审核未通过：{feedback}")]}
```

#### Edge 连接

```python
# Edge 和条件路由

from langgraph.graph import StateGraph, END

# 1. 创建图
graph_builder = StateGraph(AgentState)

# 2. 添加节点
graph_builder.add_node("planner", planner_node)
graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("reflection", reflection_node)

# 3. 添加固定边
graph_builder.add_edge("planner", "llm")  # 规划后调用 LLM
graph_builder.add_edge("tools", "llm")    # 工具执行后返回 LLM

# 4. 添加条件边
def should_use_tools(state: AgentState) -> str:
    """判断是否需要使用工具"""
    last_message = state["messages"][-1]
    
    # 检查 LLM 是否请求了工具调用
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return "end"

# 条件路由
graph_builder.add_conditional_edges(
    "llm",                    # 从 llm 节点出发
    should_use_tools,         # 条件函数
    {                         # 路由映射
        "tools": "tools",     # 如果需要工具，去 tools 节点
        "end": END            # 否则结束
    }
)

# 5. 设置入口点
graph_builder.set_entry_point("planner")

# 6. 编译图
graph = graph_builder.compile()

# 7. 可视化图
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))

# 8. 执行图
def run_agent(user_input: str):
    """执行 Agent"""
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "iteration": 0,
        "max_iterations": 10
    }
    
    # 同步执行
    result = graph.invoke(initial_state)
    
    # 返回最后一条消息
    return result["messages"][-1].content

# 使用示例
# response = run_agent("帮我分析苹果公司 2024 年的财报")
# print(response)
```

#### 条件路由

```python
# 高级条件路由模式

# 1. 多条件路由
def router_node(state: AgentState) -> str:
    """多条件路由"""
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 10)
    messages = state["messages"]
    
    # 检查迭代次数
    if iteration >= max_iterations:
        return "max_iterations_reached"
    
    # 检查最后消息
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tools"
    elif last_message.content.startswith("Final Answer"):
        return "final_answer"
    else:
        return "continue"

# 配置多条件路由
graph_builder.add_conditional_edges(
    "llm",
    router_node,
    {
        "max_iterations_reached": END,
        "use_tools": "tools",
        "final_answer": END,
        "continue": "llm"
    }
)

# 2. 基于意图的路由
from enum import Enum

class IntentType(Enum):
    QUESTION = "question"
    COMMAND = "command"
    CONVERSATION = "conversation"

def intent_router(state: AgentState) -> str:
    """基于意图的路由"""
    user_input = state["messages"][0].content
    
    # 简单分类（实际应使用 LLM 或分类模型）
    if "?" in user_input or any(word in user_input for word in ["什么", "哪里", "如何"]):
        return IntentType.QUESTION.value
    elif any(word in user_input for word in ["创建", "删除", "更新"]):
        return IntentType.COMMAND.value
    else:
        return IntentType.CONVERSATION.value

# 添加意图路由
graph_builder.add_node("question_handler", llm_node)
graph_builder.add_node("command_handler", tool_node)
graph_builder.add_node("conversation_handler", llm_node)

graph_builder.add_conditional_edges(
    "intent_classifier",
    intent_router,
    {
        "question": "question_handler",
        "command": "command_handler",
        "conversation": "conversation_handler"
    }
)

# 3. 动态路由（基于 LLM 决策）
def dynamic_router(state: AgentState) -> str:
    """动态路由：LLM 决定下一步"""
    routing_prompt = """
基于以下对话，决定下一步行动：

{messages}

可选行动：
- use_search: 需要搜索信息
- use_calculator: 需要计算
- use_database: 需要查询数据库
- answer_directly: 直接回答
- ask_user: 需要询问用户更多信息

只输出行动名称：
"""
    
    messages_text = "\n".join([m.content for m in state["messages"]])
    response = llm.invoke([HumanMessage(content=routing_prompt.format(messages=messages_text))])
    
    action = response.content.strip().lower()
    
    action_map = {
        "use_search": "search_node",
        "use_calculator": "calculator_node",
        "use_database": "database_node",
        "answer_directly": END,
        "ask_user": "human_input_node"
    }
    
    return action_map.get(action, END)
```

### 4.2 工具调用图

#### 使用 LangGraph 预构建的 ReAct Agent

```python
# LangGraph 预构建 ReAct Agent

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# 1. 初始化模型
model = ChatOpenAI(model="gpt-5.2", temperature=0)

# 2. 定义工具
tools = [
    Tool(
        name="search",
        func=lambda x: f"搜索结果：{x}",
        description="搜索网络信息"
    ),
    Tool(
        name="calculator",
        func=lambda x: str(eval(x)),
        description="数学计算"
    ),
    Tool(
        name="database_query",
        func=lambda x: f"查询结果：{x}",
        description="查询数据库"
    )
]

# 3. 创建 ReAct Agent 图
graph = create_react_agent(
    model,
    tools=tools,
    state_modifier="You are a helpful assistant"
)

# 4. 执行
inputs = {
    "messages": [("human", "计算 2^16 并搜索这个结果的意义")]
}

result = graph.invoke(inputs)

# 5. 查看结果
for message in result["messages"]:
    print(f"{message.type}: {message.content[:100]}")
```

#### 自定义工具调用图

```python
# 自定义复杂的工具调用图

from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json

class ToolCallingState(TypedDict):
    """工具调用状态"""
    messages: Annotated[List[BaseMessage], add_messages]
    tools_used: Annotated[List[str], lambda x, y: x + y]
    tool_results: Annotated[List[dict], lambda x, y: x + y]
    iteration: int

class AdvancedToolCaller:
    """高级工具调用器"""
    
    def __init__(self, llm, tools: List[Tool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建工具调用图"""
        builder = StateGraph(ToolCallingState)
        
        # 添加节点
        builder.add_node("llm_call", self.llm_call_node)
        builder.add_node("tool_execution", self.tool_execution_node)
        builder.add_node("result_aggregation", self.result_aggregation_node)
        
        # 添加边
        builder.add_edge(START, "llm_call")
        
        # 条件路由
        builder.add_conditional_edges(
            "llm_call",
            self.route_after_llm,
            {
                "execute_tools": "tool_execution",
                "aggregate": "result_aggregation",
                "end": END
            }
        )
        
        builder.add_edge("tool_execution", "llm_call")
        builder.add_edge("result_aggregation", END)
        
        return builder.compile()
    
    def llm_call_node(self, state: ToolCallingState) -> dict:
        """LLM 调用节点"""
        messages = state["messages"]
        response = self.llm.invoke(messages)
        
        return {
            "messages": [response],
            "iteration": state.get("iteration", 0) + 1
        }
    
    def tool_execution_node(self, state: ToolCallingState) -> dict:
        """工具执行节点"""
        last_message = state["messages"][-1]
        
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {"messages": []}
        
        tool_messages = []
        tools_used = []
        tool_results = []
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]
            
            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name].run(tool_input=tool_input)
                    tool_messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"]
                        )
                    )
                    tools_used.append(tool_name)
                    tool_results.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "output": result
                    })
                except Exception as e:
                    tool_messages.append(
                        ToolMessage(
                            content=f"工具执行错误: {str(e)}",
                            tool_call_id=tool_call["id"]
                        )
                    )
        
        return {
            "messages": tool_messages,
            "tools_used": tools_used,
            "tool_results": tool_results
        }
    
    def result_aggregation_node(self, state: ToolCallingState) -> dict:
        """结果聚合节点"""
        # 生成总结
        summary_prompt = """
基于以下工具执行结果，生成最终回答：

工具使用记录：
{tool_results}

用户原始问题：
{user_question}

请综合所有工具结果，给出完整、准确的回答。
"""
        
        user_question = state["messages"][0].content
        tool_results_text = json.dumps(state.get("tool_results", []), indent=2, ensure_ascii=False)
        
        summary = self.llm.invoke([
            HumanMessage(content=summary_prompt.format(
                tool_results=tool_results_text,
                user_question=user_question
            ))
        ])
        
        return {"messages": [summary]}
    
    def route_after_llm(self, state: ToolCallingState) -> str:
        """LLM 后的路由决策"""
        iteration = state.get("iteration", 0)
        max_iterations = 10
        
        # 检查迭代次数
        if iteration >= max_iterations:
            return "end"
        
        last_message = state["messages"][-1]
        
        # 检查是否有工具调用
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "execute_tools"
        
        # 检查是否已完成
        if "final answer" in last_message.content.lower() or not hasattr(last_message, "tool_calls"):
            return "aggregate"
        
        return "end"
    
    def run(self, user_input: str) -> dict:
        """运行工具调用图"""
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "tools_used": [],
            "tool_results": [],
            "iteration": 0
        }
        
        return self.graph.invoke(initial_state)

# 使用示例
# caller = AdvancedToolCaller(llm, tools)
# result = caller.run("分析 2024 年 AI 发展趋势")
```

### 4.3 人类在环 (Human-in-the-Loop)

人类在环是 LangGraph 的强大特性，允许在 Agent 执行过程中插入人类审核和干预。

```python
# 人类在环完整实现

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

# 1. 基础中断机制
def human_approval_node(state: AgentState) -> dict:
    """人类审核节点"""
    last_message = state["messages"][-1]
    
    # 使用 interrupt 暂停执行，等待人类输入
    human_feedback = interrupt({
        "action": "approval",
        "message": last_message.content,
        "options": ["approve", "reject", "modify"]
    })
    
    if human_feedback["action"] == "approve":
        return {"messages": [SystemMessage(content="人类审核通过")]}
    elif human_feedback["action"] == "reject":
        return {"messages": [SystemMessage(content="人类审核未通过，请重新生成")]}
    elif human_feedback["action"] == "modify":
        return {"messages": [
            SystemMessage(content=f"人类修改为：{human_feedback['modified_content']}")
        ]}

# 2. 构建带人类在环的图
def build_human_in_loop_graph(llm, tools):
    """构建人类在环图"""
    builder = StateGraph(AgentState)
    
    # 添加节点
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)
    builder.add_node("human_approval", human_approval_node)
    
    # 添加边
    builder.add_edge(START, "llm")
    
    builder.add_conditional_edges(
        "llm",
        lambda state: "tools" if state["messages"][-1].tool_calls else "human_approval",
        {
            "tools": "tools",
            "human_approval": "human_approval"
        }
    )
    
    builder.add_edge("tools", "human_approval")
    builder.add_edge("human_approval", END)
    
    # 使用检查点保存状态
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    
    return graph

# 3. 执行与恢复
from uuid import uuid4

def execute_with_human_in_loop():
    """执行带人类在环的 Agent"""
    graph = build_human_in_loop_graph(llm, tools)
    
    # 生成唯一的线程 ID
    thread_id = str(uuid4())
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # 初始输入
    inputs = {
        "messages": [HumanMessage(content="生成一份关于 AI 的报告")],
        "iteration": 0
    }
    
    # 第一次执行（会在 human_approval 节点暂停）
    try:
        result = graph.invoke(inputs, config=config)
    except Exception as e:
        if "interrupt" in str(e).lower():
            print("执行暂停，等待人类输入")
            
            # 查看当前状态
            state = graph.get_state(config)
            print(f"当前状态：{state}")
            
            # 人类反馈
            feedback = {
                "action": "approve",
                "comment": "报告质量很好"
            }
            
            # 继续执行
            result = graph.invoke(None, config=config)
            print(f"最终结果：{result}")

# 4. 状态修改（人类修改 Agent 状态）
def modify_agent_state():
    """人类修改 Agent 状态"""
    graph = build_human_in_loop_graph(llm, tools)
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # 执行到中断点
    try:
        graph.invoke({
            "messages": [HumanMessage(content="生成报告")],
            "iteration": 0
        }, config=config)
    except:
        pass
    
    # 获取当前状态
    state = graph.get_state(config)
    print(f"当前状态：{state}")
    
    # 修改状态（例如：修改消息）
    modified_messages = state.values["messages"] + [
        HumanMessage(content="请增加更多关于深度学习的部分")
    ]
    
    # 更新状态
    graph.update_state(config, {"messages": modified_messages})
    
    # 继续执行
    result = graph.invoke(None, config=config)
    return result

# 5. 实际 Web 应用集成示例（FastAPI）
from fastapi import FastAPI
from langgraph.types import Command

app = FastAPI()
graphs = {}  # 存储图实例

@app.post("/agent/start")
async def start_agent(task: str):
    """启动 Agent"""
    thread_id = str(uuid4())
    graph = build_human_in_loop_graph(llm, tools)
    graphs[thread_id] = graph
    
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        result = graph.invoke({
            "messages": [HumanMessage(content=task)],
            "iteration": 0
        }, config=config)
        return {"status": "completed", "result": result}
    except Exception as e:
        if "interrupt" in str(e).lower():
            return {
                "status": "waiting_for_approval",
                "thread_id": thread_id,
                "message": "等待人类审核"
            }

@app.post("/agent/{thread_id}/approve")
async def approve_action(thread_id: str, approved: bool, feedback: str = ""):
    """人类审核"""
    graph = graphs.get(thread_id)
    if not graph:
        return {"error": "Thread not found"}
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # 传递人类反馈
    command = Command(
        resume={
            "action": "approve" if approved else "reject",
            "comment": feedback
        }
    )
    
    result = graph.invoke(command, config=config)
    return {"status": "completed", "result": result}

@app.get("/agent/{thread_id}/state")
async def get_agent_state(thread_id: str):
    """获取 Agent 状态"""
    graph = graphs.get(thread_id)
    if not graph:
        return {"error": "Thread not found"}
    
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    
    return {
        "messages": [m.dict() for m in state.values["messages"]],
        "iteration": state.values.get("iteration", 0)
    }
```

### 4.4 检查点与持久化

LangGraph 的检查点系统支持状态保存、恢复和时间旅行调试。

```python
# 检查点与持久化完整指南

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
import sqlite3

# 1. 内存检查点（开发调试用）
memory_saver = MemorySaver()

# 2. SQLite 检查点（轻量级持久化）
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
sqlite_saver = SqliteSaver(conn)

# 3. PostgreSQL 检查点（生产环境）
# from langgraph.checkpoint.postgres import PostgresSaver
# postgres_saver = PostgresSaver.from_conn_string(
#     "postgresql://user:password@localhost:5432/dbname"
# )

# 4. 使用检查点创建图
def create_graph_with_checkpointer(saver):
    """创建带检查点的图"""
    builder = StateGraph(AgentState)
    
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)
    
    builder.add_edge(START, "llm")
    builder.add_conditional_edges(
        "llm",
        lambda state: "tools" if state["messages"][-1].tool_calls else END,
        {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "llm")
    
    # 编译时传入检查点
    graph = builder.compile(checkpointer=saver)
    
    return graph

# 5. 状态保存与恢复
def checkpoint_example():
    """检查点使用示例"""
    graph = create_graph_with_checkpointer(sqlite_saver)
    
    thread_id = "conversation_001"
    config = {"configurable": {"thread_id": thread_id}}
    
    # 第一次对话
    result1 = graph.invoke({
        "messages": [HumanMessage(content="你好，我想学习 Python")],
        "iteration": 0
    }, config=config)
    
    # 第二次对话（自动加载历史）
    result2 = graph.invoke({
        "messages": [HumanMessage(content="继续教我函数定义")]
    }, config=config)
    
    # 查看检查点历史
    checkpoints = list(graph.checkpointer.list(config))
    print(f"检查点数量：{len(checkpoints)}")
    
    # 恢复到特定检查点
    checkpoint_id = checkpoints[0]["checkpoint_id"]
    graph.update_state(config, {}, checkpoint_id=checkpoint_id)

# 6. 时间旅行调试
def time_travel_debugging():
    """时间旅行调试"""
    graph = create_graph_with_checkpointer(sqlite_saver)
    thread_id = "debug_001"
    config = {"configurable": {"thread_id": thread_id}}
    
    # 执行对话
    try:
        graph.invoke({
            "messages": [HumanMessage(content="复杂任务")],
            "iteration": 0
        }, config=config)
    except Exception as e:
        print(f"出现错误：{e}")
        
        # 查看所有检查点
        checkpoints = list(graph.checkpointer.list(config))
        
        # 回溯到之前的状态
        for checkpoint in checkpoints:
            print(f"\n=== 检查点 {checkpoint['checkpoint_id']} ===")
            state = graph.get_state(config, checkpoint["checkpoint_id"])
            
            print(f"迭代次数：{state.values.get('iteration')}")
            print(f"消息数量：{len(state.values.get('messages', []))}")
            
            # 尝试从该检查点重新执行
            try:
                result = graph.invoke(None, config=config)
                print("重新执行成功！")
                break
            except:
                print("重新执行失败，继续回溯...")

# 7. 自定义检查点序列化
import json
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple

class JSONFileSaver(BaseCheckpointSaver):
    """JSON 文件检查点保存器"""
    
    def __init__(self, directory: str = "./checkpoints"):
        import os
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
    
    def _get_path(self, config: dict) -> str:
        thread_id = config["configurable"]["thread_id"]
        return f"{self.directory}/{thread_id}.json"
    
    def put(self, config: dict, checkpoint: dict, metadata: dict) -> dict:
        """保存检查点"""
        path = self._get_path(config)
        
        # 读取现有检查点
        checkpoints = []
        import os
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                checkpoints = json.load(f)
        
        # 添加新检查点
        checkpoints.append({
            "checkpoint": checkpoint,
            "metadata": metadata,
            "timestamp": self._get_timestamp()
        })
        
        # 保存
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoints, f, ensure_ascii=False, indent=2)
        
        return {
            "configurable": {
                **config["configurable"],
                "checkpoint_id": metadata.get("checkpoint_id", str(len(checkpoints)))
            }
        }
    
    def get_tuple(self, config: dict) -> CheckpointTuple:
        """获取检查点"""
        path = self._get_path(config)
        
        import os
        if not os.path.exists(path):
            return None
        
        with open(path, "r", encoding="utf-8") as f:
            checkpoints = json.load(f)
        
        checkpoint_id = config["configurable"].get("checkpoint_id")
        if checkpoint_id:
            checkpoint = checkpoints[int(checkpoint_id)]
        else:
            checkpoint = checkpoints[-1]  # 最新的
        
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint["checkpoint"],
            metadata=checkpoint["metadata"]
        )
    
    def list(self, config: dict, **kwargs) -> list:
        """列出检查点"""
        path = self._get_path(config)
        
        import os
        if not os.path.exists(path):
            return []
        
        with open(path, "r", encoding="utf-8") as f:
            checkpoints = json.load(f)
        
        return [
            {
                "checkpoint_id": str(i),
                "metadata": cp["metadata"],
                "timestamp": cp["timestamp"]
            }
            for i, cp in enumerate(checkpoints)
        ]
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()

# 8. 生产环境检查点最佳实践
class ProductionCheckpointManager:
    """生产环境检查点管理器"""
    
    def __init__(self, saver, max_checkpoints: int = 100, cleanup_interval: int = 1000):
        self.saver = saver
        self.max_checkpoints = max_checkpoints
        self.cleanup_interval = cleanup_interval
        self.execution_count = 0
    
    def save_with_cleanup(self, config: dict, checkpoint: dict, metadata: dict):
        """保存并清理旧检查点"""
        self.saver.put(config, checkpoint, metadata)
        
        self.execution_count += 1
        if self.execution_count % self.cleanup_interval == 0:
            self.cleanup_old_checkpoints(config)
    
    def cleanup_old_checkpoints(self, config: dict):
        """清理旧检查点"""
        checkpoints = list(self.saver.list(config))
        
        if len(checkpoints) > self.max_checkpoints:
            # 删除最旧的检查点
            checkpoints_to_delete = checkpoints[:-self.max_checkpoints]
            for cp in checkpoints_to_delete:
                # 删除逻辑（取决于具体实现）
                pass
    
    def get_state_history(self, config: dict, limit: int = 10) -> list:
        """获取状态历史"""
        checkpoints = list(self.saver.list(config, limit=limit))
        return checkpoints

# 使用生产级检查点
production_saver = ProductionCheckpointManager(sqlite_saver)
```

---

## 5. 规划与推理

> 📌 **说明**: 本章节内容来自原"Agent设计模式"文档的规划与推理部分，已整合至本文件作为架构模式的补充。

## 3. 规划与推理

### 3.1 任务规划

#### 目标分解

```python
# 智能任务分解

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
import json

llm = ChatOpenAI(model="gpt-5.2", temperature=0)

class SubTask(BaseModel):
    """子任务"""
    id: int = Field(description="任务 ID")
    description: str = Field(description="任务描述")
    required_tools: List[str] = Field(description="需要的工具")
    estimated_complexity: int = Field(description="预估复杂度 1-5", ge=1, le=5)
    dependencies: List[int] = Field(description="依赖的任务 ID", default_factory=list)
    success_criteria: str = Field(description="成功标准")

class TaskPlan(BaseModel):
    """任务计划"""
    goal: str = Field(description="目标")
    subtasks: List[SubTask] = Field(description="子任务列表")
    execution_order: List[int] = Field(description="执行顺序")
    estimated_total_time: str = Field(description="预估总时间")
    risks: List[str] = Field(description="潜在风险", default_factory=list)

# 任务分解 Prompt
task_decomposition_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的任务规划专家。请将复杂目标分解为可执行的子任务。

分解原则：
1. **MECE 原则**：相互独立，完全穷尽
2. **原子性**：每个子任务应该是不可再分的原子操作
3. **可验证性**：每个子任务应该有明确的完成标准
4. **可并行性**：识别可以并行执行的任务
5. **依赖关系**：明确任务间的依赖

输出格式为 JSON，包含：
- goal: 目标描述
- subtasks: 子任务列表
- execution_order: 执行顺序
- estimated_total_time: 预估总时间
- risks: 潜在风险"""),
    ("human", "目标：{goal}\n\n可用工具：{tools}")
])

# 使用结构化输出
decomposer = task_decomposition_prompt | llm.with_structured_output(TaskPlan)

# 使用示例
def decompose_task(goal: str, available_tools: List[str]) -> TaskPlan:
    """分解任务"""
    tools_text = "\n".join([f"- {tool}" for tool in available_tools])
    
    plan = decomposer.invoke({
        "goal": goal,
        "tools": tools_text
    })
    
    return plan

# 执行分解
goal = "分析公司 2024 年销售数据，生成可视化报告并提出改进建议"
tools = ["database_query", "data_analysis", "chart_generation", "report_writing"]

plan = decompose_task(goal, tools)

print(f"目标：{plan.goal}")
print(f"\n子任务：")
for task in plan.subtasks:
    print(f"  {task.id}. {task.description}")
    print(f"     工具：{', '.join(task.required_tools)}")
    print(f"     依赖：{task.dependencies}")
    print(f"     复杂度：{task.estimated_complexity}")
    print(f"     成功标准：{task.success_criteria}")

print(f"\n执行顺序：{plan.execution_order}")
print(f"预估时间：{plan.estimated_total_time}")
print(f"风险：{plan.risks}")
```

#### 步骤生成

```python
# 动态步骤生成

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class StepGenerator:
    """步骤生成器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_steps(
        self,
        task: str,
        completed_steps: List[dict] = None,
        context: dict = None
    ) -> dict:
        """生成下一步"""
        prompt = f"""
当前任务：{task}

已完成步骤：
{self._format_steps(completed_steps)}

上下文信息：
{json.dumps(context or {}, ensure_ascii=False, indent=2)}

请决定：
1. 下一步应该做什么？
2. 需要使用什么工具？
3. 工具的参数是什么？
4. 预期的结果是什么？

输出 JSON：
```json
{{
    "next_step": "步骤描述",
    "tool": "工具名称",
    "tool_input": {{}},
    "expected_output": "预期结果",
    "reason": "选择此步骤的原因",
    "is_final_step": true/false
}}
```
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except:
            return {
                "next_step": "完成任务",
                "tool": None,
                "is_final_step": True
            }
    
    def _format_steps(self, steps: List[dict]) -> str:
        """格式化已完成的步骤"""
        if not steps:
            return "无"
        
        return "\n".join([
            f"{i+1}. {step.get('description', '未知')} - {'成功' if step.get('success') else '失败'}"
            for i, step in enumerate(steps)
        ])

# 使用示例
generator = StepGenerator(llm)

task = "查询北京天气并生成报告"
completed_steps = []
context = {"user_location": "北京"}

# 动态生成步骤
for i in range(10):
    step = generator.generate_steps(task, completed_steps, context)
    
    print(f"\n步骤 {i+1}: {step['next_step']}")
    print(f"工具：{step.get('tool')}")
    print(f"原因：{step.get('reason')}")
    
    # 模拟执行
    completed_steps.append({
        "description": step["next_step"],
        "success": True
    })
    
    if step.get("is_final_step"):
        print("\n任务完成！")
        break
```

#### 资源评估

```python
# 资源评估与成本估算

class ResourceEstimator:
    """资源评估器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def estimate_resources(self, task_plan: TaskPlan) -> dict:
        """评估资源需求"""
        prompt = f"""
请评估以下任务计划所需的资源：

目标：{task_plan.goal}
子任务数量：{len(task_plan.subtasks)}
总预估时间：{task_plan.estimated_total_time}

子任务详情：
{json.dumps([t.dict() for t in task_plan.subtasks], ensure_ascii=False, indent=2)}

请评估：
1. **LLM 调用次数**：预估需要调用多少次 LLM
2. **Token 消耗**：预估总 token 数量
3. **工具调用次数**：各工具调用次数
4. **执行时间**：实际执行时间（考虑并行）
5. **成本估算**：按 $0.01/1K tokens 计算

输出 JSON：
```json
{{
    "llm_calls": 10,
    "estimated_tokens": 50000,
    "tool_calls": {{
        "tool1": 5,
        "tool2": 3
    }},
    "estimated_execution_minutes": 15,
    "estimated_cost_usd": 0.50,
    "optimization_suggestions": ["优化建议"]
}}
```
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except:
            return {"error": "评估失败"}

# 使用示例
estimator = ResourceEstimator(llm)
resources = estimator.estimate_resources(plan)

print(f"LLM 调用次数：{resources['llm_calls']}")
print(f"预估 Token 消耗：{resources['estimated_tokens']}")
print(f"预估执行时间：{resources['estimated_execution_minutes']} 分钟")
print(f"预估成本：${resources['estimated_cost_usd']}")
print(f"优化建议：{resources['optimization_suggestions']}")
```

#### 风险识别

```python
# 风险识别与应对策略

class RiskAnalyzer:
    """风险分析器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_risks(self, task_plan: TaskPlan) -> List[dict]:
        """分析风险"""
        prompt = f"""
请分析以下任务计划的潜在风险：

{json.dumps(task_plan.dict(), ensure_ascii=False, indent=2)}

对每个风险，请评估：
1. 风险描述
2. 发生概率（低/中/高）
3. 影响程度（低/中/高）
4. 应对策略
5. 检测信号（如何识别风险已发生）

输出 JSON 数组：
```json
[
    {{
        "risk": "风险描述",
        "probability": "低/中/高",
        "impact": "低/中/高",
        "mitigation": "应对策略",
        "warning_signs": ["检测信号"]
    }}
]
```
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except:
            return []

# 使用示例
analyzer = RiskAnalyzer(llm)
risks = analyzer.analyze_risks(plan)

print("风险分析：")
for risk in risks:
    print(f"\n风险：{risk['risk']}")
    print(f"概率：{risk['probability']}")
    print(f"影响：{risk['impact']}")
    print(f"应对：{risk['mitigation']}")
    print(f"预警：{', '.join(risk['warning_signs'])}")
```

### 3.2 自我反思

#### 错误识别

```python
# 错误识别与诊断

from langchain_core.prompts import ChatPromptTemplate

error_diagnosis_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个错误诊断专家。请分析以下执行过程中的错误。

任务：{task}
执行步骤：{steps}
错误信息：{error}

请从以下维度分析：
1. **错误类型**：语法错误、逻辑错误、资源错误、超时等
2. **根本原因**：导致错误的根本原因
3. **影响范围**：错误影响了哪些部分
4. **修复建议**：如何修复错误
5. **预防措施**：如何避免类似错误

输出 JSON：
```json
{{
    "error_type": "错误类型",
    "root_cause": "根本原因",
    "impact": "影响范围",
    "fix_suggestion": "修复建议",
    "prevention": "预防措施",
    "severity": "low/medium/high/critical",
    "can_retry": true/false
}}
```"""),
    ("human", "请分析错误")
])

error_diagnoser = error_diagnosis_prompt | llm

class ErrorDiagnoser:
    """错误诊断器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.diagnoser = error_diagnosis_prompt | llm
    
    def diagnose(
        self,
        task: str,
        steps: List[dict],
        error: str
    ) -> dict:
        """诊断错误"""
        steps_text = json.dumps(steps, ensure_ascii=False, indent=2)
        
        response = self.diagnoser.invoke({
            "task": task,
            "steps": steps_text,
            "error": error
        })
        
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except:
            return {
                "error_type": "unknown",
                "root_cause": "诊断失败",
                "can_retry": False
            }
    
    def auto_fix(self, diagnosis: dict, original_step: dict) -> dict:
        """自动生成修复方案"""
        fix_prompt = f"""
基于以下错误诊断，生成修复后的步骤：

原始步骤：{json.dumps(original_step, ensure_ascii=False)}
诊断结果：{json.dumps(diagnosis, ensure_ascii=False)}

请输出修复后的步骤（JSON）：
"""
        
        response = self.llm.invoke([HumanMessage(content=fix_prompt)])
        
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except:
            return original_step

# 使用示例
diagnoser = ErrorDiagnoser(llm)

task = "查询数据库并生成报告"
steps = [
    {"action": "查询数据库", "status": "success"},
    {"action": "分析数据", "status": "failed"}
]
error = "数据库连接超时"

diagnosis = diagnoser.diagnose(task, steps, error)
print(f"错误类型：{diagnosis['error_type']}")
print(f"根本原因：{diagnosis['root_cause']}")
print(f"严重程度：{diagnosis['severity']}")
print(f"是否可重试：{diagnosis['can_retry']}")
print(f"修复建议：{diagnosis['fix_suggestion']}")
```

#### 策略调整

```python
# 策略调整与优化

class StrategyAdjuster:
    """策略调整器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.execution_history: List[dict] = []
    
    def record_execution(self, task: str, plan: dict, result: dict):
        """记录执行"""
        self.execution_history.append({
            "task": task,
            "plan": plan,
            "result": result,
            "timestamp": time.time()
        })
    
    def suggest_improvements(self, task: str) -> dict:
        """基于历史执行建议改进"""
        # 查找类似任务
        similar_tasks = [
            exec for exec in self.execution_history
            if self._similarity(exec["task"], task) > 0.7
        ]
        
        if not similar_tasks:
            return {"suggestions": ["无历史数据可供参考"]}
        
        # 分析失败案例
        failures = [
            exec for exec in similar_tasks
            if not exec["result"].get("success", True)
        ]
        
        if not failures:
            return {"suggestions": ["历史执行都很成功，保持当前策略"]}
        
        # LLM 分析改进建议
        analysis_prompt = f"""
分析以下失败案例，提出改进策略：

失败案例：
{json.dumps(failures[:3], ensure_ascii=False, indent=2)}

请提出：
1. 策略调整建议
2. 工具选择优化
3. 错误处理改进
4. 资源分配优化

输出 JSON：
"""
        
        response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
        
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except:
            return {"suggestions": ["分析失败"]}
    
    def _similarity(self, task1: str, task2: str) -> float:
        """计算任务相似度（简化版）"""
        words1 = set(task1.lower().split())
        words2 = set(task2.lower().split())
        
        if not words1 or not words2:
            return 0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)

# 使用示例
adjuster = StrategyAdjuster(llm)

# 记录执行历史
adjuster.record_execution(
    "查询数据",
    {"tools": ["database"]},
    {"success": False, "error": "超时"}
)

# 获取改进建议
suggestions = adjuster.suggest_improvements("查询销售数据")
print(f"改进建议：{suggestions}")
```

#### 经验积累

```python
# 经验积累与知识库

import json
from pathlib import Path

class ExperienceBase:
    """经验知识库"""
    
    def __init__(self, persist_file: str = "./experiences.json"):
        self.persist_file = persist_file
        self.experiences: List[dict] = []
        self._load()
    
    def add_experience(
        self,
        task: str,
        strategy: dict,
        outcome: dict,
        lessons: List[str]
    ):
        """添加经验"""
        experience = {
            "task": task,
            "strategy": strategy,
            "outcome": outcome,
            "lessons": lessons,
            "timestamp": time.time(),
            "success_count": 1 if outcome.get("success") else 0,
            "failure_count": 0 if outcome.get("success") else 1
        }
        
        self.experiences.append(experience)
        self._save()
    
    def query_experiences(self, task: str, limit: int = 5) -> List[dict]:
        """查询相关经验"""
        # 简化：关键词匹配
        keywords = set(task.lower().split())
        
        scored = []
        for exp in self.experiences:
            exp_keywords = set(exp["task"].lower().split())
            similarity = len(keywords & exp_keywords) / len(keywords | exp_keywords)
            scored.append((similarity, exp))
        
        # 按相似度排序
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [exp for _, exp in scored[:limit]]
    
    def get_best_practices(self) -> List[dict]:
        """获取最佳实践"""
        # 筛选成功率高的经验
        high_success = [
            exp for exp in self.experiences
            if exp["success_count"] > 0 and exp["failure_count"] == 0
        ]
        
        return high_success
    
    def _save(self):
        """保存"""
        with open(self.persist_file, "w", encoding="utf-8") as f:
            json.dump(self.experiences, f, ensure_ascii=False, indent=2)
    
    def _load(self):
        """加载"""
        if Path(self.persist_file).exists():
            with open(self.persist_file, "r", encoding="utf-8") as f:
                self.experiences = json.load(f)

# 使用示例
experience_db = ExperienceBase()

# 添加经验
experience_db.add_experience(
    task="数据库查询优化",
    strategy={"use_cache": True, "batch_size": 100},
    outcome={"success": True, "execution_time": 5.2},
    lessons=["使用缓存可以显著提升性能", "批量大小 100 是最佳平衡点"]
)

# 查询经验
experiences = experience_db.query_experiences("数据库查询")
for exp in experiences:
    print(f"任务：{exp['task']}")
    print(f"经验：{', '.join(exp['lessons'])}")
    print("---")
```

### 3.3 思维链应用

#### Zero-shot CoT

```python
# Zero-shot Chain-of-Thought

from langchain_core.prompts import ChatPromptTemplate

zero_shot_cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个逻辑推理专家。请一步步思考，展示你的推理过程。"),
    ("human", "{question}\n\n请一步步思考：")
])

zero_shot_cot = zero_shot_cot_prompt | llm

# 使用示例
question = "如果 3 个人 3 天可以完成 3 个项目，那么 9 个人 9 天可以完成多少个项目？"

response = zero_shot_cot.invoke({"question": question})
print(response.content)
# 输出会包含详细的推理步骤
```

#### Few-shot CoT

```python
# Few-shot Chain-of-Thought

few_shot_cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个逻辑推理专家。请参考示例，一步步推理。"),
    
    # 示例 1
    ("human", "问题：一个农场有 100 只鸡，每天增加 5 只，30 天后有多少只？"),
    ("ai", """让我一步步思考：

1. 初始数量：100 只
2. 每天增加：5 只
3. 天数：30 天
4. 总增加：5 × 30 = 150 只
5. 最终数量：100 + 150 = 250 只

答案：250 只"""),
    
    # 示例 2
    ("human", "问题：列车以 60km/h 速度行驶，2.5 小时行驶多远？"),
    ("ai", """让我一步步思考：

1. 速度：60 km/h
2. 时间：2.5 小时
3. 距离 = 速度 × 时间
4. 距离 = 60 × 2.5 = 150 km

答案：150 公里"""),
    
    # 用户问题
    ("human", "问题：{question}\n\n请一步步思考：")
])

few_shot_cot = few_shot_cot_prompt | llm

# 使用示例
question = "如果 5 台机器 5 分钟可以生产 5 个零件，100 台机器生产 100 个零件需要多长时间？"
response = few_shot_cot.invoke({"question": question})
print(response.content)
```

#### Self-Consistency

```python
# Self-Consistency：多次采样取一致

from typing import List
from collections import Counter

class SelfConsistencyCoT:
    """Self-Consistency CoT"""
    
    def __init__(self, llm, num_samples: int = 5):
        self.llm = llm
        self.num_samples = num_samples
        self.cot_chain = few_shot_cot_prompt | llm
    
    def reason(self, question: str) -> dict:
        """推理"""
        # 多次采样
        responses = []
        for i in range(self.num_samples):
            response = self.cot_chain.invoke({"question": question})
            responses.append(response.content)
        
        # 提取答案（简化：假设答案在最后）
        answers = []
        for resp in responses:
            lines = resp.strip().split("\n")
            for line in reversed(lines):
                if "答案" in line or "answer" in line.lower():
                    answers.append(line)
                    break
        
        # 投票
        if answers:
            counter = Counter(answers)
            most_common = counter.most_common(1)[0]
            
            return {
                "question": question,
                "all_responses": responses,
                "all_answers": answers,
                "final_answer": most_common[0],
                "confidence": most_common[1] / self.num_samples,
                "vote_distribution": dict(counter)
            }
        else:
            return {
                "question": question,
                "all_responses": responses,
                "final_answer": "无法确定答案",
                "confidence": 0
            }

# 使用示例
sc_cot = SelfConsistencyCoT(llm, num_samples=5)

question = "一个水箱有两个水管，A 管单独注水需要 6 小时，B 管单独注水需要 4 小时，两管同时注水需要多长时间？"
result = sc_cot.reason(question)

print(f"问题：{result['question']}")
print(f"最终答案：{result['final_answer']}")
print(f"置信度：{result['confidence']:.2%}")
print(f"投票分布：{result['vote_distribution']}")
```

#### Tree of Thoughts

```python
# Tree of Thoughts（思维树）

from typing import List, Optional
import heapq

class ThoughtNode:
    """思维节点"""
    def __init__(self, thought: str, score: float = 0, depth: int = 0):
        self.thought = thought
        self.score = score
        self.depth = depth
        self.children: List["ThoughtNode"] = []
        self.parent: Optional["ThoughtNode"] = None
    
    def add_child(self, child: "ThoughtNode"):
        child.parent = self
        self.children.append(child)
    
    def __lt__(self, other):
        return self.score < other.score

class TreeOfThoughts:
    """思维树"""
    
    def __init__(self, llm, max_depth: int = 3, branch_factor: int = 3):
        self.llm = llm
        self.max_depth = max_depth
        self.branch_factor = branch_factor
        self.root = None
    
    def solve(self, problem: str) -> dict:
        """解决问题"""
        # 创建根节点
        self.root = ThoughtNode(f"问题：{problem}", score=0, depth=0)
        
        # BFS 搜索
        best_solution = self._search()
        
        return {
            "problem": problem,
            "solution": best_solution.thought if best_solution else "未找到解决方案",
            "solution_score": best_solution.score if best_solution else 0
        }
    
    def _search(self) -> Optional[ThoughtNode]:
        """搜索最佳解决方案"""
        # 优先队列（按分数排序）
        queue = [-self.root]  # 负数实现最大堆
        best_node = None
        best_score = -float("inf")
        
        while queue:
            # 取出最高分节点
            current = -heapq.heappop(queue)
            
            # 更新最佳节点
            if current.score > best_score:
                best_score = current.score
                best_node = current
            
            # 如果达到最大深度，跳过
            if current.depth >= self.max_depth:
                continue
            
            # 生成子节点
            children = self._generate_thoughts(current)
            
            for child in children:
                current.add_child(child)
                heapq.heappush(queue, -child)
        
        return best_node
    
    def _generate_thoughts(self, node: ThoughtNode) -> List[ThoughtNode]:
        """生成下一步思维"""
        prompt = f"""
当前思维：{node.thought}

请生成 {self.branch_factor} 个可能的下一步思维，并为每个思维评分（0-1）。

输出 JSON：
```json
[
    {{
        "thought": "思维内容",
        "score": 0.8
    }}
]
```
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            thoughts = json.loads(json_str.strip())
            
            return [
                ThoughtNode(t["thought"], score=t["score"], depth=node.depth + 1)
                for t in thoughts
            ]
        except:
            return []

# 使用示例
tot = TreeOfThoughts(llm, max_depth=3, branch_factor=3)

problem = "如何优化一个电商网站的加载速度？"
result = tot.solve(problem)

print(f"问题：{result['problem']}")
print(f"最佳解决方案：{result['solution']}")
print(f"评分：{result['solution_score']}")
```

---

