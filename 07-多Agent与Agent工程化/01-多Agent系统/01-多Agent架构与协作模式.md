# 多 Agent 协作系统开发

> 📅 **更新时间**: 2026-06-17

---

## 目录

- [1. 多 Agent 系统概述](#1-多-agent-系统概述)
- [2. CrewAI 实战](#2-crewai-实战)
- [3. AutoGen/AG2 实战](#3-autogenag2-实战)
- [4. OpenAI Agents SDK 实战](#4-openai-agents-sdk-实战)
- [5. Google ADK 实战](#5-google-adk-实战)
- [6. 多 Agent 通信与协调](#6-多-agent-通信与协调)
- [7. Agent 编排模式](#7-agent-编排模式)
- [8. 最佳实践与陷阱](#8-最佳实践与陷阱)
- [9. 参考资料](#9-参考资料)
- [10. 工具调用深度实践](#10-工具调用深度实践)
- [11. Agent 评估与基准测试](#11-agent-评估与基准测试)
- [12. 解决方案](#12-解决方案)
- [13. 修改的文件](#13-修改的文件)
- [14. 测试](#14-测试)

---

## 1. 多 Agent 系统概述

### 1.1 为什么需要多 Agent？

单 Agent 系统面临的局限性：

- **能力瓶颈**：单个 LLM 无法同时精通多个领域
- **上下文限制**：长对话历史占用大量上下文窗口
- **可靠性问题**：单点失败影响整个系统
- **可扩展性差**：难以并行处理复杂任务
- **缺乏制衡**：没有交叉验证机制

多 Agent 系统的优势：

| 优势 | 描述 | 示例 |
|------|------|------|
| **专业分工** | 每个 Agent 专注特定领域 | 研究员、程序员、审核员 |
| **并行处理** | 多个 Agent 同时工作 | 并行搜索、并行验证 |
| **交叉验证** | 多 Agent 投票/共识 | 质量检查、事实核查 |
| **容错能力** | 单个 Agent 失败不影响全局 | fallback 机制 |
| **可扩展性** | 动态添加/移除 Agent | 弹性资源分配 |

### 1.2 单 Agent vs 多 Agent 对比

```python
# 单 Agent 示例
def single_agent_workflow(question):
    """单 Agent 处理复杂问题"""
    # 1. 研究（需要自己切换角色）
    research = llm(f"作为研究员，搜索以下问题的答案：{question}")
    
    # 2. 编程（需要自己写代码）
    code = llm(f"作为程序员，编写代码实现：{research}")
    
    # 3. 审核（需要自己检查）
    review = llm(f"作为审核员，检查以下代码：{code}")
    
    # 问题：角色切换不彻底，容易混淆
    return review

# 多 Agent 示例
def multi_agent_workflow(question):
    """多 Agent 协作处理"""
    # 1. 研究 Agent 专职研究
    research = researcher_agent.run(question)
    
    # 2. 编程 Agent 专职编程
    code = programmer_agent.run(research)
    
    # 3. 审核 Agent 专职审核
    review = reviewer_agent.run(code)
    
    # 优势：专业化、并行、可验证
    return review
```

### 1.3 多 Agent 系统架构模式

```
多 Agent 架构模式
├── 1. 顺序编排（Sequential）
│   Agent A → Agent B → Agent C
│   适用：流水线任务
│
├── 2. 并行编排（Parallel）
│   Agent A ─┐
│   Agent B ─┼→ Merge → Result
│   Agent C ─┘
│   适用：独立子任务
│
├── 3. 投票/仲裁（Voting）
│   Agent A ─┐
│   Agent B ─┼→ Vote → Consensus
│   Agent C ─┘
│   适用：需要共识的决策
│
├── 4. 层级管理（Hierarchical）
│         Manager
│        /   |   \
│   Agent A Agent B Agent C
│   适用：复杂项目管理
│
└── 5. 动态编排（Dynamic）
    Agent A → (条件判断) → Agent B 或 Agent C
    适用：分支流程
```

## 2. CrewAI 实战

### 2.1 框架简介

**CrewAI** 是一个角色驱动的多 Agent 编排框架，核心特点：

- **角色定义**：每个 Agent 有明确的角色、目标、背景
- **任务分配**：任务与角色能力匹配
- **团队协作**：Agent 间可以共享信息和工具
- **灵活编排**：支持顺序、并行、层级编排
- **易用的 API**：Pythonic 接口设计

GitHub: https://github.com/crewAIInc/crewAI

### 2.2 核心概念

#### Agent（智能体）

```python
from crewai import Agent

# 创建研究 Agent
researcher = Agent(
    role='高级研究员',
    goal='深入调查研究问题，提供准确的研究报告',
    backstory='你是一位拥有 10 年经验的研究专家，'
              '擅长从多个信息源收集、分析和综合信息。'
              '你的报告总是准确、全面且易于理解。',
    tools=[search_tool, web_scraping_tool],
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    max_rpm=100
)

# 创建编程 Agent
programmer = Agent(
    role='高级软件工程师',
    goal='基于研究结果编写高质量、可维护的代码',
    backstory='你是一位全栈工程师，'
              '精通 Python、JavaScript 和系统设计。'
              '你编写的代码总是符合最佳实践。',
    tools=[code_execution_tool, file_tools],
    verbose=True
)

# 创建审核 Agent
reviewer = Agent(
    role='技术审核专家',
    goal='审查代码和研究，确保质量和准确性',
    backstory='你是一位技术负责人，'
              '负责审查团队成员的工作。'
              '你擅长发现潜在的 bug 和改进点。',
    tools=[code_review_tool, fact_checking_tool],
    verbose=True
)
```

#### Task（任务）

```python
from crewai import Task

# 研究任务
research_task = Task(
    description='调查 "{topic}" 的最新技术趋势',
    expected_output='一份包含 5-10 个关键发现的研究报告',
    agent=researcher,
    tools=[search_tool],
    async_execution=False,  # 同步执行
    output_json={  # 结构化输出
        "key_findings": [],
        "sources": [],
        "date": "2026-06-10"
    }
)

# 编程任务
coding_task = Task(
    description='基于研究结果实现一个原型系统',
    expected_output='可运行的 Python 代码和文档',
    agent=programmer,
    context=[research_task],  # 依赖研究任务的结果
    tools=[code_execution_tool]
)

# 审核任务
review_task = Task(
    description='审查代码和研究报告的质量',
    expected_output='详细的审查报告和改进建议',
    agent=reviewer,
    context=[research_task, coding_task],
    output_file='output/review.md'
)
```

#### Crew（团队）

```python
from crewai import Crew, Process

# 创建团队
crew = Crew(
    agents=[researcher, programmer, reviewer],
    tasks=[research_task, coding_task, review_task],
    process=Process.sequential,  # 顺序执行
    verbose=True,
    memory=True,  # 启用记忆
    cache=True,   # 启用缓存
    max_rpm=1000  # 速率限制
)

# 执行团队
result = crew.kickoff(inputs={'topic': 'LLM Agent 开发框架'})

print(f"最终结果：{result}")
```

### 2.3 任务编排

#### 顺序编排

```python
# 顺序编排：任务依次执行
sequential_crew = Crew(
    agents=[agent1, agent2, agent3],
    tasks=[task1, task2, task3],
    process=Process.sequential
)
```

#### 并行编排

```python
from crewai import Task

# 创建可并行执行的任务
research_task_1 = Task(
    description='研究主题 A',
    agent=researcher_a,
    async_execution=True  # 异步执行
)

research_task_2 = Task(
    description='研究主题 B',
    agent=researcher_b,
    async_execution=True  # 异步执行
)

# 汇总任务（等待并行任务完成）
summary_task = Task(
    description='综合主题 A 和 B 的研究结果',
    agent=summarizer,
    context=[research_task_1, research_task_2]
)

# 创建并行团队
parallel_crew = Crew(
    agents=[researcher_a, researcher_b, summarizer],
    tasks=[research_task_1, research_task_2, summary_task],
    process=Process.hierarchical  # 层级管理
)
```

### 2.4 工具集成

```python
from crewai_tools import tool

@tool
def search_internet(query: str) -> str:
    """搜索互联网获取最新信息"""
    from duckduckgo_search import DDGS
    
    results = DDGS().text(query, max_results=5)
    return "\n".join([r['body'] for r in results])

@tool
def execute_python_code(code: str) -> str:
    """执行 Python 代码并返回结果"""
    import subprocess
    
    result = subprocess.run(
        ['python', '-c', code],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode == 0:
        return result.stdout
    else:
        return f"错误：{result.stderr}"

@tool
def read_file(file_path: str) -> str:
    """读取文件内容"""
    with open(file_path, 'r') as f:
        return f.read()

# 在 Agent 中使用工具
researcher = Agent(
    role='研究员',
    tools=[search_internet, read_file],
    # ...
)
```

### 2.5 实战案例：市场研究团队

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool

# 定义工具
@tool
def search_market_data(company: str) -> str:
    """搜索公司市场数据"""
    # 实现搜索逻辑
    return f"{company} 的市场数据..."

@tool
def analyze_financials(data: str) -> str:
    """分析财务数据"""
    # 实现分析逻辑
    return f"财务分析结果..."

# 创建 Agent
market_researcher = Agent(
    role='市场研究员',
    goal='收集目标公司的市场数据和竞争情报',
    backstory='你是一位资深的市场分析师，'
              '擅长从公开信息中提取关键洞察。',
    tools=[search_market_data],
    verbose=True
)

financial_analyst = Agent(
    role='财务分析师',
    goal='分析公司的财务数据和估值',
    backstory='你是一位 CFA 持证的财务分析师，'
              '精通财务建模和估值分析。',
    tools=[analyze_financials],
    verbose=True
)

report_writer = Agent(
    role='报告撰写人',
    goal='整合研究和分析结果，撰写专业的投资报告',
    backstory='你是一位金融作家，'
              '擅长将复杂的数据转化为清晰的投资建议。',
    verbose=True
)

# 创建任务
research_task = Task(
    description='调研 {company} 的市场地位、竞争对手和行业趋势',
    expected_output='市场分析报告',
    agent=market_researcher
)

analysis_task = Task(
    description='分析 {company} 的财务数据、盈利能力和发展前景',
    expected_output='财务分析报告',
    agent=financial_analyst,
    context=[research_task]
)

writing_task = Task(
    description='综合市场研究和财务分析，撰写投资建议报告',
    expected_output='完整的投资报告',
    agent=report_writer,
    context=[research_task, analysis_task],
    output_file='output/investment_report.md'
)

# 创建并执行团队
investment_crew = Crew(
    agents=[market_researcher, financial_analyst, report_writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential,
    verbose=True
)

# 执行
result = investment_crew.kickoff(inputs={'company': 'Tesla'})
print(result)
```

## 3. AutoGen/AG2 实战

### 3.1 框架简介

**AutoGen**（现更名为 **AG2**）是微软开发的多 Agent 对话框架，核心特点：

- **对话式架构**：Agent 通过对话协作
- **GroupChat**：支持多 Agent 群组对话
- **代码执行**：内置代码执行沙箱
- **灵活定制**：可自定义 Agent 行为
- **生产级**：支持流式输出、日志记录

GitHub: https://github.com/ag2ai/ag2

### 3.2 对话式 Agent 架构

#### 基础 Agent

```python
from autogen import AssistantAgent, UserProxyAgent

# 配置
config_list = [
    {
        "model": "gpt-5.2",
        "api_key": "your-api-key"
    }
]

# 创建助手 Agent
assistant = AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list,
        "temperature": 0.7
    },
    system_message="你是一个有用的 AI 助手。"
)

# 创建用户代理（可执行代码）
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # 全自动模式
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": True,  # 使用 Docker 沙箱
        "timeout": 60,
    }
)

# 开始对话
user_proxy.initiate_chat(
    assistant,
    message="编写一个 Python 函数计算斐波那契数列"
)
```

### 3.3 GroupChat 与 GroupChatManager

#### GroupChat 示例

```python
from autogen import GroupChat, GroupChatManager

# 创建多个 Agent
researcher = AssistantAgent(
    name="researcher",
    system_message="你是一名研究员。负责搜集信息。",
    llm_config={"config_list": config_list}
)

engineer = AssistantAgent(
    name="engineer",
    system_message="你是一名工程师。负责编写代码。",
    llm_config={"config_list": config_list}
)

reviewer = AssistantAgent(
    name="reviewer",
    system_message="你是一名审核员。负责审查工作。",
    llm_config={"config_list": config_list}
)

# 创建群组
groupchat = GroupChat(
    agents=[user_proxy, researcher, engineer, reviewer],
    messages=[],
    max_round=20,
    speaker_selection_method="auto",  # 自动选择发言人
    allow_repeat_speaker=False  # 不允许连续发言
)

# 创建群组管理器
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list}
)

# 启动群组对话
user_proxy.initiate_chat(
    manager,
    message="开发一个简单的 Web 应用，包括研究、编码和审核"
)
```

### 3.4 嵌套对话与代理类型

#### 嵌套对话

```python
# 创建子群组处理特定任务
def create_coding_subgroup():
    """创建编程子群组"""
    # 子 Agent
    coder = AssistantAgent(
        name="coder",
        system_message="你是专业程序员。",
        llm_config={"config_list": config_list}
    )
    
    debugger = AssistantAgent(
        name="debugger",
        system_message="你是调试专家。",
        llm_config={"config_list": config_list}
    )
    
    # 子群组
    coding_group = GroupChat(
        agents=[coder, debugger],
        messages=[],
        max_round=10
    )
    
    coding_manager = GroupChatManager(
        groupchat=coding_group,
        llm_config={"config_list": config_list}
    )
    
    return coding_manager

# 在主群组中使用子群组
main_group = GroupChat(
    agents=[user_proxy, researcher, create_coding_subgroup(), reviewer],
    messages=[],
    max_round=30
)
```

### 3.5 代码执行与沙箱

```python
from autogen import UserProxyAgent

# 配置代码执行
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "work_dir": "workspace",
        "use_docker": {
            "image": "python:3.11-slim",
            "volumes": ["./workspace:/workspace"]
        },
        "timeout": 120,
        "last_n_messages": 3
    }
)

# 安全执行
# - Docker 隔离
# - 资源限制
# - 超时控制
# - 文件系统只读
```

### 3.6 实战案例：代码开发团队

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

def create_dev_team():
    """创建开发团队"""
    config = {"config_list": [{"model": "gpt-5.2", "api_key": "sk-xxx"}]}
    
    # 产品经理
    pm = AssistantAgent(
        name="product_manager",
        system_message="""你是产品经理。
职责：
1. 理解用户需求
2. 编写产品规格文档
3. 验收最终产品""",
        llm_config=config
    )
    
    # 架构师
    architect = AssistantAgent(
        name="architect",
        system_message="""你是系统架构师。
职责：
1. 设计系统架构
2. 选择技术栈
3. 定义接口规范""",
        llm_config=config
    )
    
    # 开发者
    developer = AssistantAgent(
        name="developer",
        system_message="""你是全栈开发者。
职责：
1. 编写代码
2. 单元测试
3. 修复 bug""",
        llm_config=config
    )
    
    # 测试工程师
    tester = AssistantAgent(
        name="tester",
        system_message="""你是测试工程师。
职责：
1. 编写测试用例
2. 执行测试
3. 报告问题""",
        llm_config=config
    )
    
    # 用户代理
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=30,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": True
        }
    )
    
    # 创建群组
    dev_team = GroupChat(
        agents=[user_proxy, pm, architect, developer, tester],
        messages=[],
        max_round=50,
        speaker_selection_method="round_robin"
    )
    
    manager = GroupChatManager(groupchat=dev_team, llm_config=config)
    
    return user_proxy, manager

# 使用示例
user_proxy, manager = create_dev_team()

user_proxy.initiate_chat(
    manager,
    message="""
开发一个待办事项管理 Web 应用：
- 前端：React + TypeScript
- 后端：FastAPI + SQLite
- 功能：创建、读取、更新、删除任务
- 包含单元测试
"""
)
```

## 4. OpenAI Agents SDK 实战

### 4.1 框架简介

**OpenAI Agents SDK** 是 OpenAI 官方发布的多 Agent 框架，核心特点：

- **极简原语**：Agent、Handoff、Guardrail 三大核心概念
- **Handoff 机制**：Agent 间的优雅交接
- **Guardrails**：内置安全护栏
- **TypeScript + Python**：双语言支持
- **生产就绪**：OpenAI 官方维护

GitHub: https://github.com/openai/openai-agents-python

### 4.2 Agent 定义与工具注册

```python
from agents import Agent, function_tool, Runner

# 定义工具
@function_tool
def search_knowledge_base(query: str) -> str:
    """搜索知识库"""
    # 实现搜索逻辑
    return f"搜索结果：{query}"

@function_tool
def create_ticket(issue: str, priority: str) -> str:
    """创建支持工单"""
    ticket_id = f"TK-{random.randint(1000, 9999)}"
    return f"工单已创建：{ticket_id}"

# 创建 Agent
support_agent = Agent(
    name="Customer Support",
    instructions="你是客户支持专家。帮助用户解决问题。",
    tools=[search_knowledge_base, create_ticket]
)

technical_agent = Agent(
    name="Technical Specialist",
    instructions="你是技术专家。处理复杂技术问题。",
    tools=[search_knowledge_base]
)
```

### 4.3 Handoff（Agent 交接）

```python
from agents import Agent, handoff

# 定义交接
support_agent = Agent(
    name="Support Agent",
    instructions="处理一般客户问题。如果是技术问题，交接给技术专家。",
    handoffs=[handoff(technical_agent)]  # 可以交接给技术 Agent
)

technical_agent = Agent(
    name="Technical Agent",
    instructions="处理复杂技术问题。"
)

# 运行
async def main():
    result = await Runner.run(
        support_agent,
        "我的账户无法登录，已经尝试了重置密码"
    )
    print(result.final_output)
```

### 4.4 Guardrails（安全护栏）

```python
from agents import Agent, GuardrailFunction, InputGuardrail, OutputGuardrail

# 输入护栏
def check_input_safety(input_data: str) -> bool:
    """检查输入安全性"""
    # 检查是否包含敏感信息
    sensitive_keywords = ["password", "ssn", "credit card"]
    for keyword in sensitive_keywords:
        if keyword in input_data.lower():
            return False
    return True

# 输出护栏
def check_output_quality(output: str) -> bool:
    """检查输出质量"""
    # 检查是否包含不当内容
    if len(output) < 10:
        return False
    return True

# 应用护栏
safe_agent = Agent(
    name="Safe Agent",
    instructions="安全的 AI 助手",
    input_guardrails=[
        InputGuardrail(guardrail_function=check_input_safety)
    ],
    output_guardrails=[
        OutputGuardrail(guardrail_function=check_output_quality)
    ]
)
```

### 4.5 多 Agent 编排

```python
from agents import Agent, Runner, handoff

# 创建专业 Agent
research_agent = Agent(
    name="Researcher",
    instructions="负责研究和信息收集",
    tools=[search_tool]
)

writing_agent = Agent(
    name="Writer",
    instructions="负责撰写报告",
    tools=[write_file_tool]
)

review_agent = Agent(
    name="Reviewer",
    instructions="负责审查内容质量",
)

# 编排流程
# 研究 → 写作 → 审查
workflow = [
    research_agent,
    handoff(writing_agent),
    handoff(review_agent)
]

# 执行
async def run_workflow():
    result = await Runner.run(
        research_agent,
        "调研 2026 年 AI Agent 开发趋势"
    )
    print(result.final_output)
```

### 4.6 实战案例：客户服务系统

```python
from agents import Agent, function_tool, Runner, handoff
import asyncio

# 工具
@function_tool
def check_order_status(order_id: str) -> str:
    """查询订单状态"""
    # 查询数据库
    return f"订单 {order_id} 状态：已发货"

@function_tool
def process_refund(order_id: str) -> str:
    """处理退款"""
    return f"订单 {order_id} 退款已处理"

@function_tool
def escalate_to_human(issue: str) -> str:
    """升级到人工客服"""
    ticket_id = f"ESC-{random.randint(1000, 9999)}"
    return f"已升级到人工客服，工单号：{ticket_id}"

# Agent
triage_agent = Agent(
    name="Triage Agent",
    instructions="""你是客服分拣员。
职责：
1. 理解客户问题
2. 判断问题类型
3. 分派给合适的专业 Agent

问题类型：
- 订单查询 → Order Agent
- 退款请求 → Refund Agent
- 复杂问题 → Escalation Agent""",
    handoffs=[
        handoff(order_agent),
        handoff(refund_agent),
        handoff(escalation_agent)
    ]
)

order_agent = Agent(
    name="Order Agent",
    instructions="处理订单查询问题",
    tools=[check_order_status]
)

refund_agent = Agent(
    name="Refund Agent",
    instructions="处理退款请求",
    tools=[process_refund]
)

escalation_agent = Agent(
    name="Escalation Agent",
    instructions="处理复杂问题，必要时升级人工",
    tools=[escalate_to_human]
)

# 运行系统
async def customer_service_system():
    """客户服务系统"""
    # 模拟客户输入
    customer_queries = [
        "我的订单 #12345 到哪了？",
        "我想退款，订单号 #67890",
        "我收到的商品有质量问题，要求赔偿"
    ]
    
    for query in customer_queries:
        print(f"\n客户：{query}")
        result = await Runner.run(triage_agent, query)
        print(f"系统：{result.final_output}")

# 执行
asyncio.run(customer_service_system())
```

## 5. Google ADK 实战

### 5.1 框架简介

**Google ADK**（Agent Development Kit）是 Google 发布的企业级 Agent 开发工具包，核心特点：

- **模块化架构**：Agent、Tool、Session 核心组件
- **Google 生态集成**：与 Gemini、Vertex AI、Google Cloud 深度集成
- **企业级特性**：会话管理、状态持久化、监控
- **多语言支持**：Python、TypeScript
- **生产就绪**：Google Cloud 部署优化

### 5.2 Agent 开发套件架构

```
Google ADK 架构
┌──────────────────────────────────────────────────────┐
│                   Application Layer                    │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │ Agent       │  │ Tool        │  │ Session      │  │
│  │ Orchestrator│  │ Registry    │  │ Manager      │  │
│  └─────────────┘  └─────────────┘  └──────────────┘  │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                   Core Framework                       │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │ LLM         │  │ Memory      │  │ Streaming    │  │
│  │ Interface   │  │ Management  │  │ Handler      │  │
│  └─────────────┘  └─────────────┘  └──────────────┘  │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                 Cloud Integration                      │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │ Gemini API  │  │ Vertex AI   │  │ Cloud Run    │  │
│  └─────────────┘  └─────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 5.3 工具定义与注册

```python
from google.adk import Agent, Tool, Session

# 定义工具
class WeatherTool(Tool):
    """天气查询工具"""
    
    name = "weather_query"
    description = "查询指定城市的天气信息"
    
    parameters = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称"
            }
        },
        "required": ["city"]
    }
    
    async def execute(self, city: str) -> str:
        """执行天气查询"""
        # 调用天气 API
        weather_data = await fetch_weather(city)
        return f"{city} 的天气：{weather_data['description']}, 温度 {weather_data['temp']}°C"

# 注册工具
tools = [WeatherTool()]
```

### 5.4 会话管理

```python
from google.adk import SessionManager

# 会话管理器
session_manager = SessionManager()

# 创建会话
session = await session_manager.create_session(
    user_id="user_123",
    app_id="weather_app"
)

# 添加消息
await session_manager.add_message(
    session_id=session.id,
    role="user",
    content="北京今天天气怎么样？"
)

# 获取会话历史
history = await session_manager.get_session_history(session.id)
```

### 5.5 与 Google 生态集成

```python
from google.adk import Agent
from google.adk.llm import GeminiModel

# 使用 Gemini 模型
model = GeminiModel(
    model_name="gemini-pro",
    api_key="your-api-key"
)

# 创建 Agent
agent = Agent(
    name="Google Assistant",
    model=model,
    instructions="你是 Google 助手。",
    tools=[WeatherTool(), CalendarTool(), MailTool()]
)
```

### 5.6 实战案例：企业知识助手

```python
from google.adk import Agent, Tool, SessionManager
import asyncio

# 企业搜索工具
class EnterpriseSearchTool(Tool):
    """企业知识库搜索"""
    
    name = "enterprise_search"
    description = "搜索企业内部文档和知识库"
    
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查询"
            },
            "department": {
                "type": "string",
                "description": "部门（可选）"
            }
        },
        "required": ["query"]
    }
    
    async def execute(self, query: str, department: str = None) -> str:
        """执行企业搜索"""
        # 搜索企业知识库
        results = await search_enterprise_knowledge(
            query=query,
            department=department
        )
        return format_search_results(results)

# HR 专用工具
class HRTool(Tool):
    """HR 相关查询"""
    
    name = "hr_query"
    description = "查询 HR 政策、假期、福利等信息"
    
    async def execute(self, query: str) -> str:
        """执行 HR 查询"""
        return await query_hr_system(query)

# 创建 Agent
knowledge_agent = Agent(
    name="Enterprise Knowledge Assistant",
    instructions="""你是企业知识助手。
职责：
1. 回答员工关于公司政策的问题
2. 搜索企业内部文档
3. 提供准确的信息和引用""",
    tools=[EnterpriseSearchTool(), HRTool()]
)

# 运行
async def enterprise_assistant():
    """企业知识助手"""
    session_manager = SessionManager()
    session = await session_manager.create_session(
        user_id="employee_001",
        app_id="enterprise_knowledge"
    )
    
    # 员工查询
    queries = [
        "公司的年假政策是什么？",
        "如何申请远程工作？",
        "报销流程是什么？"
    ]
    
    for query in queries:
        print(f"\n员工：{query}")
        response = await knowledge_agent.run(query, session=session)
        print(f"助手：{response}")

asyncio.run(enterprise_assistant())
```

## 6. 多 Agent 通信与协调

### 6.1 通信协议设计

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional

class MessageType(Enum):
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    REQUEST = "request"
    RESPONSE = "response"

@dataclass
class AgentMessage:
    """Agent 间通信消息"""
    message_id: str
    sender: str
    receiver: str
    message_type: MessageType
    content: Any
    timestamp: float
    reply_to: Optional[str] = None
    metadata: dict = None

class MessageBus:
    """消息总线"""
    
    def __init__(self):
        self.subscribers = {}
        self.message_queue = []
    
    def subscribe(self, agent_name: str, callback):
        """订阅消息"""
        if agent_name not in self.subscribers:
            self.subscribers[agent_name] = []
        self.subscribers[agent_name].append(callback)
    
    def publish(self, message: AgentMessage):
        """发布消息"""
        self.message_queue.append(message)
        
        if message.receiver in self.subscribers:
            for callback in self.subscribers[message.receiver]:
                callback(message)
    
    def get_pending_messages(self, agent_name: str) -> list:
        """获取待处理消息"""
        return [
            msg for msg in self.message_queue
            if msg.receiver == agent_name
        ]
```

### 6.2 消息格式标准化

```python
import json
from datetime import datetime

def create_task_message(
    sender: str,
    receiver: str,
    task_type: str,
    task_data: dict
) -> AgentMessage:
    """创建任务消息"""
    return AgentMessage(
        message_id=f"msg_{uuid.uuid4().hex[:8]}",
        sender=sender,
        receiver=receiver,
        message_type=MessageType.TASK,
        content={
            "task_type": task_type,
            "task_data": task_data,
            "priority": "normal",
            "deadline": None
        },
        timestamp=datetime.now().timestamp()
    )

def create_result_message(
    sender: str,
    receiver: str,
    result: Any,
    reply_to: str
) -> AgentMessage:
    """创建结果消息"""
    return AgentMessage(
        message_id=f"msg_{uuid.uuid4().hex[:8]}",
        sender=sender,
        receiver=receiver,
        message_type=MessageType.RESULT,
        content={
            "result": result,
            "status": "success",
            "metadata": {}
        },
        timestamp=datetime.now().timestamp(),
        reply_to=reply_to
    )
```

### 6.3 冲突解决策略

```python
class ConflictResolver:
    """冲突解决器"""
    
    @staticmethod
    def voting_resolution(responses: list) -> dict:
        """投票决议"""
        # 统计投票
        vote_count = {}
        for response in responses:
            result = response['result']
            vote_count[result] = vote_count.get(result, 0) + 1
        
        # 选择最多投票
        best_result = max(vote_count, key=vote_count.get)
        
        return {
            "result": best_result,
            "confidence": vote_count[best_result] / len(responses),
            "votes": vote_count
        }
    
    @staticmethod
    def expert_resolution(responses: list, weights: list) -> dict:
        """专家权重决议"""
        # 加权投票
        weighted_votes = {}
        for response, weight in zip(responses, weights):
            result = response['result']
            weighted_votes[result] = weighted_votes.get(result, 0) + weight
        
        best_result = max(weighted_votes, key=weighted_votes.get)
        
        return {
            "result": best_result,
            "confidence": weighted_votes[best_result] / sum(weights),
            "weighted_votes": weighted_votes
        }
    
    @staticmethod
    def consensus_resolution(responses: list, threshold: float = 0.8) -> dict:
        """共识决议"""
        # 检查是否达成共识
        vote_count = {}
        for response in responses:
            result = response['result']
            vote_count[result] = vote_count.get(result, 0) + 1
        
        total = len(responses)
        for result, count in vote_count.items():
            if count / total >= threshold:
                return {
                    "result": result,
                    "confidence": count / total,
                    "consensus": True
                }
        
        # 未达成共识，返回最多投票
        return voting_resolution(responses)
```

### 6.4 共识机制

```python
class ConsensusProtocol:
    """共识协议"""
    
    def __init__(self, agents: list, threshold: float = 0.75):
        self.agents = agents
        self.threshold = threshold
    
    async def reach_consensus(self, task: str) -> dict:
        """达成共识"""
        # 1. 并行收集所有 Agent 的意见
        responses = await asyncio.gather(*[
            agent.run(task) for agent in self.agents
        ])
        
        # 2. 分析响应
        result_counts = {}
        for response in responses:
            result = response['result']
            result_counts[result] = result_counts.get(result, 0) + 1
        
        # 3. 检查是否达成共识
        total = len(responses)
        for result, count in result_counts.items():
            if count / total >= self.threshold:
                return {
                    "result": result,
                    "consensus": True,
                    "confidence": count / total,
                    "responses": responses
                }
        
        # 4. 未达成共识，使用投票
        return {
            "result": max(result_counts, key=result_counts.get),
            "consensus": False,
            "confidence": max(result_counts.values()) / total,
            "responses": responses
        }
```

## 7. Agent 编排模式

### 7.1 顺序编排

```python
def sequential_orchestration(agents: list, task: str):
    """顺序编排"""
    result = task
    
    for agent in agents:
        result = agent.run(result)
    
    return result

# 示例：文档处理流水线
agents = [
    research_agent,    # 研究
    writing_agent,     # 写作
    editing_agent,     # 编辑
    publishing_agent   # 发布
]

final_result = sequential_orchestration(agents, "撰写技术博客")
```

### 7.2 并行编排

```python
async def parallel_orchestration(agents: list, task: str):
    """并行编排"""
    # 并行执行所有 Agent
    results = await asyncio.gather(*[
        agent.run(task) for agent in agents
    ])
    
    # 合并结果
    merged_result = merge_results(results)
    
    return merged_result

# 示例：并行研究多个主题
research_topics = [
    "LLM 训练技术",
    "Agent 开发框架",
    "RAG 系统优化"
]

agents = [create_researcher(topic) for topic in research_topics]
results = await parallel_orchestration(agents, "调研最新进展")
```

### 7.3 投票与仲裁

```python
def voting_orchestration(agents: list, task: str):
    """投票编排"""
    # 收集所有 Agent 的投票
    votes = [agent.run(task) for agent in agents]
    
    # 投票决议
    result = ConflictResolver.voting_resolution(votes)
    
    return result

# 示例：代码审查投票
reviewers = [reviewer1, reviewer2, reviewer3]
review_result = voting_orchestration(reviewers, "审查代码质量")
```

### 7.4 层级管理

```python
class HierarchicalOrchestrator:
    """层级编排器"""
    
    def __init__(self, manager: Agent, workers: list):
        self.manager = manager
        self.workers = workers
    
    def execute(self, task: str):
        """执行层级任务"""
        # 1. 管理器分解任务
        subtasks = self.manager.decompose_task(task)
        
        # 2. 分配给工人
        results = []
        for subtask, worker in zip(subtasks, self.workers):
            result = worker.run(subtask)
            results.append(result)
        
        # 3. 管理器整合结果
        final_result = self.manager.integrate_results(results)
        
        return final_result

# 示例：项目管理
project_manager = Agent(
    name="Project Manager",
    instructions="分解项目任务，整合结果"
)

workers = [developer, designer, tester]

orchestrator = HierarchicalOrchestrator(project_manager, workers)
result = orchestrator.execute("开发 Web 应用")
```

### 7.5 动态编排

```python
def dynamic_orchestration(agent_pool: dict, task: str):
    """动态编排"""
    # 1. 分析任务需求
    required_skills = analyze_task_requirements(task)
    
    # 2. 选择合适的 Agent
    selected_agents = []
    for skill in required_skills:
        if skill in agent_pool:
            selected_agents.append(agent_pool[skill])
    
    # 3. 执行任务
    result = task
    for agent in selected_agents:
        result = agent.run(result)
    
    return result

# 示例：动态技能匹配
agent_pool = {
    "research": researcher,
    "coding": developer,
    "design": designer,
    "testing": tester,
    "writing": writer
}

result = dynamic_orchestration(agent_pool, "开发并文档化一个 API")
```

## 8. 最佳实践与陷阱

### 8.1 调试与可观测性

```python
import logging
from typing import Optional

class AgentTracer:
    """Agent 追踪器"""
    
    def __init__(self):
        self.traces = []
    
    def trace_agent_execution(
        self,
        agent_name: str,
        input_data: str,
        output_data: str,
        duration: float,
        metadata: dict = None
    ):
        """追踪 Agent 执行"""
        trace = {
            "agent": agent_name,
            "input": input_data[:200],  # 截断
            "output": output_data[:200],
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.traces.append(trace)
        logging.info(f"Agent {agent_name} 执行完成：{duration:.2f}s")
    
    def generate_report(self) -> dict:
        """生成执行报告"""
        if not self.traces:
            return {}
        
        total_duration = sum(t['duration'] for t in self.traces)
        agent_stats = {}
        
        for trace in self.traces:
            agent = trace['agent']
            if agent not in agent_stats:
                agent_stats[agent] = {
                    "count": 0,
                    "total_duration": 0
                }
            agent_stats[agent]['count'] += 1
            agent_stats[agent]['total_duration'] += trace['duration']
        
        return {
            "total_traces": len(self.traces),
            "total_duration": total_duration,
            "agent_stats": agent_stats,
            "traces": self.traces
        }
```

### 8.2 成本控制

```python
class CostManager:
    """成本管理器"""
    
    def __init__(self, budget: float):
        self.budget = budget
        self.spent = 0.0
    
    def track_cost(self, agent_name: str, tokens: int, cost: float):
        """追踪成本"""
        self.spent += cost
        
        logging.info(
            f"Agent {agent_name} 使用 {tokens} tokens, "
            f"成本 ${cost:.4f}, "
            f"累计 ${self.spent:.4f}"
        )
        
        if self.spent > self.budget:
            logging.warning(f"超出预算！已花费 ${self.spent:.4f}")
    
    def get_remaining_budget(self) -> float:
        """获取剩余预算"""
        return self.budget - self.spent
    
    def is_within_budget(self) -> bool:
        """是否在预算内"""
        return self.spent <= self.budget

# 使用示例
cost_manager = CostManager(budget=10.0)  # $10 预算

def cost_aware_agent_run(agent, task):
    """成本感知的 Agent 执行"""
    result = agent.run(task)
    
    # 记录成本
    cost_manager.track_cost(
        agent_name=agent.name,
        tokens=result.tokens_used,
        cost=result.cost
    )
    
    if not cost_manager.is_within_budget():
        logging.error("预算不足，停止执行")
        return None
    
    return result
```

### 8.3 错误处理

```python
class AgentErrorHandler:
    """Agent 错误处理器"""
    
    @staticmethod
    def handle_with_retry(agent, task, max_retries: int = 3):
        """带重试的错误处理"""
        for attempt in range(max_retries):
            try:
                result = agent.run(task)
                return result
            except Exception as e:
                logging.warning(
                    f"Agent {agent.name} 第 {attempt + 1} 次尝试失败：{e}"
                )
                if attempt == max_retries - 1:
                    raise
        
        return None
    
    @staticmethod
    def handle_with_fallback(primary_agent, fallback_agent, task):
        """带 fallback 的错误处理"""
        try:
            return primary_agent.run(task)
        except Exception as e:
            logging.warning(
                f"主 Agent {primary_agent.name} 失败，使用 fallback: {e}"
            )
            try:
                return fallback_agent.run(task)
            except Exception as e2:
                logging.error(f"Fallback 也失败：{e2}")
                return None
```

### 8.4 性能优化

```python
# 性能优化建议
optimization_tips = {
    "并行化": "尽可能并行执行独立任务",
    "缓存": "缓存 Agent 结果避免重复计算",
    "批处理": "批量处理相似请求",
    "流式输出": "使用流式输出减少延迟",
    "模型选择": "简单任务使用小模型，复杂任务用大模型",
    "上下文管理": "及时清理不需要的上下文",
    "工具优化": "优化工具调用，减少不必要调用"
}

# 缓存示例
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_agent_run(agent_name: str, task: str) -> str:
    """缓存 Agent 执行结果"""
    agent = get_agent(agent_name)
    return agent.run(task)
```

## 9. 参考资料

### 9.1 框架文档

- CrewAI: https://github.com/crewAIInc/crewAI
- AutoGen/AG2: https://github.com/ag2ai/ag2
- OpenAI Agents SDK: https://github.com/openai/openai-agents-python
- Google ADK: https://cloud.google.com/vertex-ai/generative-ai/docs/agent-dev-kit/overview

### 9.2 对比与评测

- Best Multi-Agent Frameworks in 2026: https://gurusup.com/blog/best-multi-agent-frameworks-2026
- AutoGen vs CrewAI vs LangGraph: https://newsletter.victordibia.com/p/autogen-vs-crewai-vs-langgraph-vs
- Complete Guide to AI Agent Frameworks 2026: https://turion.ai/blog/complete-guide-ai-agent-frameworks-2026

### 9.3 教程

- 10 AI Agent Frameworks You Should Know in 2026: https://medium.com/@atnoforgenai/10-ai-agent-frameworks-you-should-know-in-2026
- 一文解析 7 大主流 AI Agent 开发框架: https://zhuanlan.zhihu.com/p/1955596999832351185

---

## 10. 工具调用深度实践

### A.1 OpenAI Function Calling 详解

#### JSON Schema 定义

OpenAI Function Calling 使用 JSON Schema 定义工具的结构：

```python
from openai import OpenAI

client = OpenAI()

# 定义工具
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的当前天气信息",
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
                    "description": "温度单位",
                    "default": "celsius"
                }
            },
            "required": ["city"]
        }
    }
}

# 使用工具
response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=[weather_tool],
    tool_choice="auto"
)

# 处理工具调用
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    if tool_call.function.name == "get_weather":
        import json
        args = json.loads(tool_call.function.arguments)
        weather = get_weather(args["city"], args.get("unit", "celsius"))
        
        # 继续对话
        response2 = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "user", "content": "北京今天天气怎么样？"},
                response.choices[0].message,
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": weather
                }
            ],
            tools=[weather_tool]
        )
```

#### 函数描述最佳实践

```python
# 好的描述示例
good_function_def = {
    "name": "search_database",
    "description": "在企业知识库中搜索相关文档和信息。当用户询问公司政策、流程或历史数据时使用此工具。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词或短语。应该简洁且具体，避免使用模糊词汇。"
            },
            "category": {
                "type": "string",
                "enum": ["hr", "finance", "engineering", "general"],
                "description": "搜索的文档类别。如果不清楚，使用 'general'。"
            },
            "limit": {
                "type": "integer",
                "description": "返回结果的最大数量，默认为 5，最大为 20。",
                "minimum": 1,
                "maximum": 20,
                "default": 5
            }
        },
        "required": ["query"]
    }
}

# 描述最佳实践要点：
# 1. 函数名：使用动词_名词格式，清晰表达功能
# 2. 描述：说明何时使用此工具，包含使用场景
# 3. 参数描述：每个参数都要详细说明，包括格式、范围、默认值
# 4. 枚举值：尽可能使用 enum 限制参数范围
# 5. 必填参数：只标记真正必需的参数为 required
```

#### 参数验证

```python
import jsonschema
from typing import Any, Dict

class ToolParameterValidator:
    """工具参数验证器"""
    
    def __init__(self, tool_schema: Dict[str, Any]):
        self.schema = tool_schema['function']['parameters']
    
    def validate(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """验证参数是否符合 schema"""
        try:
            jsonschema.validate(instance=parameters, schema=self.schema)
            return True, "验证通过"
        except jsonschema.ValidationError as e:
            return False, f"验证失败：{e.message}"
    
    def validate_and_fix(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """验证并修复参数"""
        validated = {}
        
        # 填充默认值
        for prop_name, prop_schema in self.schema['properties'].items():
            if prop_name in parameters:
                validated[prop_name] = parameters[prop_name]
            elif 'default' in prop_schema:
                validated[prop_name] = prop_schema['default']
        
        # 验证必填参数
        for required_param in self.schema.get('required', []):
            if required_param not in validated:
                raise ValueError(f"缺少必填参数：{required_param}")
        
        return validated

# 使用示例
validator = ToolParameterValidator(weather_tool)
is_valid, message = validator.validate({"city": "北京", "unit": "celsius"})
print(f"{is_valid}: {message}")
```

#### 多工具调用

```python
# 定义多个工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "获取股票价格",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "搜索新闻",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "date_range": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]

# 并行工具调用
response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[
        {"role": "user", "content": "北京天气如何？另外帮我查下苹果股价和最近的 AI 新闻"}
    ],
    tools=tools
)

# 处理多个工具调用
if response.choices[0].message.tool_calls:
    tool_calls = response.choices[0].message.tool_calls
    print(f"需要调用 {len(tool_calls)} 个工具")
    
    tool_results = []
    for tool_call in tool_calls:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        # 执行对应的函数
        if func_name == "get_weather":
            result = get_weather(args["city"])
        elif func_name == "get_stock_price":
            result = get_stock_price(args["symbol"])
        elif func_name == "search_news":
            result = search_news(args["query"], args.get("date_range"))
        
        tool_results.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })
    
    # 发送所有工具结果
    final_response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "user", "content": "北京天气如何？另外帮我查下苹果股价和最近的 AI 新闻"},
            response.choices[0].message,
            *tool_results
        ],
        tools=tools
    )
```

#### Structured Outputs

```python
from pydantic import BaseModel
from openai import pydantic_function_tool

# 定义输出结构
class WeatherReport(BaseModel):
    city: str
    temperature: float
    humidity: float
    condition: str
    recommendation: str

# 使用 Structured Outputs
response = client.beta.chat.completions.parse(
    model="gpt-5.2",
    messages=[{"role": "user", "content": "北京天气怎么样？给我穿衣建议"}],
    response_format=WeatherReport
)

# 直接获取结构化对象
weather_report = response.choices[0].message.parsed
print(f"城市：{weather_report.city}")
print(f"温度：{weather_report.temperature}°C")
print(f"建议：{weather_report.recommendation}")

# 在工具中使用 Structured Outputs
class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    relevance_score: float

class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_count: int
    search_time_ms: int
```

### A.2 Anthropic Tool Use

#### XML 格式工具定义

Anthropic Claude 使用 XML 格式定义工具：

```python
from anthropic import Anthropic

client = Anthropic()

# 定义工具
weather_tool = {
    "name": "get_weather",
    "description": "获取指定城市的天气信息",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称"
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

# 调用模型
message = client.messages.create(
    model="claude-4-6-20251001",
    max_tokens=1024,
    tools=[weather_tool],
    messages=[
        {"role": "user", "content": "北京今天天气怎么样？"}
    ]
)

# 处理工具使用
if message.stop_reason == "tool_use":
    tool_use = next(block for block in message.content if block.type == "tool_use")
    print(f"工具：{tool_use.name}")
    print(f"参数：{tool_use.input}")
    
    # 执行工具
    weather = get_weather(tool_use.input["city"])
    
    # 继续对话
    message2 = client.messages.create(
        model="claude-4-6-20251001",
        max_tokens=1024,
        tools=[weather_tool],
        messages=[
            {"role": "user", "content": "北京今天天气怎么样？"},
            {"role": "assistant", "content": message.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": weather
                    }
                ]
            }
        ]
    )
```

#### 工具使用流程

```python
def anthropic_tool_workflow(query: str, tools: list) -> str:
    """Anthropic 工具使用完整流程"""
    messages = [{"role": "user", "content": query}]
    
    max_iterations = 10
    for _ in range(max_iterations):
        # 1. 调用模型
        response = client.messages.create(
            model="claude-4-6-20251001",
            max_tokens=2048,
            tools=tools,
            messages=messages
        )
        
        # 2. 检查是否需要使用工具
        if response.stop_reason != "tool_use":
            # 模型完成响应
            final_text = "".join(
                block.text for block in response.content 
                if block.type == "text"
            )
            return final_text
        
        # 3. 添加工具使用到消息
        messages.append({"role": "assistant", "content": response.content})
        
        # 4. 执行工具
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # 执行对应工具
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        
        # 5. 添加工具结果到消息
        messages.append({"role": "user", "content": tool_results})
    
    return "达到最大迭代次数"
```

#### 与 OpenAI 对比

| 特性 | OpenAI | Anthropic |
|------|--------|-----------|
| **工具格式** | JSON Schema | JSON Schema |
| **消息格式** | 纯文本 | XML 标记 |
| **多工具调用** | 支持并行 | 支持并行 |
| **工具结果** | tool role | tool_result block |
| **强制工具** | tool_choice="required" | 无直接等价 |
| **流式输出** | 支持 | 支持 |
| **最大工具数** | 128 | 无明确限制 |

```python
# OpenAI 格式
openai_messages = [
    {"role": "user", "content": "查询天气"},
    {"role": "assistant", "content": None, "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "结果"}
]

# Anthropic 格式
anthropic_messages = [
    {"role": "user", "content": "查询天气"},
    {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "...", "name": "get_weather", "input": {...}}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "...", "content": "结果"}
        ]
    }
]
```

#### 最佳实践

```python
# 1. 提供清晰的工具描述
best_practice_tools = [
    {
        "name": "search_knowledge_base",
        "description": "在企业知识库中搜索文档。当用户询问公司内部信息、政策或流程时使用。不要用于查询外部信息。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询。应该简洁明确，包含关键词。"
                }
            },
            "required": ["query"]
        }
    }
]

# 2. 使用系统提示引导工具使用
system_prompt = """你是一个有帮助的助手。你可以使用以下工具：

- search_knowledge_base: 搜索内部知识库
- send_email: 发送邮件
- create_ticket: 创建支持工单

当需要使用工具时，你会收到工具使用请求。执行工具后将结果返回给用户。
"""

# 3. 错误处理
try:
    response = client.messages.create(
        model="claude-4-6-20251001",
        max_tokens=2048,
        tools=tools,
        messages=messages
    )
except anthropic.BadRequestError as e:
    if "tool_use" in str(e):
        print("工具使用错误，尝试调整参数")
    else:
        print(f"API 错误：{e}")
```

### A.3 LangChain 工具系统

#### BaseTool 抽象

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type

class WeatherInput(BaseModel):
    """天气查询输入"""
    city: str = Field(description="城市名称")
    unit: str = Field(default="celsius", description="温度单位")

class WeatherTool(BaseTool):
    """天气查询工具"""
    name: str = "get_weather"
    description: str = "获取指定城市的天气信息"
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, city: str, unit: str = "celsius") -> str:
        """同步执行"""
        # 调用天气 API
        weather_data = fetch_weather_api(city, unit)
        return format_weather_response(weather_data)
    
    async def _arun(self, city: str, unit: str = "celsius") -> str:
        """异步执行"""
        weather_data = await fetch_weather_api_async(city, unit)
        return format_weather_response(weather_data)

# 使用工具
tool = WeatherTool()
result = tool.invoke({"city": "北京", "unit": "celsius"})
```

#### 内置工具库

```python
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun,
    ShellTool,
    FileReadTool,
    FileWriteTool,
    YouTubeSearchTool
)

# 搜索工具
search_tool = DuckDuckGoSearchRun()
search_result = search_tool.run("2026 年 AI 趋势")

# 维基百科
wiki_tool = WikipediaQueryRun()
wiki_result = wiki_tool.run("人工智能")

# 学术搜索
arxiv_tool = ArxivQueryRun()
arxiv_result = arxiv_tool.run("LLM Agent")

# Shell 执行
shell_tool = ShellTool()
shell_result = shell_tool.run("ls -la")

# 文件操作
read_tool = FileReadTool()
write_tool = FileWriteTool()
```

#### 自定义工具创建

```python
from langchain.tools import tool

# 使用装饰器创建工具
@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """货币转换"""
    rates = {
        "USD_CNY": 7.2,
        "EUR_CNY": 7.8,
        "GBP_CNY": 9.1
    }
    key = f"{from_currency}_{to_currency}"
    if key in rates:
        converted = amount * rates[key]
        return f"{amount} {from_currency} = {converted:.2f} {to_currency}"
    return f"不支持的货币对：{key}"

# 工具列表
tools = [calculate, convert_currency]
```

#### 工具链编排

```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# 初始化 LLM
llm = ChatOpenAI(model="gpt-5.2", temperature=0)

# 创建 Agent
agent = initialize_agent(
    tools=[search_tool, calculator_tool, weather_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 运行
result = agent.run("北京今天天气如何？如果温度低于 10 度，计算 100 美元能换多少人民币")

# 使用 ReAct 模式
from langchain.agents import AgentExecutor, create_react_agent

prompt = hub.pull("hwchase17/react")
react_agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)
result = agent_executor.invoke({"input": "查询北京天气并计算"})
```

#### 错误处理

```python
from langchain_core.tools import ToolException

class RobustTool(BaseTool):
    """健壮的工具实现"""
    
    def _run(self, *args, **kwargs) -> str:
        try:
            return self.execute(*args, **kwargs)
        except TimeoutError:
            raise ToolException("工具执行超时")
        except ValueError as e:
            raise ToolException(f"参数错误：{e}")
        except Exception as e:
            raise ToolException(f"工具执行失败：{e}")
    
    def execute(self, *args, **kwargs):
        """实际执行逻辑"""
        pass

# 工具降级
@tool
def search_with_fallback(query: str) -> str:
    """带降级的搜索"""
    try:
        # 主搜索
        return duckduckgo_search(query)
    except Exception:
        try:
            # 备用搜索
            return wikipedia_search(query)
        except Exception:
            return f"无法搜索：{query}"
```

### A.4 MCP (Model Context Protocol)

#### MCP 协议概述

MCP 是 Anthropic 提出的开放协议，用于标准化 LLM 与外部工具的集成：

```
MCP 架构
┌─────────────┐         ┌─────────────┐
│   Client    │◄───────►│   Server    │
│  (LLM App)  │  JSON   │  (Tools)    │
│             │  RPC    │             │
└─────────────┘         └─────────────┘
       │                       │
       │                       │
┌──────▼───────────────────────▼──────┐
│          Transport Layer            │
│  (stdio, HTTP, SSE, WebSockets)     │
└─────────────────────────────────────┘
```

#### 工具标准化

```python
# MCP Server 实现
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather Server")

@mcp.tool()
def get_weather(city: str) -> str:
    """获取天气信息"""
    return fetch_weather(city)

@mcp.tool()
def get_forecast(city: str, days: int = 3) -> str:
    """获取天气预报"""
    return fetch_forecast(city, days)

if __name__ == "__main__":
    mcp.run()
```

#### 客户端实现

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_client():
    server_params = StdioServerParameters(
        command="python",
        args=["weather_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化
            await session.initialize()
            
            # 列出工具
            tools = await session.list_tools()
            print(f"可用工具：{[t.name for t in tools.tools]}")
            
            # 调用工具
            result = await session.call_tool(
                "get_weather",
                {"city": "北京"}
            )
            print(result.content)
```

#### 服务器实现

```python
# 资源服务器
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Knowledge Base")

@mcp.resource("kb://documents/{doc_id}")
def get_document(doc_id: str) -> str:
    """获取文档内容"""
    return load_document(doc_id)

@mcp.resource("kb://search?query={query}")
def search_documents(query: str) -> str:
    """搜索文档"""
    return search_knowledge_base(query)

# 提示模板
@mcp.prompt()
def analyze_code(code: str) -> str:
    """代码分析提示"""
    return f"""请分析以下代码：

{code}

分析：
1. 代码结构
2. 潜在问题
3. 改进建议
"""
```

#### 生态工具

```python
# 官方 MCP 工具
mcp_tools = [
    "filesystem",      # 文件系统操作
    "github",          # GitHub 集成
    "slack",           # Slack 集成
    "postgres",        # PostgreSQL 数据库
    "puppeteer",       # 浏览器自动化
    "google-maps",     # 谷歌地图
    "fetch",           # HTTP 请求
    "git",             # Git 操作
    "memory",          # 向量数据库
    "sequential-thinking"  # 链式思考
]

# 使用文件系统工具
@mcp.tool()
def list_files(path: str) -> str:
    """列出目录内容"""
    import os
    files = os.listdir(path)
    return "\n".join(files)

@mcp.tool()
def read_file(path: str) -> str:
    """读取文件"""
    with open(path, 'r') as f:
        return f.read()
```

### A.5 工具调用模式

#### ReAct 模式

```python
"""
ReAct (Reasoning + Acting) 模式

Thought: 我需要查询北京天气
Action: get_weather
Action Input: {"city": "北京"}
Observation: 北京今天晴，温度 15°C
Thought: 我已经获取了天气信息
Final Answer: 北京今天天气晴朗，温度 15°C
"""

from langchain.agents import create_react_agent

react_prompt = """回答以下问题，你可以使用工具：

{input}

可用的工具：
{tools}

使用以下格式：

Question: 输入问题
Thought: 你应该思考下一步做什么
Action: 工具名称
Action Input: 工具输入
Observation: 工具结果
... (这个过程可以重复多次)
Thought: 我知道最终答案了
Final Answer: 最终答案
"""

agent = create_react_agent(llm, tools, react_prompt)
```

#### Plan-and-Execute

```python
from typing import List

class PlanAndExecute:
    """计划-执行模式"""
    
    def __init__(self, planner, executor, replanner):
        self.planner = planner
        self.executor = executor
        self.replanner = replanner
    
    def run(self, task: str) -> str:
        # 1. 制定计划
        plan = self.planner.create_plan(task)
        print(f"计划：{plan}")
        
        # 2. 执行计划
        results = []
        for step in plan.steps:
            result = self.executor.execute(step)
            results.append(result)
            
            # 3. 必要时重新规划
            if self.replanner.need_replan(step, result):
                new_plan = self.replanner.replan(plan, results)
                plan = new_plan
        
        # 4. 生成最终答案
        return self.planner.synthesize(task, results)

# 使用示例
planner = create_planner_agent(llm)
executor = create_executor_agent(llm, tools)
replanner = create_replanner_agent(llm)

plan_execute = PlanAndExecute(planner, executor, replanner)
result = plan_execute.run("开发一个简单的 Web 应用并部署")
```

#### Reflection 模式

```python
class ReflectionAgent:
    """反思模式"""
    
    def __init__(self, generator, critic, max_iterations=3):
        self.generator = generator
        self.critic = critic
        self.max_iterations = max_iterations
    
    def run(self, task: str) -> str:
        # 1. 生成初始方案
        solution = self.generator.generate(task)
        
        for i in range(self.max_iterations):
            # 2. 批评方案
            feedback = self.critic.critic(solution, task)
            
            # 3. 检查是否通过
            if feedback.passed:
                return solution
            
            # 4. 根据反馈改进
            solution = self.generator.improve(
                solution, 
                feedback.suggestions
            )
        
        return solution

# 使用示例
generator = create_generator_agent(llm)
critic = create_critic_agent(llm)

reflection = ReflectionAgent(generator, critic, max_iterations=5)
result = reflection.run("编写一个安全的用户认证系统")
```

#### 多 Agent 工具共享

```python
from crewai import Agent, Task, Crew

# 共享工具库
shared_tools = [
    search_tool,
    file_tool,
    code_execution_tool
]

# 创建 Agent（共享工具）
researcher = Agent(
    role="研究员",
    tools=shared_tools,
    goal="研究问题"
)

programmer = Agent(
    role="程序员",
    tools=shared_tools + [specific_code_tool],  # 额外工具
    goal="编写代码"
)

# 工具权限管理
class ToolAccessControl:
    """工具访问控制"""
    
    def __init__(self):
        self.permissions = {}
    
    def grant_access(self, agent_name: str, tools: list):
        self.permissions[agent_name] = tools
    
    def check_access(self, agent_name: str, tool_name: str) -> bool:
        return tool_name in self.permissions.get(agent_name, [])

# 使用访问控制
access_control = ToolAccessControl()
access_control.grant_access("researcher", ["search", "read_file"])
access_control.grant_access("programmer", ["search", "read_file", "execute_code"])
```

### A.6 工具调用评估

#### 选择准确率

```python
class ToolSelectionEvaluator:
    """工具选择评估器"""
    
    def __init__(self):
        self.results = []
    
    def evaluate(self, test_cases: list) -> dict:
        """评估工具选择准确率"""
        correct = 0
        total = len(test_cases)
        
        for case in test_cases:
            predicted_tool = model_select_tool(case["query"])
            expected_tool = case["expected_tool"]
            
            is_correct = predicted_tool == expected_tool
            if is_correct:
                correct += 1
            
            self.results.append({
                "query": case["query"],
                "predicted": predicted_tool,
                "expected": expected_tool,
                "correct": is_correct
            })
        
        return {
            "accuracy": correct / total,
            "correct": correct,
            "total": total
        }
    
    def analyze_errors(self) -> list:
        """分析错误案例"""
        return [r for r in self.results if not r["correct"]]

# 测试用例
test_cases = [
    {
        "query": "北京今天天气怎么样？",
        "expected_tool": "get_weather"
    },
    {
        "query": "计算 123 * 456",
        "expected_tool": "calculator"
    }
]

evaluator = ToolSelectionEvaluator()
results = evaluator.evaluate(test_cases)
print(f"工具选择准确率：{results['accuracy']:.2%}")
```

#### 参数提取准确率

```python
class ParameterExtractionEvaluator:
    """参数提取评估器"""
    
    def evaluate_parameters(self, test_cases: list) -> dict:
        """评估参数提取准确率"""
        total_params = 0
        correct_params = 0
        
        for case in test_cases:
            predicted = model_extract_parameters(case["query"])
            expected = case["expected_params"]
            
            for param_name, expected_value in expected.items():
                total_params += 1
                if param_name in predicted:
                    if predicted[param_name] == expected_value:
                        correct_params += 1
        
        return {
            "parameter_accuracy": correct_params / total_params if total_params > 0 else 0,
            "correct": correct_params,
            "total": total_params
        }

# 测试
test_cases = [
    {
        "query": "查询北京明天天气，用华氏度",
        "expected_params": {
            "city": "北京",
            "unit": "fahrenheit",
            "date": "tomorrow"
        }
    }
]

param_evaluator = ParameterExtractionEvaluator()
param_results = param_evaluator.evaluate_parameters(test_cases)
print(f"参数提取准确率：{param_results['parameter_accuracy']:.2%}")
```

#### 执行成功率

```python
class ToolExecutionEvaluator:
    """工具执行评估器"""
    
    def evaluate_execution(self, test_cases: list) -> dict:
        """评估工具执行成功率"""
        successful = 0
        total = len(test_cases)
        errors = []
        
        for case in test_cases:
            try:
                result = execute_tool(case["tool_name"], case["params"])
                successful += 1
            except Exception as e:
                errors.append({
                    "tool": case["tool_name"],
                    "params": case["params"],
                    "error": str(e)
                })
        
        return {
            "success_rate": successful / total,
            "successful": successful,
            "total": total,
            "errors": errors
        }
```

#### 性能指标

```python
import time
from dataclasses import dataclass

@dataclass
class ToolMetrics:
    """工具性能指标"""
    tool_name: str
    call_count: int = 0
    total_time: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    avg_latency: float = 0.0
    
    def record_call(self, duration: float, success: bool):
        self.call_count += 1
        self.total_time += duration
        self.avg_latency = self.total_time / self.call_count
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
    
    def track_tool_call(self, tool_name: str):
        """追踪工具调用"""
        start_time = time.time()
        
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception:
                    result = None
                    success = False
                
                duration = time.time() - start_time
                
                if tool_name not in self.metrics:
                    self.metrics[tool_name] = ToolMetrics(tool_name)
                
                self.metrics[tool_name].record_call(duration, success)
                
                return result
            return wrapper
        return decorator
    
    def generate_report(self) -> dict:
        """生成性能报告"""
        return {
            name: {
                "call_count": m.call_count,
                "avg_latency_ms": m.avg_latency * 1000,
                "success_rate": m.success_rate
            }
            for name, m in self.metrics.items()
        }
```

## 11. Agent 评估与基准测试

### B.1 Agent 能力评估

#### 工具使用能力

```python
class ToolUseEvaluator:
    """工具使用能力评估"""
    
    dimensions = [
        "工具选择准确率",      # 能否选择正确的工具
        "参数提取准确率",      # 能否正确提取参数
        "工具组合能力",        # 能否组合多个工具
        "错误恢复能力",        # 工具失败时能否恢复
        "工具理解深度"         # 是否理解工具的局限性
    ]
    
    def evaluate(self, agent, test_suite) -> dict:
        """评估工具使用能力"""
        scores = {}
        
        for dimension in self.dimensions:
            score = self.test_dimension(agent, dimension, test_suite)
            scores[dimension] = score
        
        return {
            "overall_score": sum(scores.values()) / len(scores),
            "dimension_scores": scores
        }
    
    def test_dimension(self, agent, dimension, test_suite):
        """测试特定维度"""
        # 实现具体的测试逻辑
        pass
```

#### 规划能力

```python
class PlanningEvaluator:
    """规划能力评估"""
    
    def evaluate_planning(self, agent, tasks: list) -> dict:
        """评估规划能力"""
        metrics = {
            "plan_completeness": [],    # 计划完整性
            "plan_efficiency": [],      # 计划效率
            "plan_correctness": [],     # 计划正确性
            "replanning_ability": []    # 重新规划能力
        }
        
        for task in tasks:
            # 生成计划
            plan = agent.create_plan(task)
            
            # 评估完整性
            completeness = self.evaluate_completeness(plan, task)
            metrics["plan_completeness"].append(completeness)
            
            # 评估效率
            efficiency = self.evaluate_efficiency(plan)
            metrics["plan_efficiency"].append(efficiency)
            
            # 执行并评估正确性
            result = agent.execute_plan(plan)
            correctness = self.evaluate_correctness(result, task)
            metrics["plan_correctness"].append(correctness)
        
        return {
            metric: sum(values) / len(values) 
            for metric, values in metrics.items()
        }
```

#### 协作能力

```python
class CollaborationEvaluator:
    """协作能力评估"""
    
    def evaluate_collaboration(self, agents: list, task: str) -> dict:
        """评估多 Agent 协作"""
        # 执行协作任务
        result = execute_collaborative_task(agents, task)
        
        metrics = {
            "communication_efficiency": self.evaluate_communication(result),
            "task_allocation": self.evaluate_task_allocation(result),
            "conflict_resolution": self.evaluate_conflict_resolution(result),
            "final_quality": self.evaluate_final_quality(result)
        }
        
        return metrics
    
    def evaluate_communication(self, result) -> float:
        """评估通信效率"""
        # 分析消息数量、冗余度、信息密度
        pass
    
    def evaluate_task_allocation(self, result) -> float:
        """评估任务分配"""
        # 分析任务分配是否合理
        pass
```

#### 学习能力

```python
class LearningEvaluator:
    """学习能力评估"""
    
    def evaluate_learning(self, agent, training_tasks: list, test_tasks: list) -> dict:
        """评估学习能力"""
        # 训练前性能
        pre_performance = self.test_performance(agent, test_tasks)
        
        # 训练
        for task in training_tasks:
            feedback = agent.execute_with_feedback(task)
            agent.learn_from_feedback(feedback)
        
        # 训练后性能
        post_performance = self.test_performance(agent, test_tasks)
        
        improvement = post_performance - pre_performance
        
        return {
            "pre_performance": pre_performance,
            "post_performance": post_performance,
            "improvement": improvement,
            "learning_rate": improvement / len(training_tasks)
        }
```

### B.2 AgentBench

#### 任务类型

AgentBench 是评估 LLM Agent 能力的综合基准测试：

```
AgentBench 任务类型
├── 操作系统交互 (OSWorld)
│   ├── 文件操作
│   ├── 进程管理
│   └── 系统配置
│
├── 数据库交互 (DBBench)
│   ├── SQL 查询
│   ├── 数据库设计
│   └── 数据清洗
│
├── 网页交互 (WebArena)
│   ├── 表单填写
│   ├── 信息检索
│   └── 电子商务
│
├── 知识图谱 (KG-Bench)
│   ├── 实体查询
│   ├── 关系推理
│   └── 知识更新
│
├── 代码开发 (SWE-Bench)
│   ├── Bug 修复
│   ├── 功能开发
│   └── 代码审查
│
└── 数学推理 (Math-Bench)
    ├── 代数问题
    ├── 几何问题
    └── 统计问题
```

#### 评估指标

```python
class AgentBenchEvaluator:
    """AgentBench 评估器"""
    
    def evaluate(self, agent, benchmark_suite) -> dict:
        """运行完整基准测试"""
        results = {}
        
        for task_type, tasks in benchmark_suite.items():
            task_results = []
            
            for task in tasks:
                result = self.evaluate_single_task(agent, task)
                task_results.append(result)
            
            results[task_type] = {
                "success_rate": self.calculate_success_rate(task_results),
                "avg_steps": self.calculate_avg_steps(task_results),
                "avg_cost": self.calculate_avg_cost(task_results)
            }
        
        return results
    
    def evaluate_single_task(self, agent, task) -> dict:
        """评估单个任务"""
        start_time = time.time()
        
        try:
            result = agent.run(task)
            success = self.check_result(result, task.expected_output)
            
            return {
                "success": success,
                "steps": result.steps_taken,
                "cost": result.cost,
                "time": time.time() - start_time
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "time": time.time() - start_time
            }
```

#### 基准对比

```python
# AgentBench 典型结果（示例数据）
benchmark_results = {
    "GPT-4": {
        "OSWorld": 0.65,
        "DBBench": 0.72,
        "WebArena": 0.58,
        "SWE-Bench": 0.45,
        "overall": 0.60
    },
    "Claude-3-Opus": {
        "OSWorld": 0.62,
        "DBBench": 0.70,
        "WebArena": 0.55,
        "SWE-Bench": 0.42,
        "overall": 0.57
    },
    "Gemini-Ultra": {
        "OSWorld": 0.60,
        "DBBench": 0.68,
        "WebArena": 0.53,
        "SWE-Bench": 0.40,
        "overall": 0.55
    }
}
```

#### 局限性

```
AgentBench 局限性：
1. 任务覆盖有限：无法涵盖所有真实场景
2. 评估指标单一：主要依赖成功率，忽略质量差异
3. 成本考量不足：未充分评估 token 消耗
4. 时间约束简化：未模拟真实时间压力
5. 人类偏好缺失：未考虑人类主观评价
```

### B.3 WebArena

#### 网页交互评估

WebArena 是评估 Agent 网页操作能力的基准测试：

```python
class WebArenaEvaluator:
    """WebArena 评估器"""
    
    def __init__(self, browser):
        self.browser = browser
    
    def evaluate_task(self, agent, task) -> dict:
        """评估网页交互任务"""
        # 初始化浏览器
        self.browser.goto(task.start_url)
        
        # Agent 执行任务
        actions = []
        max_steps = 30
        
        for step in range(max_steps):
            # 获取当前页面状态
            page_state = self.get_page_state()
            
            # Agent 决定下一步
            action = agent.decide_action(page_state, task.goal)
            actions.append(action)
            
            # 执行动作
            self.execute_action(action)
            
            # 检查是否完成
            if self.check_completion(task):
                return {
                    "success": True,
                    "steps": step + 1,
                    "actions": actions
                }
        
        return {
            "success": False,
            "steps": max_steps,
            "actions": actions
        }
    
    def get_page_state(self) -> dict:
        """获取页面状态"""
        return {
            "url": self.browser.current_url,
            "title": self.browser.title,
            "content": self.browser.get_accessible_content(),
            "interactive_elements": self.browser.get_interactive_elements()
        }
```

#### 任务复杂度

```python
# WebArena 任务复杂度分级
task_complexity = {
    "简单": {
        "描述": "单页面操作，1-3 步",
        "示例": "填写表单、点击按钮",
        "预期成功率": ">80%"
    },
    "中等": {
        "描述": "多页面导航，4-10 步",
        "示例": "搜索商品、添加到购物车",
        "预期成功率": "50-80%"
    },
    "复杂": {
        "描述": "多步骤工作流，10+ 步",
        "示例": "完成购物结账流程",
        "预期成功率": "20-50%"
    },
    "非常复杂": {
        "描述": "跨网站操作，需要推理",
        "示例": "比较多个网站价格并购买",
        "预期成功率": "<20%"
    }
}
```

#### 成功率测量

```python
def calculate_webarena_metrics(results: list) -> dict:
    """计算 WebArena 指标"""
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r["success"])
    
    # 整体成功率
    overall_success_rate = successful_tasks / total_tasks
    
    # 按复杂度分组
    complexity_groups = {"简单": [], "中等": [], "复杂": [], "非常复杂": []}
    for result in results:
        complexity = result["complexity"]
        complexity_groups[complexity].append(result["success"])
    
    complexity_success_rates = {
        complexity: sum(successes) / len(successes) if successes else 0
        for complexity, successes in complexity_groups.items()
    }
    
    # 平均步骤数
    avg_steps = sum(r["steps"] for r in results) / total_tasks
    
    # 步骤效率（成功任务的平均步骤）
    successful_steps = [r["steps"] for r in results if r["success"]]
    avg_successful_steps = sum(successful_steps) / len(successful_steps) if successful_steps else 0
    
    return {
        "overall_success_rate": overall_success_rate,
        "complexity_success_rates": complexity_success_rates,
        "avg_steps": avg_steps,
        "avg_successful_steps": avg_successful_steps,
        "total_tasks": total_tasks,
        "successful_tasks": successful_tasks
    }
```

#### 人类对比

```python
class HumanComparisonStudy:
    """人类对比研究"""
    
    def conduct_study(self, tasks: list, num_participants: int = 30) -> dict:
        """进行人类对比研究"""
        # 人类参与者完成任务
        human_results = self.collect_human_results(tasks, num_participants)
        
        # Agent 完成相同任务
        agent_results = self.collect_agent_results(tasks)
        
        # 对比分析
        comparison = {
            "success_rate_comparison": {
                "human": human_results["success_rate"],
                "agent": agent_results["success_rate"],
                "gap": human_results["success_rate"] - agent_results["success_rate"]
            },
            "time_comparison": {
                "human": human_results["avg_time"],
                "agent": agent_results["avg_time"],
                "ratio": agent_results["avg_time"] / human_results["avg_time"]
            },
            "error_analysis": self.compare_errors(human_results, agent_results)
        }
        
        return comparison
```

### B.4 SWE-bench

#### 软件工程任务

SWE-bench 评估 Agent 解决真实 GitHub Issue 的能力：

```python
class SWEBenchEvaluator:
    """SWE-bench 评估器"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
    
    def evaluate_issue(self, agent, issue: dict) -> dict:
        """评估 Issue 解决能力"""
        # 1. 理解 Issue
        problem_statement = issue["description"]
        
        # 2. Agent 分析和定位问题
        analysis = agent.analyze_issue(problem_statement)
        
        # 3. Agent 编写修复代码
        patch = agent.generate_patch(analysis)
        
        # 4. 应用补丁并测试
        result = self.apply_and_test(patch, issue)
        
        return {
            "issue_id": issue["id"],
            "success": result["tests_pass"],
            "patch_quality": result["quality_score"],
            "analysis_accuracy": result["analysis_correct"],
            "time_taken": result["time"]
        }
    
    def apply_and_test(self, patch: str, issue: dict) -> dict:
        """应用补丁并运行测试"""
        import subprocess
        
        # 应用补丁
        apply_result = subprocess.run(
            ["git", "apply", "--check"],
            input=patch,
            capture_output=True,
            text=True
        )
        
        if apply_result.returncode != 0:
            return {"tests_pass": False, "error": "补丁应用失败"}
        
        # 运行测试
        test_result = subprocess.run(
            ["pytest", issue["test_file"]],
            capture_output=True,
            text=True
        )
        
        return {
            "tests_pass": test_result.returncode == 0,
            "quality_score": self.evaluate_patch_quality(patch),
            "analysis_correct": True,
            "time": 0
        }
```

#### Issue 解决

```python
class IssueSolver:
    """Issue 解决 Agent"""
    
    def solve_issue(self, repo, issue_description: str) -> str:
        """解决 Issue"""
        # 1. 理解问题
        problem = self.parse_issue(issue_description)
        
        # 2. 定位相关代码
        relevant_files = self.find_relevant_files(repo, problem)
        
        # 3. 分析问题根源
        root_cause = self.analyze_root_cause(relevant_files, problem)
        
        # 4. 设计解决方案
        solution = self.design_solution(root_cause)
        
        # 5. 实现修复
        patch = self.implement_fix(solution)
        
        # 6. 测试修复
        if self.test_fix(patch):
            return patch
        else:
            # 7. 迭代改进
            return self.iterate_fix(patch, problem)
    
    def find_relevant_files(self, repo, problem) -> list:
        """定位相关文件"""
        # 使用代码搜索
        search_results = search_codebase(repo, problem["keywords"])
        
        # 使用调用链分析
        call_graph = analyze_call_graph(repo, search_results)
        
        return call_graph
```

#### PR 生成

```python
class PRGenerator:
    """PR 生成器"""
    
    def generate_pr(self, patch: str, issue: dict) -> dict:
        """生成 Pull Request"""
        # 生成 PR 标题
        title = self.generate_title(issue)
        
        # 生成 PR 描述
        description = self.generate_description(issue, patch)
        
        # 生成测试说明
        test_notes = self.generate_test_notes(patch)
        
        return {
            "title": title,
            "description": description,
            "patch": patch,
            "test_notes": test_notes,
            "labels": self.suggest_labels(issue)
        }
    
    def generate_description(self, issue: dict, patch: str) -> str:
        """生成 PR 描述"""
        return f"""## 问题描述

{issue['description']}

## 12. 解决方案

{self.summarize_changes(patch)}

## 13. 修改的文件

{self.list_changed_files(patch)}

## 14. 测试

{self.describe_tests(patch)}
"""
```

#### 代码质量

```python
class CodeQualityEvaluator:
    """代码质量评估器"""
    
    def evaluate_patch(self, patch: str) -> dict:
        """评估补丁质量"""
        metrics = {
            "correctness": self.evaluate_correctness(patch),
            "readability": self.evaluate_readability(patch),
            "maintainability": self.evaluate_maintainability(patch),
            "test_coverage": self.evaluate_test_coverage(patch),
            "performance": self.evaluate_performance_impact(patch)
        }
        
        # 综合评分
        weights = {
            "correctness": 0.3,
            "readability": 0.2,
            "maintainability": 0.2,
            "test_coverage": 0.2,
            "performance": 0.1
        }
        
        overall_score = sum(
            metrics[metric] * weights[metric] 
            for metric in metrics
        )
        
        return {
            "overall_score": overall_score,
            "metrics": metrics
        }
    
    def evaluate_correctness(self, patch: str) -> float:
        """评估正确性"""
        # 运行测试套件
        test_results = run_tests(patch)
        return test_results["pass_rate"]
    
    def evaluate_readability(self, patch: str) -> float:
        """评估可读性"""
        # 使用代码分析工具
        metrics = analyze_code_metrics(patch)
        return normalize_score(metrics)
```

### B.5 自定义评估框架

#### 评估指标设计

```python
from dataclasses import dataclass
from typing import List

@dataclass
class EvaluationMetric:
    """评估指标"""
    name: str
    description: str
    weight: float
    evaluation_function: callable

@dataclass
class EvaluationResult:
    """评估结果"""
    agent_name: str
    task_name: str
    metric_scores: dict
    overall_score: float
    feedback: str

class CustomEvaluationFramework:
    """自定义评估框架"""
    
    def __init__(self, metrics: List[EvaluationMetric]):
        self.metrics = metrics
        self.results = []
    
    def evaluate(self, agent, task) -> EvaluationResult:
        """执行评估"""
        # 运行任务
        result = agent.run(task)
        
        # 计算各项指标得分
        metric_scores = {}
        for metric in self.metrics:
            score = metric.evaluation_function(result, task)
            metric_scores[metric.name] = score
        
        # 计算加权总分
        overall_score = sum(
            metric_scores[m.name] * m.weight 
            for m in self.metrics
        )
        
        evaluation_result = EvaluationResult(
            agent_name=agent.name,
            task_name=task.name,
            metric_scores=metric_scores,
            overall_score=overall_score,
            feedback=self.generate_feedback(metric_scores)
        )
        
        self.results.append(evaluation_result)
        return evaluation_result
    
    def generate_feedback(self, scores: dict) -> str:
        """生成反馈"""
        strengths = [k for k, v in scores.items() if v > 0.8]
        weaknesses = [k for k, v in scores.items() if v < 0.5]
        
        feedback = "优势：" + ", ".join(strengths) if strengths else ""
        if weaknesses:
            feedback += "\n需改进：" + ", ".join(weaknesses)
        
        return feedback
```

#### 测试用例编写

```python
class TestCase:
    """测试用例"""
    
    def __init__(self, name: str, description: str, expected_output: str):
        self.name = name
        self.description = description
        self.expected_output = expected_output
        self.difficulty = "medium"
        self.category = "general"
    
    def evaluate(self, actual_output: str) -> float:
        """评估输出"""
        # 精确匹配
        if actual_output == self.expected_output:
            return 1.0
        
        # 语义相似度
        similarity = calculate_semantic_similarity(
            actual_output, 
            self.expected_output
        )
        
        return similarity

# 测试用例集
test_suite = [
    TestCase(
        name="简单问答",
        description="回答基础事实问题",
        expected_output="北京是中国的首都"
    ),
    TestCase(
        name="工具使用",
        description="正确使用工具查询信息",
        expected_output="根据查询结果，..."
    ),
    TestCase(
        name="多步推理",
        description="执行多步骤推理任务",
        expected_output="经过分析，..."
    )
]
```

#### 自动化评估

```python
class AutomatedEvaluator:
    """自动化评估器"""
    
    def __init__(self, test_suite: list, metrics: list):
        self.test_suite = test_suite
        self.metrics = metrics
    
    async def run_full_evaluation(self, agents: list) -> dict:
        """运行完整评估"""
        all_results = {}
        
        for agent in agents:
            agent_results = []
            
            for test_case in self.test_suite:
                # 运行测试
                result = await agent.run_async(test_case.description)
                
                # 评估结果
                score = test_case.evaluate(result)
                
                agent_results.append({
                    "test_case": test_case.name,
                    "score": score,
                    "result": result
                })
            
            # 计算总分
            avg_score = sum(r["score"] for r in agent_results) / len(agent_results)
            
            all_results[agent.name] = {
                "average_score": avg_score,
                "test_results": agent_results
            }
        
        return all_results
    
    def generate_report(self, results: dict) -> str:
        """生成评估报告"""
        report = "# Agent 评估报告\n\n"
        
        for agent_name, agent_results in results.items():
            report += f"## {agent_name}\n\n"
            report += f"**平均分**: {agent_results['average_score']:.2%}\n\n"
            
            report += "| 测试用例 | 得分 |\n"
            report += "|---------|------|\n"
            
            for test_result in agent_results["test_results"]:
                report += f"| {test_result['test_case']} | {test_result['score']:.2%} |\n"
            
            report += "\n"
        
        return report
```

#### 持续监控

```python
import asyncio
from datetime import datetime

class ContinuousMonitor:
    """持续监控器"""
    
    def __init__(self, agents: list, test_suite: list, interval: int = 3600):
        self.agents = agents
        self.test_suite = test_suite
        self.interval = interval  # 秒
        self.history = []
    
    async def start_monitoring(self):
        """开始监控"""
        while True:
            # 运行评估
            results = await self.run_evaluation()
            
            # 记录结果
            self.history.append({
                "timestamp": datetime.now().isoformat(),
                "results": results
            })
            
            # 检查性能下降
            if self.detect_performance_degradation():
                self.send_alert()
            
            # 等待下次评估
            await asyncio.sleep(self.interval)
    
    def detect_performance_degradation(self) -> bool:
        """检测性能下降"""
        if len(self.history) < 2:
            return False
        
        current = self.history[-1]["results"]
        previous = self.history[-2]["results"]
        
        # 检查是否有显著下降
        for agent_name in current:
            current_score = current[agent_name]["average_score"]
            previous_score = previous[agent_name]["average_score"]
            
            if previous_score - current_score > 0.1:  # 下降超过 10%
                return True
        
        return False
    
    def send_alert(self):
        """发送警报"""
        print("⚠️ 检测到 Agent 性能下降！")
        # 可以集成邮件、Slack 等通知
    
    def generate_trend_report(self) -> str:
        """生成趋势报告"""
        report = "# Agent 性能趋势\n\n"
        
        for agent_name in self.agents:
            scores = [
                h["results"][agent_name]["average_score"]
                for h in self.history
                if agent_name in h["results"]
            ]
            
            report += f"## {agent_name}\n\n"
            report += f"- 当前分数: {scores[-1]:.2%}\n"
            report += f"- 平均分数: {sum(scores)/len(scores):.2%}\n"
            report += f"- 最高分数: {max(scores):.2%}\n"
            report += f"- 最低分数: {min(scores):.2%}\n\n"
        
        return report
```
