# crewai - 多 Agent 角色协作框架学习指南

## 项目概述

**一句话总结**: CrewAI 是一个轻量、快速的 Python 多 Agent 编排框架，让你像组建真实团队一样，让 AI Agent 们分工协作完成复杂任务。

**核心亮点**:
- **角色驱动**: 每个 Agent 都有明确的角色（role）、目标（goal）和背景故事（backstory），就像给团队成员写岗位描述
- **双引擎架构**: 提供 **Crews（团队协作）** 和 **Flows（流程编排）** 两种模式，既可让 Agent 自主决策，又能精确控制执行流程
- **零外部依赖**: 完全独立开发，不依赖 LangChain 等框架，性能更快、更轻量
- **生产就绪**: 支持日志、监控、持久化、断点续跑等企业级特性

**适合谁学**:
- 想构建多 Agent 协作系统的开发者
- 需要自动化复杂工作流（如调研→分析→报告）的工程师
- 对 Agent 编排感兴趣的 AI 应用开发者

---

## 核心架构解析

CrewAI 的架构可以用一个公司来类比：

```
Flow（流程） = 总经理，负责整体调度和状态管理
  └── Crew（团队） = 部门经理，组织一群专家完成特定任务
        ├── Agent（员工） = 专业人才，有明确岗位职责
        └── Task（任务） = 具体工作项，有输入输出要求
```

### 1. Agent 角色定义模式

每个 Agent 就像一位员工，需要三个核心要素：

```yaml
# 配置文件: config/agents.yaml
researcher:
  role: >           # 职位头衔
    {topic} 高级数据研究员
  goal: >           # KPI/目标
    挖掘{topic}领域的前沿发展
  backstory: >      # 个人履历/性格
    你是一位资深研究员，擅长发现最新趋势...
```

**为什么这样设计**: 
- `role` 决定 Agent 的"专业领域"，影响其决策倾向
- `goal` 是任务的"北极星指标"，Agent 会围绕它工作
- `backstory` 赋予 Agent "人格特质"，让输出更符合角色定位

**关键参数**（源码 `src/crewai/agent.py`）:
- `tools`: Agent 能使用的工具（如搜索引擎、数据库）
- `allow_delegation`: 是否允许委派任务给其他 Agent
- `memory`: 是否记住历史交互
- `max_iter`: 最大尝试次数（默认 20 次）

### 2. Crew（团队）组织方式

Crew 就是一支团队，把 Agent 和 Task 组装起来：

```python
# 文件: src/my_project/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase  # 装饰器：标记这是一个 Crew 类
class ResearchCrew:
    agents_config = "config/agents.yaml"  # Agent 配置路径
    tasks_config = "config/tasks.yaml"    # Task 配置路径

    @agent  # 装饰器：自动注册 Agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[SerperDevTool()]  # 给研究员配搜索工具
        )

    @task  # 装饰器：自动注册 Task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config['research_task'])

    @crew  # 装饰器：返回完整的 Crew 对象
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,   # @agent 装饰器自动收集
            tasks=self.tasks,     # @task 装饰器自动收集
            process=Process.sequential,  # 执行流程
            verbose=True
        )
```

**装饰器魔法**: `@CrewBase` + `@agent`/`@task`/`@crew` 让框架自动收集所有 Agent 和 Task，你不用手动管理列表。

### 3. Task（任务）与依赖管理

Task 是具体的工作项，定义"做什么"和"交付什么"：

```yaml
# 配置文件: config/tasks.yaml
research_task:
  description: >
    深入调研{topic}，使用网络搜索找到当前、可靠的信息
  expected_output: >
    一份包含关键趋势、知名公司、行业影响的 Markdown 报告
  agent: researcher          # 指派给哪个 Agent
  output_file: report.md     # 结果保存到文件

analysis_task:
  description: >
    分析调研结果，提炼核心观点
  expected_output: >
    包含 5 个关键洞察的分析报告
  agent: analyst
  context: [research_task]   # 依赖：等 research_task 完成后才开始
```

**任务依赖的两种方式**:
1. **顺序依赖**: `process=Process.sequential` 按 Task 定义顺序执行
2. **显式依赖**: 用 `context=[task1, task2]` 指定前置任务

### 4. 执行流程（顺序/并行/层级）

CrewAI 提供三种执行模式：

| 模式 | 类比 | 适用场景 |
|------|------|----------|
| **Sequential（顺序）** | 流水线作业，一个接一个 | 调研→写作→审校 |
| **Hierarchical（层级）** | 经理分配任务，审核结果 | 复杂项目需要统筹协调 |
| **异步并行** | 多人同时开工，最后汇总 | 独立子任务可并行处理 |

**层级模式示例**:
```python
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, write_task],
    process=Process.hierarchical,
    manager_llm="gpt-4o"  # 需要一个"经理"来分配和审核任务
)
```

---

## 代码逻辑主线

### Crew 运行流程

```
1. kickoff(inputs={...})          # 启动 Crew，传入变量（如 topic）
   ↓
2. 替换模板变量                    # {topic} → "AI Agents"
   ↓
3. 按 Process 执行 Task 列表      # 顺序/层级/异步
   ↓
4. 每个 Agent 执行自己的 Task     # 调用 LLM + Tools
   ↓
5. 收集输出 → CrewOutput          # 包含 raw/pydantic/json 格式
   ↓
6. 可选：保存到 output_file       # 如 report.md
```

**关键类与方法**（源码路径）:
- `crewai.agent.Agent` - Agent 核心类（`src/crewai/agent.py`）
- `crewai.task.Task` - Task 核心类（`src/crewai/task.py`）
- `crewai.crew.Crew` - Crew 编排器（`src/crewai/crew.py`）
- `crew.kickoff()` - 启动执行（同步）
- `crew.akickoff()` - 启动执行（原生异步）
- `crew.kickoff_for_each()` - 批量执行多个输入

### 多 Agent 通信机制

Agent 之间不直接"对话"，而是通过 **Task 输出** 传递信息：

```
Agent A 完成 Task 1
   ↓
TaskOutput (包含 raw 文本/pydantic 对象)
   ↓
作为 Task 2 的 context 传入
   ↓
Agent B 读取 context，开始 Task 2
```

**示例**:
```python
# Task A 的输出自动成为 Task B 的上下文
research_task = Task(description="调研 AI 发展", agent=researcher)
write_task = Task(
    description="写报告",
    agent=writer,
    context=[research_task]  # 等待 research_task 的输出
)
```

### Flows：更精细的流程控制

如果说 Crew 是"团队协作"，那 Flow 就是"总经理调度"。Flow 用事件驱动的方式编排多个 Crew 或普通代码：

```python
# 文件: src/my_project/main.py
from crewai.flow import Flow, listen, start
from pydantic import BaseModel

class ResearchState(BaseModel):
    topic: str = ""
    report: str = ""

class ResearchFlow(Flow[ResearchState]):
    @start()  # 入口点
    def prepare_topic(self):
        self.state.topic = "AI Agents"

    @listen(prepare_topic)  # 监听上一个步骤
    def run_research(self):
        # 调用 Crew 完成调研
        result = ResearchCrew().crew().kickoff(
            inputs={"topic": self.state.topic}
        )
        self.state.report = result.raw

    @listen(run_research)  # 继续监听
    def save_report(self):
        print(f"报告已生成: {self.state.report}")

# 启动 Flow
ResearchFlow().kickoff()
```

**Flow 的核心能力**:
- `@start()` - 定义入口
- `@listen(method)` - 监听某个方法完成
- `@router(method)` - 条件分支（if-else）
- `or_()` / `and_()` - 组合多个监听条件
- `self.state` - 跨步骤共享状态（支持 Pydantic 类型安全）

---

## 快速上手实践

### 环境配置步骤

```bash
# 1. 确保 Python 版本 >=3.10 <3.14
python --version

# 2. 安装 crewai
uv pip install crewai

# 3. 安装额外工具（可选，含搜索工具等）
uv pip install 'crewai[tools]'

# 4. 创建新项目
crewai create crew my_research_project
cd my_research_project

# 5. 配置环境变量
# 编辑 .env 文件，填入：
OPENAI_API_KEY=sk-...        # 或其他 LLM 的 API Key
SERPER_API_KEY=your_key      # 从 serper.dev 获取（用于网络搜索）
```

### 运行第一个示例

项目结构已自动生成：
```
my_research_project/
├── .env
├── src/
│   └── my_research_project/
│       ├── main.py          # 入口文件
│       ├── crew.py          # Crew 定义
│       ├── config/
│       │   ├── agents.yaml  # Agent 配置
│       │   └── tasks.yaml   # Task 配置
│       └── tools/
│           └── custom_tool.py
```

**修改配置**（以调研 AI 趋势为例）:

编辑 `config/agents.yaml`:
```yaml
researcher:
  role: >
    {topic} 高级研究员
  goal: >
    发现{topic}领域的最新趋势
  backstory: >
    你是一位擅长挖掘前沿信息的研究员
```

编辑 `config/tasks.yaml`:
```yaml
research_task:
  description: >
    调研{topic}的最新发展，使用网络搜索
  expected_output: >
    一份包含关键趋势的 Markdown 报告
  agent: researcher
  output_file: output/report.md
```

**运行项目**:
```bash
# 安装依赖（可选）
crewai install

# 启动 Crew
crewai run

# 或直接运行 Python
python src/my_research_project/main.py
```

### 预期输出与验证方法

**控制台输出**（verbose=True 时）:
```
[researcher]: 开始调研 AI Agents...
[researcher]: 使用 SerperDevTool 搜索...
[researcher]: 找到 10 条相关信息
[researcher]: 任务完成
```

**验证结果**:
1. 检查 `output/report.md` 文件是否生成
2. 打开文件，应该看到结构化的调研报告
3. 如果使用了网络搜索工具，内容应该是最新的

**排错提示**:
- 报 `ModuleNotFoundError: No module named 'tiktoken'` → 运行 `uv pip install 'crewai[embeddings]'`
- API Key 错误 → 检查 `.env` 文件是否正确配置
- 任务卡住 → 检查 Agent 的 `max_iter` 是否过小

---

## 核心知识点总结

### 必须掌握的 8 个概念

#### 1. Agent 三要素（role/goal/backstory）
- **是什么**: 定义 Agent 的"身份画像"
- **为什么重要**: 决定了 Agent 的决策倾向和输出风格。没有明确的 backstory，Agent 就像没有经验的员工

#### 2. Crew 与 Process
- **是什么**: Crew 是团队，Process 是工作方式（顺序/层级）
- **为什么重要**: 选择错误的 Process 会导致任务执行混乱。简单任务用 Sequential，复杂协调用 Hierarchical

#### 3. Task 的 context 依赖
- **是什么**: Task 可以等待其他 Task 的输出作为输入
- **为什么重要**: 这是多 Agent 协作的核心机制。没有 context，Agent 就是孤岛

#### 4. YAML 配置 vs 代码定义
- **是什么**: 两种定义 Agent/Task 的方式
- **为什么重要**: YAML 更易维护，适合配置变更频繁的场景；代码更灵活，适合动态逻辑

#### 5. Tools 工具集成
- **是什么**: Agent 可调用的外部能力（搜索、数据库、API）
- **为什么重要**: 没有 Tools 的 Agent 只能"空谈"。Tools 让 Agent 能与真实世界交互

#### 6. Flow 状态管理
- **是什么**: Flow 用 `self.state` 在步骤间传递数据
- **为什么重要**: 相比 Crew 的"黑盒"执行，Flow 让你精确控制每一步的输入输出

#### 7. 结构化输出（output_pydantic）
- **是什么**: 用 Pydantic 模型约束 Task 输出格式
- **为什么重要**: 保证下游任务拿到的是"规整数据"而不是"自由文本"，适合需要程序化处理的场景

#### 8. 断点续跑（Checkpoint）
- **是什么**: 保存 Crew 执行状态，中断后可恢复
- **为什么重要**: 长流程任务（如调研 100 篇文献）中断后不用从头开始

---

## 常见疑问解答

### Q1: CrewAI 和 LangChain 的 LangGraph 有什么区别？

**通俗解释**:
- LangGraph 像"乐高积木"，灵活但需要自己拼装每个零件
- CrewAI 像"预制房屋"，内置了房间布局（Agent 角色、团队协作），拎包入住

**关键差异**:
- CrewAI 有明确的"角色"概念（role/goal/backstory），LangGraph 更偏底层状态机
- CrewAI 的 YAML 配置让非程序员也能定义 Agent
- 官方基准测试显示 CrewAI 在某些任务上快 5.76 倍

### Q2: 什么时候用 Crew，什么时候用 Flow？

**经验法则**:
- **单团队完成一个目标** → 用 Crew（如"调研并写报告"）
- **多阶段、需要条件判断** → 用 Flow（如"先调研，如果置信度>0.8 则写报告，否则继续调研"）
- **生产环境** → 推荐 Flow + Crew 组合，Flow 管调度，Crew 管执行

### Q3: Agent 之间怎么"对话"？

**重要认知**: Agent 不直接聊天！它们通过 **Task 输出** 间接协作：
```
Agent A → 完成 Task 1 → 输出文本 → 作为 Task 2 的 context → Agent B 读取
```

如果需要 Agent 实时交互，应该启用 `allow_delegation=True`，让 Agent 可以委派任务给对方。

### Q4: 可以用本地模型（如 Ollama）吗？

**完全可以**。在创建 Agent 时指定 `llm` 参数：
```python
agent = Agent(
    role="研究员",
    llm="ollama/llama3"  # 使用本地 Ollama 模型
)
```

详见官方文档 LLM Connections 章节。

### Q5: 如何调试 Agent 的"思考过程"？

**三种方法**:
1. **设置 `verbose=True`**: 打印每个步骤的日志
2. **查看 `output_log_file`**: 保存完整执行日志到文件
3. **使用 `crewai log-tasks-outputs`**: CLI 查看任务输出历史

---

## 延伸学习路径

1. **官方示例库**: https://github.com/crewAIInc/crewAI-examples
   - Trip Planner（旅行规划器）
   - Stock Analysis（股票分析）
   - Landing Page Generator（落地页生成）

2. **官方文档**: https://docs.crewai.com
   - 重点阅读: Concepts → Agents/Crews/Tasks/Flows

3. **进阶主题**:
   - Memory 系统（Agent 如何记住历史）
   - Guardrails（任务输出校验）
   - Human-in-the-loop（人工审核环节）
   - 部署到 CrewAI AMP（企业云平台）

---

**总结**: CrewAI 的核心理念是"**让 AI 像人类团队一样工作**"。通过明确的角色定义、灵活的流程编排、强大的工具集成，你可以快速构建出从简单自动化到复杂决策的多 Agent 系统。掌握 Agent、Task、Crew、Flow 这四个核心概念，你就已经具备了使用 CrewAI 构建生产级应用的能力。
