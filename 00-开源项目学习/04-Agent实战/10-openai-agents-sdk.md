# openai-agents-sdk - OpenAI 官方 Agent SDK 学习指南

## 项目概述

**一句话总结**：OpenAI 官方出品的轻量级多 Agent 开发框架，让你用简洁的 Python 代码构建可协作、可扩展的智能体系统。

### 核心亮点

1. **官方出品，权威可靠**：由 OpenAI 团队开发维护，与 OpenAI API 深度集成，确保最佳兼容性
2. **轻量但强大**：相比 LangChain 等重型框架，代码量更少，学习曲线更平缓，但覆盖了多 Agent 协作的核心能力
3. **类型安全**：基于 Python 类型注解和 Pydantic，开发时有完整的 IDE 智能提示和编译时检查
4. **多模型支持**：默认支持 OpenAI Responses API，同时兼容 100+ 其他 LLM 提供商
5. **内置可观测性**：自带 Tracing 追踪系统，可视化查看 Agent 运行流程，方便调试和优化

### 适合谁学

- 想用 OpenAI 模型构建多 Agent 应用的 Python 开发者
- 对 Agent 编排、工具调用、人机协作感兴趣的学习者
- 希望理解生产级 Agent 架构设计的工程师

---

## 核心架构解析

OpenAI Agents SDK 的设计哲学可以用一个比喻来理解：**就像组建一个专业团队**。

### 1. Agent（智能体）- 团队中的专业人士

每个 Agent 就像一个有特定技能的员工，你可以给它配置：

```python
from agents import Agent, function_tool

@function_tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"{city} 的天气是晴天"

agent = Agent(
    name="天气助手",
    instructions=" Always respond in haiku form（总是用俳句形式回答）",
    model="gpt-5-nano",
    tools=[get_weather],
)
```

**关键配置项**：

| 属性 | 作用 | 比喻 |
|------|------|------|
| `name` | Agent 名称 | 员工的名字 |
| `instructions` | 系统提示词 | 岗位职责说明书 |
| `tools` | 可调用的工具 | 工作中能用的设备/权限 |
| `model` | 使用的 LLM | 员工的大脑型号 |
| `handoffs` | 可交接的对象 | 可以转接给哪些同事 |
| `guardrails` | 安全检查 | 质量审查员 |

源码位置：[`src/agents/agent.py`](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/04-Agent实战/openai-agents-sdk/src/agents/agent.py)

---

### 2. 工具注册机制 - 给 Agent 配备武器

工具是 Agent 执行实际操作的能力。SDK 支持 **5 类工具**：

#### ① 函数工具（最常用）

用 `@function_tool` 装饰器将普通 Python 函数变成 Agent 可调用的工具：

```python
@function_tool
async def fetch_weather(location: dict) -> str:
    """获取指定位置的天气"""
    return "晴天"
```

**自动完成的工作**：
- 从函数签名提取参数类型 → 生成 JSON Schema
- 从 docstring 提取描述 → 告诉 LLM 这个工具的作用
- 支持同步/异步函数
- 支持可选的 `RunContextWrapper` 参数获取运行时上下文

源码位置：[`src/agents/tool.py`](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/04-Agent实战/openai-agents-sdk/src/agents/tool.py#L244-L380)

#### ② 托管工具（OpenAI 内置）

OpenAI 服务器直接运行的工具，无需你自己实现：

```python
from agents import WebSearchTool, FileSearchTool, CodeInterpreterTool

agent = Agent(
    name="研究助手",
    tools=[
        WebSearchTool(),           # 网络搜索
        FileSearchTool(...),       # 向量存储检索
        CodeInterpreterTool(),     # 沙箱代码执行
    ],
)
```

#### ③ Agent 作为工具

把一个 Agent 当成另一个 Agent 的工具来调用，**不转移控制权**：

```python
spanish_agent = Agent(name="西班牙语翻译", instructions="翻译成西班牙语")

orchestrator = Agent(
    name="协调者",
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="将用户消息翻译成西班牙语",
        ),
    ],
)
```

这就像团队中的专家被项目经理"借用"，但项目主导权还在经理手里。

#### ④ 本地运行工具

在你的环境中执行实际操作的工具：
- `ComputerTool`：GUI/浏览器自动化
- `ShellTool`：执行 Shell 命令
- `ApplyPatchTool`：应用代码补丁

#### ⑤ MCP 工具

通过 Model Context Protocol 连接外部工具服务器。

---

### 3. Handoff 协议 - 智能体间的交接棒

**Handoff** 是 SDK 最核心的多 Agent 协作机制。想象医院的分诊台：前台护士（Triage Agent）根据病情把病人转给不同专科医生（Specialist Agent）。

```python
from agents import Agent, handoff

# 专科 Agent
billing_agent = Agent(name="账单专员", handoff_description="处理账单问题")
refund_agent = Agent(name="退款专员", handoff_description="处理退款请求")

# 分诊 Agent
triage_agent = Agent(
    name="分诊员",
    instructions="帮助用户解决问题。如果询问账单，转给账单专员；如果询问退款，转给退款专员。",
    handoffs=[billing_agent, refund_agent],  # 可交接的对象
)
```

**Handoff 的工作原理**：
1. Handoff 在 LLM 看来就是一个**特殊工具**，例如 `transfer_to_billing_agent`
2. 当 LLM 决定调用这个"工具"时，SDK 拦截调用，将控制权移交给目标 Agent
3. 目标 Agent **继承对话历史**，无缝继续对话
4. 整个过程在**同一次 Runner.run()** 调用内完成

**高级定制**：

```python
from pydantic import BaseModel
from agents import handoff, RunContextWrapper

class EscalationData(BaseModel):
    reason: str  # 升级原因

async def on_handoff(ctx: RunContextWrapper, input_data: EscalationData):
    print(f"升级原因: {input_data.reason}")

handoff_obj = handoff(
    agent=escalation_agent,
    on_handoff=on_handoff,      # 交接时执行的回调
    input_type=EscalationData,  # 要求 LLM 提供结构化数据
)
```

源码位置：[`src/agents/handoffs/`](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/04-Agent实战/openai-agents-sdk/src/agents/handoffs/)

---

### 4. Guardrails（守护函数）- 安全护栏

Guardrails 就像机场安检，在 Agent 运行前后进行安全检查：

```python
from agents import Agent, input_guardrail, GuardrailFunctionOutput

@input_guardrail
async def math_checker(ctx, agent, input: str) -> GuardrailFunctionOutput:
    """检查用户是否在让 Agent 做数学作业"""
    # 用一个小 Agent 来判断
    result = await Runner.run(guard_agent, input)
    return GuardrailFunctionOutput(
        tripwire_triggered=result.final_output.is_math_homework,
    )

agent = Agent(
    name="客服助手",
    input_guardrails=[math_checker],  # 输入检查
    output_guardrails=[...],           # 输出检查
)
```

**两种执行模式**：
- **并行模式**（默认）：Guardrail 和 Agent 同时运行，延迟最低，但如果触发则浪费 token
- **阻塞模式**：Guardrail 先运行，通过后 Agent 才开始，节省成本

**Guardrails 运行边界**：
- `input_guardrails` 只在**第一个 Agent** 上运行
- `output_guardrails` 只在**最后一个 Agent** 上运行
- 如果需要检查每个工具调用，使用 `tool_input_guardrails` 和 `tool_output_guardrails`

源码位置：[`src/agents/guardrail.py`](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/04-Agent实战/openai-agents-sdk/src/agents/guardrail.py)

---

## 代码逻辑主线

### Agent 运行流程

整个 SDK 的核心执行引擎是 `Runner`，它的工作流程如下：

```
用户输入 → Runner.run() → 输入 Guardrails 检查 → 调用 LLM
    ↓
LLM 返回结果 → 是否有工具调用？
    ├─ 是 → 执行工具 → 结果返回 LLM → 再次调用 LLM → ...
    └─ 否 → 是否是 Handoff？
        ├─ 是 → 切换目标 Agent → 继续对话
        └─ 否 → 输出 Guardrails 检查 → 返回最终结果
```

关键类与方法：

| 类/方法 | 作用 | 源码位置 |
|---------|------|----------|
| `Runner.run()` | 异步运行 Agent | `src/agents/run.py` |
| `Runner.run_sync()` | 同步运行 Agent | `src/agents/run.py` |
| `Runner.run_streamed()` | 流式运行 Agent | `src/agents/run.py` |
| `Agent` | Agent 定义 | `src/agents/agent.py` |
| `RunConfig` | 运行配置（最大轮数等） | `src/agents/run_config.py` |
| `RunResult` | 运行结果 | `src/agents/result.py` |

---

### 多 Agent 协作机制

SDK 提供两种多 Agent 协作模式：

#### 模式 1：Handoff（去中心化）

Agent 之间平级交接，像接力赛：

```
用户 → Triage Agent → Billing Agent → 输出结果
```

特点：
- 每个 Agent 独立运作
- 交接后控制权完全转移
- 适合模块化、专业化场景

#### 模式 2：Agents as Tools（中心化编排）

一个主 Agent 协调多个子 Agent，像项目经理：

```
用户 → Orchestrator Agent
              ↓ (调用工具)
       ├─ Spanish Agent
       ├─ French Agent
       └─ 汇总结果 → 输出
```

特点：
- 主 Agent 保持控制权
- 子 Agent 像工具一样被调用
- 适合需要统一管理和结果整合的场景

**选择建议**：
- 需要**模块化**和**独立性** → 用 Handoff
- 需要**集中控制**和**结果整合** → 用 Agents as Tools

---

## 快速上手实践

### 环境配置步骤

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. 安装 SDK
pip install openai-agents

# 3. 设置 API Key
export OPENAI_API_KEY="your-api-key-here"
```

### 运行第一个示例

创建文件 `hello_agent.py`：

```python
from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city} 的天气晴朗，温度 25°C"

agent = Agent(
    name="天气助手",
    instructions="你是友好的天气助手，用简洁的方式回答天气问题。",
    model="gpt-4o",
    tools=[get_weather],
)

result = Runner.run_sync(agent, "北京天气怎么样？")
print(result.final_output)
```

运行：

```bash
python hello_agent.py
```

### 预期输出与验证方法

**预期输出**：
```
北京的天气晴朗，温度 25°C，非常适合外出活动！
```

**验证方法**：
1. 检查是否成功调用 `get_weather` 工具（可通过 Tracing 查看）
2. 确认输出包含天气信息
3. 尝试不同城市名验证工具参数传递

**查看运行追踪**：
```python
from agents import Runner

result = await Runner.run(agent, "上海天气？")
# 在 OpenAI 平台的 Tracing UI 中查看运行详情
```

---

## 核心知识点总结

### 1. Agent 是核心抽象
- **是什么**：Agent = LLM + 指令 + 工具 + 配置
- **为什么重要**：整个 SDK 围绕 Agent 构建，理解 Agent 是掌握框架的第一步

### 2. 工具是 Agent 的能力延伸
- **是什么**：工具让 Agent 能执行实际操作（查询 API、运行代码等）
- **为什么重要**：没有工具的 Agent 只能聊天，有了工具才能"做事"

### 3. Handoff 实现多 Agent 协作
- **是什么**：Agent 之间交接控制权的机制
- **为什么重要**：让复杂任务可以分解给多个专业 Agent 处理

### 4. Runner 是执行引擎
- **是什么**：负责调度 LLM 调用、工具执行、Handoff 转移的运行时
- **为什么重要**：你定义好 Agent 后，Runner 负责"让一切跑起来"

### 5. Guardrails 保障安全
- **是什么**：输入输出的自动检查机制
- **为什么重要**：防止恶意输入、不当输出，在生产环境中必不可少

### 6. Context 是依赖注入
- **是什么**：在 Agent 运行全程传递的状态和依赖对象
- **为什么重要**：让工具、Handoff、Hook 都能访问用户信息、数据库连接等

### 7. Hook 监听生命周期
- **是什么**：在 Agent 运行各阶段（开始、LLM 调用、工具调用等）触发的回调
- **为什么重要**：用于日志记录、性能监控、预取数据等副作用操作

### 8. Structured Output 结构化输出
- **是什么**：让 Agent 返回 Pydantic 模型而非纯文本
- **为什么重要**：程序化处理结果，避免文本解析的脆弱性

---

## 常见疑问解答

### Q1: Handoff 和 Agent as Tool 有什么区别？我该用哪个？

**关键区别**：
- **Handoff**：交接后**控制权转移**，目标 Agent 直接对话用户
- **Agent as Tool**：主 Agent **保持控制权**，子 Agent 的返回结果给主 Agent 处理

**选择原则**：
- 如果子 Agent 需要**直接与用户交互** → 用 Handoff
- 如果子 Agent 是**幕后专家**，结果需要整合 → 用 Agent as Tool

---

### Q2: 工具调用后 LLM 会无限循环怎么办？

SDK **自动处理**了这个问题：
- 每次工具调用后，`tool_choice` 自动重置为 `"auto"`
- 防止 LLM 持续调用同一个工具

你也可以通过 `agent.reset_tool_choice = False` 自定义此行为。

---

### Q3: Guardrails 触发后会发生什么？

当 Guardrail 触发 tripwire 时：
1. 抛出 `InputGuardrailTripwireTriggered` 或 `OutputGuardrailTripwireTriggered` 异常
2. Agent 运行**立即停止**
3. 你可以在 `try/except` 中捕获异常，给用户返回适当提示

```python
try:
    result = await Runner.run(agent, user_input)
except InputGuardrailTripwireTriggered:
    print("输入安全检查未通过，请重新输入")
```

---

### Q4: 如何调试 Agent 的运行过程？

三种方法：
1. **Tracing**：SDK 内置，在 OpenAI 平台可视化查看每一步
2. **Hooks**：自定义日志，在每个阶段打印信息
3. **RunResult**：检查 `result.new_items` 查看所有中间产物

---

### Q5: 可以使用非 OpenAI 模型吗？

**可以**。SDK 支持 100+ 模型提供商，通过配置不同的 Model Provider：

```python
# 示例：使用其他提供商
from agents import Agent

agent = Agent(
    name="助手",
    model="your-provider/your-model",  # 替换为你的模型标识
    instructions="...",
)
```

详细配置见文档 `docs/models/` 目录。

---

## 学习路径建议

1. **第一步**：运行 `examples/basic/` 中的简单示例，熟悉 Agent 和工具
2. **第二步**：尝试 Handoff 示例，理解多 Agent 协作
3. **第三步**：学习 Guardrails 和 Hooks，掌握安全和可观测性
4. **第四步**：探索 `examples/agent_patterns/` 中的高级模式
5. **第五步**：阅读源码 `src/agents/agent.py` 和 `src/agents/run.py`，深入理解架构

**官方文档**：https://openai.github.io/openai-agents-python/
**示例代码**：`examples/` 目录
