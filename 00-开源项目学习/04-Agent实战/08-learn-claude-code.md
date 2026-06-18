# learn-claude-code - Harness Engineering 学习指南

## 项目概述

**一句话总结价值**：learn-claude-code 深度解析 Claude Code 源码，教你理解生产级 Agent Harness（执行框架）的设计原理，是理解"智能体能力来自模型，产品需要执行环境"这一核心理念的最佳实践。

**核心亮点**：
- ✅ **15 个核心模块** - 从 Agent 循环到子 Agent 调度，完整拆解 Claude Code 架构
- ✅ **Harness Engineering 理念** - 理解"模型提供智能，执行环境提供操作空间"的设计哲学
- ✅ **生产级代码示例** - 每个模块都有可运行的 Python 实现，不是理论讲解
- ✅ **系统化学习路径** - s01→s15 逐步进阶，覆盖 Agent 产品所有关键技术点

**适合谁学**：
- 想深入理解 Claude Code 架构和源码的开发者
- 想学习如何构建生产级 Agent 产品的工程师
- 对"模型 vs 执行环境"设计哲学感兴趣的学习者

---

## 核心架构解析

### Harness Engineering 核心理念

```
Agent 产品 = 模型（Model） + 执行环境（Harness）

模型（Driver）:
  - 感知、推理、决策能力来自训练
  - 决定做什么、怎么做

执行环境（Vehicle）:
  - 工具（Tools）: 文件读写、Shell、网络、数据库、浏览器
  - 知识（Knowledge）: 产品文档、领域参考、API 规范
  - 观察（Observation）: git diff、错误日志、浏览器状态
  - 行动（Action）: CLI 命令、API 调用、UI 交互
  - 权限（Permissions）: 沙箱隔离、审批流程、信任边界

模型决定，环境执行。
```

### Claude Code 架构拆解

```
Claude Code = 1 个 Agent 循环
            + 工具系统（bash, read, write, edit, glob, grep, browser...）
            + 按需技能加载（Skill Loading）
            + 上下文压缩（Context Compaction）
            + 子 Agent 调度（Subagent Spawning）
            + 任务系统与依赖图（Task System）
            + 异步邮箱团队协作（Async Mailbox）
            + 工作树隔离并行执行（Worktree-Isolated）
            + 权限治理（Permission Governance）
```

### 15 个模块全景

| 模块 | 目录 | 核心内容 | 学习价值 |
|------|------|---------|---------|
| **s01** | s01_agent_loop | Agent 主循环设计 | 理解"接收→推理→执行→观察"循环 |
| **s02** | s02_tool_use | 工具调用机制 | 学习如何给模型"双手" |
| **s03** | s03_permission | 权限控制系统 | 沙箱、审批、信任边界 |
| **s04** | s04_hooks | 钩子与事件系统 | 扩展点与生命周期 |
| **s05** | s05_todo_write | 任务管理 | 目标持久化与进度跟踪 |
| **s06** | s06_subagent | 子 Agent 调度 | 多 Agent 协作模式 |
| **s07** | s07_skill_loading | 技能按需加载 | 动态能力扩展 |
| **s08** | s08_context_compact | 上下文压缩 | 防止历史淹没当前 |
| **s09** | s09_memory | 记忆系统 | 短期/长期记忆管理 |
| **s10** | s10_system_prompt | 系统提示词工程 | 行为约束与角色定义 |
| **s11** | s11_error_recovery | 错误恢复机制 | 失败重试与自愈 |
| **s12** | s12_task_system | 任务系统与依赖图 | 复杂任务分解 |
| **s13** | s13_background_tasks | 后台任务调度 | 异步执行与协调 |
| **s14** | s14_cron_scheduler | 定时任务调度 | 周期性任务管理 |
| **s15** | s15_agent_teams | Agent 团队协作 | 多 Agent 组织模式 |

### 数据流向

```
用户输入
    ↓
s01 Agent 循环接收
    ↓
s10 系统提示词约束
    ↓
s07 按需加载技能 + s09 检索记忆
    ↓
s02 调用工具执行
    ↓
s03 权限检查
    ↓
s04 钩子触发事件
    ↓
s05 更新任务状态
    ↓
s08 上下文压缩
    ↓
返回结果给用户
    ↓
（并行时）s06 调度子 Agent / s15 团队协作
    ↓
（异步时）s13 后台任务 / s14 定时任务
    ↓
s11 错误恢复（如失败）
```

---

## 代码逻辑主线

### Agent 主循环（s01）

```python
def agent_loop(user_input, system_prompt, tools):
    """
    Agent 主循环 - 理解核心执行流程
    
    1. 接收用户输入
    2. 构建完整上下文（系统提示词 + 历史 + 输入）
    3. 调用 LLM 推理
    4. 解析输出（是否调用工具？）
    5. 执行工具调用
    6. 观察执行结果
    7. 将结果加入上下文
    8. 重复或返回最终结果
    """
    conversation = [system_prompt]
    
    while True:
        # 调用模型
        response = call_llm(conversation)
        
        # 检查是否需要调用工具
        if has_tool_call(response):
            tool_result = execute_tool(response.tool)
            conversation.append(tool_result)
        else:
            # 最终回复
            return response.text
```

**关键设计**：
- 循环直到模型不再调用工具
- 每次工具执行结果都加入上下文
- 模型根据完整上下文决定下一步

### 工具调用机制（s02）

```python
class Tool:
    """
    工具定义 - 给模型"双手"
    
    每个工具包含:
    - name: 工具名称
    - description: 工具描述（模型理解用）
    - parameters: 参数 schema（模型生成用）
    - execute: 实际执行函数
    """
    
    def __init__(self, name, description, schema, func):
        self.name = name
        self.description = description
        self.schema = schema
        self.func = func
    
    def execute(self, params):
        """执行工具调用"""
        # 1. 参数校验
        validate_params(params, self.schema)
        
        # 2. 权限检查
        check_permission(self.name, params)
        
        # 3. 执行
        return self.func(params)
```

**关键设计**：
- 工具描述给模型理解"能做什么"
- 参数 schema 给模型知道"怎么调用"
- 权限检查在执行前确保安全

### 上下文压缩（s08）

```python
def compact_context(conversation, max_tokens=4000):
    """
    上下文压缩 - 防止历史淹没当前
    
    策略:
    1. 保留系统提示词（永远不能丢）
    2. 保留最近 N 轮对话
    3. 压缩早期对话为摘要
    """
    system_prompt = conversation[0]
    recent_turns = conversation[-6:]  # 保留最近 3 轮
    
    # 早期对话压缩为摘要
    early_turns = conversation[1:-6]
    summary = summarize(early_turns)
    
    return [system_prompt, summary] + recent_turns
```

**关键设计**：
- 系统提示词永远保留
- 最近对话完整保留
- 早期对话压缩为摘要
- 防止 token 超限

### 子 Agent 调度（s06）

```python
def spawn_subagent(task, context):
    """
    子 Agent 调度 - 复杂任务分解
    
    场景:
    - 主 Agent 发现子任务可并行
    - 创建子 Agent 独立执行
    - 子 Agent 完成后汇总结果
    """
    # 1. 创建子 Agent
    subagent = Agent(
        task=task,
        context=context,
        isolation="worktree"  # 工作树隔离
    )
    
    # 2. 异步执行
    result = await subagent.run()
    
    # 3. 汇总结果
    return result
```

**关键设计**：
- 工作树隔离（每个子 Agent 独立工作目录）
- 异步执行（不阻塞主循环）
- 结果汇总（合并到主上下文）

---

## 快速上手实践

### 环境配置步骤

```bash
# 1. 进入项目目录
cd 04-Agent实战/learn-claude-code

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux

# 3. 安装依赖
pip install -r requirements.txt
# 包含: anthropic>=0.25.0, python-dotenv>=1.0.0, pyyaml>=6.0

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 ANTHROPIC_API_KEY

# 5. 验证环境
python -c "import anthropic; print('✅ anthropic 导入成功')"
```

### 运行第一个示例

```bash
# 从 s01 开始，理解 Agent 循环
cd s01_agent_loop

# 查看代码结构
ls -la

# 运行示例（需要先配置 API Key）
python agent_loop.py
```

**预期输出**：
```
🤖 Agent Loop 启动...
📥 接收用户输入: "读取 README.md 并总结"
🧠 调用模型推理...
🔧 调用工具: read_file("README.md")
📤 工具返回: 文件内容（1024 字符）
🧠 模型生成总结...
✅ 输出: "本文档介绍了..."
```

### 学习路径建议

```
阶段1: 理解核心概念（s01-s03）
  s01_agent_loop → s02_tool_use → s03_permission
  重点: Agent 循环、工具调用、权限控制

阶段2: 掌握扩展机制（s04-s07）
  s04_hooks → s05_todo_write → s06_subagent → s07_skill_loading
  重点: 事件系统、任务管理、子 Agent、技能加载

阶段3: 理解生产级设计（s08-s12）
  s08_context_compact → s09_memory → s10_system_prompt → s11_error_recovery → s12_task_system
  重点: 上下文压缩、记忆管理、错误恢复

阶段4: 高级特性（s13-s15）
  s13_background_tasks → s14_cron_scheduler → s15_agent_teams
  重点: 后台任务、定时调度、团队协作
```

---

## 核心知识点总结

### 1. Harness Engineering 设计哲学

**核心观点**: 智能来自模型训练，不是代码编排。

**为什么重要**：
- 纠正"写代码构建 Agent"的错误认知
- 模型是司机，执行环境是车辆
- 你的工作是造好车，不是教司机开车

### 2. Agent 循环设计

```
接收输入 → 模型推理 → 工具执行 → 观察结果 → 循环或返回
```

**为什么重要**：这是所有 Agent 产品的核心执行模式。

### 3. 工具调用三要素

- **name** - 工具唯一标识
- **description** - 模型理解工具能力
- **parameters** - 模型知道如何调用

**为什么重要**：工具描述质量直接决定模型调用准确率。

### 4. 上下文压缩策略

```
系统提示词（永远保留） + 最近 N 轮（完整） + 早期对话（摘要）
```

**为什么重要**：防止 token 超限导致上下文丢失。

### 5. 子 Agent 隔离模式

- **工作树隔离** - 每个子 Agent 独立 Git worktree
- **上下文隔离** - 不共享对话历史
- **权限隔离** - 独立权限检查

**为什么重要**：防止子 Agent 互相干扰。

### 6. 技能按需加载

不是启动时加载所有技能，而是在需要时动态加载。

**为什么重要**：
- 节省 token
- 减少启动时间
- 技能可随时更新

### 7. 权限治理三层

- **沙箱** - 文件系统隔离
- **审批** - 危险操作需人工确认
- **信任边界** - 外部系统访问控制

**为什么重要**：生产级 Agent 必须的安全机制。

---

## 常见疑问解答

### Q1: 为什么说"Agent 不是代码写出来的"？

**核心观点**：Agent 的感知、推理、决策能力来自模型训练（数十亿参数调整），不是周围的代码。

代码只是给模型提供：
- 工具（双手）
- 知识（领域专长）
- 环境（操作空间）
- 权限（安全边界）

### Q2: Harness 和框架（如 LangChain）有什么区别？

- **框架** - 提供通用组件，你拼装 Agent
- **Harness** - 提供完整执行环境，模型自主决策

框架是"你用代码定义流程"，Harness 是"模型决定流程，你提供执行环境"。

### Q3: 上下文压缩为什么不直接用摘要？

**问题**：早期对话全部摘要会丢失细节。

**更好方案**：
- 保留系统提示词（约束）
- 保留最近对话（相关性最高）
- 摘要早期对话（概览）

### Q4: 为什么 Claude Code 用工作树隔离而不是进程隔离？

**工作树隔离优势**：
- 每个子 Agent 独立 Git 分支
- 可并行执行不冲突
- 结果可合并回主分支
- 比进程隔离更轻量

---

## 学习路径建议

### 第一步：理解核心理念（1 小时）

1. 阅读 README.md 理解 Harness Engineering 哲学
2. 理解"模型提供智能，环境提供操作空间"
3. 思考你的项目中哪里是模型，哪里是环境

### 第二步：掌握核心模块（3 小时）

1. s01_agent_loop - 运行示例，理解循环机制
2. s02_tool_use - 定义自己的工具
3. s03_permission - 理解权限检查流程

### 第三步：深入扩展模块（4 小时）

1. s06_subagent - 实现子 Agent 调度
2. s08_context_compact - 实现上下文压缩
3. s07_skill_loading - 理解按需加载

### 第四步：生产级设计（3 小时）

1. s11_error_recovery - 错误恢复机制
2. s12_task_system - 任务依赖图
3. s15_agent_teams - 团队协作模式

---

## 与教程目录的关联

本项目与以下教程内容强关联：

| 教程章节 | 关联点 |
|---------|-------|
| `07-多Agent与Agent工程化/04-Agent工程化框架` | Harness Engineering 原理 |
| `06-单Agent开发与框架/01-Agent核心概念` | Agent 循环与架构 |
| `07-多Agent与Agent工程化/02-Agent编排与通信` | 子 Agent 调度与协作 |

---

## 总结

**learn-claude-code 不是 Claude Code 的简单 clone，而是系统化学习 Harness Engineering 的教科书**。它展示了：

1. ✅ **Agent 核心循环** - 接收→推理→执行→观察
2. ✅ **工具系统设计** - 给模型"双手"
3. ✅ **上下文管理** - 压缩、记忆、隔离
4. ✅ **权限治理** - 沙箱、审批、信任边界
5. ✅ **子 Agent 调度** - 多 Agent 协作模式

**面试中可以讲的亮点**：
- 我深入学习了 Claude Code 源码，理解生产级 Agent 架构设计
- 我掌握了 Harness Engineering 理念，知道模型和执行环境的边界
- 我理解了上下文压缩、子 Agent 隔离、权限治理等生产级机制
