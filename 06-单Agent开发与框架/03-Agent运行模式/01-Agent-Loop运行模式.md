# Agent Loop 运行模式与持续执行

> 📅 **更新时间**: 2026-06-17  

---

## 目录

- [1. 从 Prompt 到 Loop 的演进](#1-从-prompt-到-loop-的演进)
- [2. Agent Loop 的核心组件](#2-agent-loop-的核心组件)
- [3. 记忆系统设计](#3-记忆系统设计)
- [4. 项目约定](#4-项目约定)
- [5. 构建步骤](#5-构建步骤)
- [6. 历史教训](#6-历史教训)
- [7. 当前状态](#7-当前状态)
- [8. 子 Agent 架构](#8-子-agent-架构)
- [9. Loop 实战案例](#9-loop-实战案例)
- [10. Token 成本优化](#10-token-成本优化)
- [11. Loop 的风险与应对](#11-loop-的风险与应对)
- [12. 总结:构建 Loop,但保持你是工程师](#12-总结构建-loop但保持你是工程师)
- [13. 参考链接](#13-参考链接)

---

## 1. 从 Prompt 到 Loop 的演进

### 1.1 Agent 工程的四代范式

| 范式 | 时间 | 核心思想 | 代表技术 | 局限 |
|------|------|---------|---------|------|
| **Prompt Engineering** | 2023-2024 | 写好提示词 | CoT, Few-shot, Role | Token 预算限制 |
| **Context Engineering** | 2024-2025 | 管理上下文 | RAG, 向量数据库 | 信息过载 |
| **Harness Engineering** | 2025-2026 | 构建工具链 | Skills, MCP | 工程复杂度 |
| **Loop Engineering** | 2026-2027 | 设计循环系统 | /loop, /goal, Worktree | 成本、调试难度 |

**关键转变**：从"手动提示 Agent"到"设计循环系统让 Agent 自主运行"。

### 1.2 为什么需要 Loop？

**传统单次执行模式的问题**：
```python
# 单次执行：人全程握着 Agent，一轮接一轮
用户: "帮我修复这个 bug"
Agent: "好的，我来看看...需要修改第 42 行"
用户: "好，继续"
Agent: "改完了，但还有另一个问题..."
用户: "继续修"
# ... 重复 20 轮，人工成本高
```

**Loop 持续运行模式**：
```python
# Loop 模式：设计一次，自动运行
设计者: 定义目标 + 验证标准 + 循环逻辑
Loop: 
  while not 目标完成:
    1. Agent 自主规划并执行
    2. 验证器检查结果
    3. 根据反馈调整策略
    4. 记录进度到记忆
  return 完成结果
```

**核心优势**：
- 减少人工干预，适合 7×24 小时运行
- 适合重复性、标准化任务（代码审查、CI 失败修复）
- 开发者从"操作者"升级为"系统设计者"

---

## 2. Agent Loop 的核心组件

### 2.1 五大组件 + 记忆系统

一个完整的 Loop 需要以下组件：

```
Agent Loop 架构
├── 1. 自动化（心跳）
│   ├── 定时触发（cron, 间隔）
│   ├── 事件触发（hook, 生命周期）
│   └── 工作发现与分流
│
├── 2. 工作树隔离（Worktree）
│   ├── Git worktree 独立目录
│   ├── 多 Agent 并行不冲突
│   └── 分支隔离与合并
│
├── 3. 技能系统（Skills）
│   ├── SKILL.md 项目知识
│   ├── 避免每次重新解释项目
│   └── 意图复利效应
│
├── 4. MCP 连接器
│   ├── Issue 追踪器
│   ├── 数据库/API 访问
│   └── Slack/邮件通知
│
├── 5. 子 Agent
│   ├── 执行 Agent（干活）
│   └── 验证 Agent（检查）
│
└── 6. 记忆系统
    ├── 工作记忆（当前会话）
    ├── 长期记忆（MEMORY.md）
    └── 状态文件（做了什么、下一步）
```

### 2.2 自动化——Loop 的心跳

自动化是把 Loop 变成"真循环"而不是"跑一次就完了"的关键。

**三种触发方式**：

```python
# 1. 定时触发（间隔运行）
/loop --interval 1h --prompt "检查 CI 失败并修复"

# 2. Cron 触发（特定时间）
/loop --cron "0 9 * * 1-5" --prompt "每日代码审查"

# 3. 事件触发（Hook）
# 在 Agent 生命周期的特定节点触发
hooks:
  PostToolUse:
    - matcher: "Write|Edit"
      command: "运行 linter 检查"
  SubagentStop:
    - command: "验证子 Agent 产出"
```

**目标驱动模式（/goal）**：
```python
# /loop 是按节奏重复执行
# /goal 是一直干到某个条件真正满足为止

/goal --condition "所有测试通过且无 lint 错误" --prompt "修复当前 CI 失败"

# 内部逻辑：
while not 条件满足:
    Agent 执行修复
    独立小模型检查"完没完成"  # 关键：不是自己给自己打分
    if 完成:
        break
    if 超过最大轮数:
        报告失败原因
```

### 2.3 工作树隔离——并行不冲突

**问题**：同时跑两个以上 Agent，文件就开始冲突了。

**解决方案**：Git Worktree

```bash
# 创建独立工作目录，共享同一仓库历史
git worktree add ../worktree-fix-bug-123 -b fix/bug-123
git worktree add ../worktree-feature-456 -b fix/feature-456

# Agent A 在 worktree-fix-bug-123 中修改
# Agent B 在 worktree-feature-456 中修改
# 两者物理隔离，互不影响
```

**在 Loop 中的应用**：
```python
# Loop 发现 3 个需要修复的 issue
for issue in issues:
    # 为每个 issue 创建独立 worktree
    worktree = create_worktree(f"fix/issue-{issue.id}")
    
    # 派子 Agent 在隔离环境中修复
    result = sub_agent.run(
        workdir=worktree,
        prompt=f"修复 issue #{issue.id}: {issue.title}"
    )
    
    # 验证后提交 PR
    if verify(result):
        create_pr(branch=f"fix/issue-{issue.id}")
```

---

## 3. 记忆系统设计

### 3.1 为什么需要记忆？

**关键洞察**：模型在两次运行之间会忘掉一切，所以记忆必须在硬盘上，不能只在上下文里。

> Agent 会忘，仓库不会。

### 3.2 多层记忆架构

```
记忆系统
├── 工作记忆（Working Memory）
│   ├── 当前会话的短期状态
│   ├── 存储在上下文窗口中
│   └── 会话结束后丢失
│
├── 长期记忆（MEMORY.md）
│   ├── 项目约定、构建步骤
│   ├── "我们不这么做是因为那次出过事"
│   └── 每次会话加载，持续有效
│
└── 状态文件（Status File）
    ├── Loop 运行状态
    ├── 什么试过了、什么通过了、什么还开着
    └── 明天早上的运行从今天停下的地方接着来
```

### 3.3 实战：MEMORY.md 编写

```markdown
# 项目记忆

## 4. 项目约定
- 使用 Python 3.12，不使用 3.9
- 所有 API 必须有类型注解
- 禁止使用 `eval()` 和 `exec()`

## 5. 构建步骤
```bash
pip install -r requirements.txt
pytest tests/
ruff check .
```

## 6. 历史教训
- 2026-05-20: 不要在主分支直接提交，必须走 PR
- 2026-06-01: 数据库迁移必须先备份再执行
- 2026-06-10: 使用 `requests.Session()` 而不是每次新建连接

## 7. 当前状态
- 正在进行：修复 CI 失败（issue #123, #125）
- 已完成：重构用户认证模块
- 待办：添加集成测试
```

**没有记忆 vs 有记忆的效果对比**：

| 场景 | 无记忆 | 有记忆 |
|------|--------|--------|
| 每次会话开始 | 从零推导项目结构 | 直接读取 MEMORY.md |
| 代码风格 | 可能违反约定 | 严格遵守历史约定 |
| 错误处理 | 重蹈覆辙 | 避免已知陷阱 |
| Token 消耗 | 每次都重新解释 | 复利效应，越跑越省 |

---

## 8. 子 Agent 架构

### 4.1 制造者与验证者必须分开

**核心原则**：写代码的模型给自己打分太心软了。

```python
# ❌ 错误做法：自己检查自己
result = agent.execute("修复 bug")
if agent.verify(result):  # 自己给自己打分，容易放水
    commit(result)

# ✅ 正确做法：独立验证者
result = builder_agent.execute("修复 bug")
review = reviewer_agent.verify(result)  # 独立 Agent，不同指令/模型
if review.passed:
    commit(result)
else:
    feedback_to_builder(review.issues)
```

### 4.2 生成器-评估器模式

```python
# Anthropic 内部实验的 Harness 模式
class GeneratorEvaluatorHarness:
    """
    借鉴 GAN 思想：
    - 生成器负责构建应用
    - 评估器负责批判和打分
    - 两者通过对抗压力不断提升结果质量
    """
    
    def run(self, task: str, max_rounds: int = 10):
        for round in range(max_rounds):
            # 生成器构建
            result = self.generator.execute(
                task=task,
                previous_feedback=feedback if round > 0 else None
            )
            
            # 评估器批判（独立上下文窗口）
            evaluation = self.evaluator.evaluate(
                result=result,
                # 评估器真正运行测试，而不只是阅读代码
                run_tests=True,
                use_playwright=True  # 打开网页实际点击测试
            )
            
            if evaluation.passed:
                return result
            
            # 反馈给生成器
            feedback = evaluation.issues
        
        raise Exception(f"超过最大轮数 {max_rounds}，最终结果未通过验证")
```

**关键洞察**：自我评估是一个陷阱。模型很容易对自己的作品过于宽容。把"构建者"和"批评者"拆开，单独训练一个更严苛的评估器会更可控。

---

## 9. Loop 实战案例

### 5.1 完整的 Loop 长什么样

```python
# 这是一个每天早上自动运行的 Loop
# 设计一次，之后无需手动提示

自动化触发（每天早上 9 点）
    ↓
分流技能：读取昨天的 CI 失败、开放的 issue、最近的 commit
    ↓
写入状态文件（markdown 或 Linear 看板）
    ↓
对于每个值得处理的发现：
    ├── 打开隔离的 worktree
    ├── 派子 Agent A 起草修复方案
    ├── 派子 Agent B 根据项目技能和现有测试审查方案
    ├── 连接器：自己打开 PR、更新工单
    └── CI 绿了之后自动 ping 频道
    ↓
Loop 处理不了的进分流收件箱给人类
    ↓
状态文件记录：什么试过了、什么通过了、什么还开着
    ↓
明天早上的运行从今天停下的地方接着来
```

**看看你实际做了什么**：你设计了一次，然后你没有提示其中任何一个步骤。这就是 Loop Engineering 的活生生体现。

### 5.2 适用场景

**适合用 Loop 的场景**：
- ✅ 有明确目标和验证标准的任务
- ✅ 重复性、标准化工作（代码审查、CI 修复）
- ✅ Token 充裕的团队
- ✅ 能够接受一定试错成本的项目

**不适合用 Loop 的场景**：
- ❌ Token 预算有限（20 美元套餐根本不够）
- ❌ 需要高度创意和判断力的工作
- ❌ 连可靠的一次性 prompt 都还没写好的团队

---

## 10. Token 成本优化

### 6.1 for 循环 vs while 循环

```python
# while 循环（Token 充裕）：一直跑到条件满足
while not goal_achieved:
    agent.run()  # 每次都是完整调用，成本高昂

# for 循环（Token 紧张）：限定最大轮数
for round in range(max_rounds):  # 例如 5 轮
    agent.run()
    if verify():
        break
# 花的时间更长，但成本可控
```

**实际成本估算**：
```
假设每次 Loop 迭代消耗 0.10 美元
- 1 分钟执行一次，8 小时 = 480 次调用 = 48 美元
- 10 分钟执行一次，8 小时 = 48 次调用 = 4.8 美元
- 1 小时执行一次，8 小时 = 8 次调用 = 0.8 美元
```

### 6.2 成本优化策略

1. **降低调用频率**：从 1 分钟改为 10 分钟或 1 小时
2. **使用小模型做验证**：验证器用便宜模型（如 Claude Haiku），生成器用强模型
3. **跳过不必要的轮次**：如果上次运行什么都没发现，直接跳过
4. **缓存中间结果**：避免重复计算

---

## 11. Loop 的风险与应对

### 7.1 理解债务

**问题**：Loop 越快地发布你没写过的代码，"代码库里有什么"和"你真正理解什么"之间的差距就越大。

**应对**：
- 亲自阅读 Loop 产出的代码
- 定期做代码审查
- 不要盲目接受 Loop 的"完成"声明

> "完成"是个声明，不是证明。你的工作是发布你确认能用的代码。

### 7.2 保持判断力

**两种 Loop 设计者**：
```
两个人搭完全一样的 Loop：
- 人 A：用它在自己深刻理解的工作上加速
- 人 B：用它来逃避理解工作本身

Loop 不知道两者的区别。你知道。
```

**这就是为什么 Loop 设计比 Prompt 工程更难**：杠杆的支点挪了，不是活儿变轻松了。

### 7.3 验证体系

```python
# Loop 中必须包含说"不"的机制
Loop 设计:
    ├── 测试（Test）
    ├── 类型检查（Type Check）
    ├── Lint（代码规范）
    └── 真实错误（Real Errors）

# 没有反馈机制的 Loop，只会让 Agent 不断重复并自我确认
```

---

## 12. 总结：构建 Loop，但保持你是工程师

### 12.1 核心原则

✅ **DO**：
- 带着判断力设计 Loop
- 亲自审核 Loop 产出的代码
- 分开制造者和验证者
- 建立反馈机制和验证体系

❌ **DON'T**：
- 为了逃避思考而设计 Loop
- 完全信任 Loop 的"完成"声明
- 忽略理解债务的累积
- 放弃工程师的判断力

### 12.2 未来展望

Loop Engineering 代表了 AI Agent 工程的一个新方向：
- 从手动操作到自动循环
- 从单次任务到持续运行
- 从工具使用者到系统设计者

但真正成熟的 Loop 系统可能要到 2027 年才会大规模出现，基础架构的设计工作必须从现在开始。

> 构建 Loop。但像打算继续当工程师的人那样去构建，而不只是那个按下启动键的人。

---

## 13. 参考链接

1. 知乎讨论：https://www.zhihu.com/question/2048003050531558553
2. 掘金文章：Loop Engineering 完整解析
3. CSDN 文章：Agent 工程范式四代演进
4. 虎嗅报道：大人，AI 编程又变天了
5. OpenClaw 文档：https://github.com/OpenClaw/openclaw
6. Claude Code Loop 文档：https://docs.claude.com/en/docs/claude-code/loops
