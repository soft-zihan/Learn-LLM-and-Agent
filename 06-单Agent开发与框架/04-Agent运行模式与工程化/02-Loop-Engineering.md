# Loop Engineering: 从 Harness 到自主循环

> 📅 **更新时间**: 2026-06-18  

---

## 目录

- [1. 从 Harness 到 Loop 的演进](#1-从-harness-到-loop-的演进)
- [2. Loop Engineering 五大核心组件](#2-loop-engineering-五大核心组件)
- [3. 生成器-评估器-规划器架构](#3-生成器-评估器-规划器架构)
- [4. Loop 实战模式](#4-loop-实战模式)
- [5. Loop 的风险与治理](#5-loop-的风险与治理)
- [6. Token 成本优化实战](#6-token-成本优化实战)
- [7. 最佳实践](#7-最佳实践)
- [8. 总结: 构建 Loop,但保持你是工程师](#8-总结-构建-loop但保持你是工程师)
- [9. 参考链接](#9-参考链接)

---

## 1. 从 Harness 到 Loop 的演进

### 1.1 四代范式对比

| 范式 | 时间 | 核心问题 | 解决方案 | 代表 |
|------|------|---------|---------|------|
| **Prompt Engineering** | 2023-2024 | 如何让模型输出更好？ | 写好提示词 | CoT, Few-shot |
| **Context Engineering** | 2024-2025 | 如何给模型正确信息？ | 管理上下文 | RAG, 向量数据库 |
| **Harness Engineering** | 2025-2026 | 如何让 Agent 可靠工作？ | 构建工具链 | Skills, MCP, AGENTS.md |
| **Loop Engineering** | 2026-2027 | 如何让 Agent 自主持续运行？ | 设计循环系统 | /loop, /goal, Worktree |

### 1.2 核心转变

**Harness Engineering 的局限**：
```python
# Harness 模式：人启动 → Agent 执行 → 人检查结果
human: "帮我修复这个 bug"
harness: 
    agent.execute(task)
    agent.verify(result)
    return result
human: "检查结果"  # 人必须参与每次循环
```

**Loop Engineering 的突破**：
```python
# Loop 模式：人设计 → 系统自主运行 → 人只看最终结果
human_designer: 定义目标 + 验证标准 + 循环逻辑
loop_system:
    while not 目标完成:
        agent.execute()
        independent_verifier.verify()  # 独立验证者
        memory.update()
        if 需要人类:
            escalate_to_human()
    return 最终结果
human_reviewer: "审核最终结果"  # 人只在关键节点介入
```

**关键洞察**：
> Harness 让 Agent 单次工作更可靠，Loop 让 Agent 持续工作无需人工干预。

---

## 2. Loop Engineering 五大核心组件

### 2.1 组件全景图

```
Loop Engineering 架构
│
├── 1. 自动化（心跳）
│   ├── 定时触发：cron, 间隔
│   ├── 事件触发：hook, 生命周期
│   └── 工作发现与分流
│
├── 2. 工作树隔离（Worktree）
│   ├── Git worktree 独立目录
│   ├── 多 Agent 并行不冲突
│   └── 分支隔离与合并策略
│
├── 3. 技能系统（Skills）
│   ├── SKILL.md 项目知识
│   ├── 避免每次重新解释项目
│   └── 意图复利效应
│
├── 4. MCP 连接器
│   ├── Issue 追踪器（Linear, Jira）
│   ├── 数据库/API 访问
│   └── 通知系统（Slack, 邮件）
│
├── 5. 子 Agent 架构
│   ├── 执行 Agent（干活）
│   ├── 验证 Agent（检查）
│   └── 规划 Agent（调度）
│
└── 6. 记忆系统
    ├── 工作记忆（当前会话）
    ├── 长期记忆（MEMORY.md）
    └── 状态文件（进度追踪）
```

### 2.2 各组件协同工作流

```python
# 完整 Loop 工作流示例
class DailyCICDFixLoop:
    """每天自动修复 CI 失败的 Loop"""
    
    def run(self):
        # 1. 自动化触发（每天早上 9 点）
        @cron("0 9 * * 1-5")
        def trigger():
            # 2. 工作发现
            failures = discover_ci_failures()
            issues = fetch_open_issues()
            
            # 3. 分流
            tasks = triage(failures + issues)
            
            # 4. 并行处理
            for task in tasks:
                # 4.1 创建隔离环境
                worktree = git_worktree.create(f"fix/{task.id}")
                
                # 4.2 执行修复
                result = executor_agent.run(
                    workdir=worktree,
                    task=task,
                    skills=load_skills()  # 加载项目知识
                )
                
                # 4.3 独立验证
                review = reviewer_agent.verify(
                    result=result,
                    run_tests=True,
                    use_playwright=True  # 实际打开网页测试
                )
                
                # 4.4 提交或反馈
                if review.passed:
                    pr = create_pr(branch=f"fix/{task.id}")
                    notify_slack(f"PR #{pr.id} 已创建")
                    update_linear_ticket(task.id, status="done")
                else:
                    # 4.5 记录到状态文件
                    memory.update({
                        task.id: {
                            "status": "failed",
                            "issues": review.issues,
                            "next_action": "人工审查"
                        }
                    })
            
            # 5. 汇总报告
            send_daily_report(memory.get_status())
```

---

## 3. 生成器-评估器-规划器架构

### 3.1 为什么需要拆开制造者和验证者

**问题**：模型给自己打分太心软了。

```python
# ❌ 自我评估陷阱
result = agent.execute("修复 bug")
if agent.self_evaluate(result):  # 容易放水
    commit(result)

# Anthropic 工程师 Ash 的洞察：
# "自我评估是一个陷阱。模型很容易对自己的作品过于宽容。
#  但如果把'构建者'和'批评者'拆开，单独训练一个更严苛的评估器会更可控。"
```

### 3.2 三组件架构

```python
class GeneratorEvaluatorPlannerHarness:
    """
    生成器-评估器-规划器架构
    
    借鉴 GAN（生成对抗网络）思想：
    - 生成器负责构建应用
    - 评估器负责批判和打分
    - 规划器协调两者，决定下一步
    """
    
    def __init__(self):
        # 生成器：强模型，负责创造性工作
        self.generator = Agent(
            model="claude-opus-4-5",
            system_prompt="你是资深工程师，负责高质量实现"
        )
        
        # 评估器：严苛模型，独立上下文
        self.evaluator = Agent(
            model="claude-sonnet-4",  # 可以用不同模型
            system_prompt="""你是资深代码审查专家，负责发现所有问题。
你的标准应该比生成器高 10 倍。"""
        )
        
        # 规划器：轻量模型，负责调度
        self.planner = Agent(
            model="claude-haiku",  # 便宜模型做调度
            system_prompt="你是项目经理，负责协调资源"
        )
    
    def run(self, task: str, max_rounds: int = 10):
        for round in range(max_rounds):
            # 1. 生成器构建
            result = self.generator.execute(
                task=task,
                previous_feedback=feedback if round > 0 else None
            )
            
            # 2. 评估器批判（独立上下文窗口）
            evaluation = self.evaluator.evaluate(
                result=result,
                # 关键：真正运行测试，而不只是阅读代码 diff
                run_tests=True,
                use_playwright=True,  # 打开网页实际点击测试
                check_security=True
            )
            
            # 3. 规划器决策
            decision = self.planner.decide(
                evaluation=evaluation,
                round=round,
                max_rounds=max_rounds
            )
            
            if decision == "pass":
                return result
            elif decision == "revise":
                feedback = evaluation.issues
                # 继续下一轮
            else:  # escalate
                return escalate_to_human(result, evaluation)
        
        raise Exception(f"超过最大轮数 {max_rounds}")
```

### 3.3 评估器的关键设计

**评估器应该做什么**：
```python
# ✅ 正确做法：实际运行验证
evaluator.evaluate(
    result=code_diff,
    # 1. 运行单元测试
    run_tests=True,
    # 2. 运行集成测试
    run_integration_tests=True,
    # 3. 实际打开网页测试
    use_playwright=True,  # 打开网页 → 点击按钮 → 截图 → 验证
    # 4. 安全检查
    check_security=True,  # 检查 SQL 注入、XSS 等
    # 5. 性能检查
    benchmark_performance=True
)

# ❌ 错误做法：只读代码
evaluator.evaluate(
    result=code_diff,
    # 只阅读代码 diff，不实际运行
    # 模型很容易自我欺骗："这段代码看起来没问题"
)
```

**能力差距利用**：
> 一个人评价一幅画或一道菜很容易，但亲自画出来或做出来要难得多。大语言模型同样如此：它作为批评者和作为生成者之间存在能力差距，而 Harness 可以利用这种差距。

---

## 4. Loop 实战模式

### 4.1 完整 Loop 案例：自动化代码审查

```python
class AutomatedCodeReviewLoop:
    """自动代码审查 Loop"""
    
    def run(self):
        # 触发：每次 PR 提交
        @hook("PullRequestOpened")
        def on_pr_opened(pr):
            # 1. 创建工作树
            worktree = git_worktree.checkout(pr.branch)
            
            # 2. 多个审查 Agent 并行
            reviewers = [
                SecurityReviewer(),    # 安全审查
                PerformanceReviewer(), # 性能审查
                StyleReviewer(),       # 代码风格
                LogicReviewer(),       # 逻辑正确性
            ]
            
            results = parallel(reviewer.run(worktree) for reviewer in reviewers)
            
            # 3. 汇总审查意见
            report = ReviewReport(results)
            
            # 4. 如果问题严重，拒绝 PR
            if report.has_critical_issues():
                pr.reject(reason=report.summary)
                notify_author(report)
            else:
                # 5. 小问题自动生成修复建议
                suggestions = report.generate_fixes()
                pr.comment(suggestions)
                pr.approve(condition="修复上述小问题")
```

### 4.2 Loop 模式对比

| 模式 | 触发方式 | 运行时长 | 适用场景 |
|------|---------|---------|---------|
| **/loop** | 按节奏重复执行 | 持续运行 | CI 修复、代码审查 |
| **/goal** | 一直跑到条件满足 | 直到目标达成 | 复杂任务、多步骤工作 |
| **Hook 触发** | 特定事件 | 短任务 | PR 审查、部署验证 |
| **Cron 触发** | 定时任务 | 中等时长 | 每日报告、定期检查 |

---

## 5. Loop 的风险与治理

### 5.1 理解债务

**问题定义**：Loop 越快地发布你没写过的代码，"代码库里有什么"和"你真正理解什么"之间的差距就越大。

**风险演进**：
```
Week 1: Loop 修复 10 个 bug → 你看了 10 个 PR → 理解 100%
Week 2: Loop 修复 50 个 bug → 你看了 20 个 PR → 理解 60%
Week 4: Loop 修复 200 个 bug → 你看了 10 个 PR → 理解 20%
        ↓
        出了生产事故，你看不懂代码库，无法快速定位问题
```

**应对策略**：
```python
# 1. 强制审查比例
class UnderstandingDebtGuard:
    def __init__(self):
        self.reviewed_count = 0
        self.total_count = 0
        self.min_review_ratio = 0.3  # 至少审查 30%
    
    def should_review(self, pr):
        current_ratio = self.reviewed_count / max(self.total_count, 1)
        if current_ratio < self.min_review_ratio:
            return True  # 强制审查
        return random.random() < 0.2  # 20% 随机抽查

# 2. 定期深度审查
@cron("0 10 * * 5")  # 每周五上午 10 点
def weekly_deep_review():
    """每周深度审查 Loop 产出的代码"""
    prs = get_loop_generated_prs(week="last")
    for pr in prs.sample(10):  # 随机抽 10 个
        human_review(pr, depth="deep")
```

### 5.2 认知投降风险

**两种 Loop 设计者**：
```
同样一个 Loop，不同人跑出来的结果可能天差地别：

人 A：用它在自己深刻理解的工作上加速
     → Loop 是杠杆，放大已有能力
     → 结果：效率提升，质量稳定

人 B：用它来逃避理解工作本身
     → Loop 是拐杖，掩盖能力不足
     → 结果：效率提升，质量下滑
```

**解药 vs 毒药**：
```
带着判断力设计 Loop = 解药
为了逃避思考而设计 Loop = 毒药

同样的动作，完全相反的结果。
```

### 5.3 验证体系设计

**Loop 中必须包含说"不"的机制**：
```python
class LoopVerificationSystem:
    """Loop 的验证体系"""
    
    def verify(self, result):
        checks = [
            # 1. 单元测试
            self.run_unit_tests,
            # 2. 集成测试
            self.run_integration_tests,
            # 3. 类型检查
            self.run_type_check,
            # 4. Lint 检查
            self.run_lint,
            # 5. 安全扫描
            self.run_security_scan,
            # 6. 性能基准
            self.run_benchmark,
            # 7. 实际端到端测试
            self.run_e2e_tests,
        ]
        
        for check in checks:
            passed, error = check(result)
            if not passed:
                return False, f"验证失败：{check.__name__} - {error}"
        
        return True, "所有验证通过"
```

> 没有反馈机制的 Loop，只会让 Agent 不断重复并自我确认。

---

## 6. Token 成本优化实战

### 6.1 成本分析

```python
# Loop 成本估算
class LoopCostAnalyzer:
    def estimate(self, loop_config):
        # 假设每次迭代消耗
        cost_per_iteration = {
            "generator": 0.08,      # Opus 生成
            "evaluator": 0.02,      # Sonnet 评估
            "planner": 0.005,       # Haiku 规划
        }
        
        total_per_round = sum(cost_per_iteration.values())  # 0.105 美元
        
        # 运行 8 小时
        iterations = {
            "1min_interval": 480,   # 48 美元
            "10min_interval": 48,   # 4.8 美元
            "1hour_interval": 8,    # 0.8 美元
        }
        
        return {
            interval: count * total_per_round
            for interval, count in iterations.items()
        }

# 输出：
# {
#   "1min_interval": 50.4,   # 480 轮 × 0.105
#   "10min_interval": 5.04,  # 48 轮 × 0.105
#   "1hour_interval": 0.84   # 8 轮 × 0.105
# }
```

### 6.2 优化策略

```python
# 1. 降低调用频率
loop.set_interval("10min")  # 而不是 1min

# 2. 使用小模型做验证
evaluator = Agent(model="claude-haiku")  # 0.002 美元/次
# 而不是
evaluator = Agent(model="claude-opus")   # 0.08 美元/次

# 3. 跳过不必要的轮次
if nothing_changed_since_last_run():
    skip_this_round()

# 4. 缓存中间结果
cache = ResultCache(ttl="1h")
if cache.has_valid_result(task):
    return cache.get(task)

# 5. for 循环替代 while 循环
for round in range(5):  # 最多 5 轮，成本可控
    run_iteration()
    if verified():
        break
```

---

## 7. 最佳实践

### 7.1 设计原则

✅ **DO**：
- 分开制造者和验证者
- 建立完整的验证体系（测试、类型检查、Lint）
- 设计反馈机制，让 Loop 能说"不"
- 保持人类判断力，定期深度审查
- 从 for 循环开始，验证有效后再考虑 while 循环

❌ **DON'T**：
- 让 Agent 自我评估
- 没有验证机制就启动 Loop
- 为了逃避思考而设计 Loop
- 盲目接受 Loop 的"完成"声明
- 忽略理解债务的累积

### 7.2 成熟度模型

```
Loop Engineering 成熟度

Level 1: 单次 Prompt
  → 人全程握着 Agent，一轮接一轮

Level 2: Harness Engineering
  → 有 Skills、MCP、验证系统
  → 单次任务更可靠

Level 3: 简单 Loop
  → /loop 定时运行
  → 人工检查结果

Level 4: 自主 Loop
  → 生成器-评估器架构
  → 独立验证，自动提交 PR
  → 人类只看最终结果

Level 5: 智能 Loop
  → 多 Agent 协作
  → 自我优化循环策略
  → 理解债务管理
  → 人类只在关键节点介入
```

### 7.3 未来展望

---

## 8. 总结：构建 Loop，但保持你是工程师

### 8.1 核心原则

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

### 8.2 未来展望

> 真正成熟的 Loop Engineering 系统可能要到 2027 年才会大规模出现，但基础架构的设计工作必须从现在开始。

**关键趋势**：
1. 从被动响应到主动循环
2. 从单次任务到持续运行
3. 从工具使用者到系统设计者
4. 验证体系成为核心竞争力

> 构建 Loop。但像打算继续当工程师的人那样去构建，而不只是那个按下启动键的人。

---

## 9. 参考链接

1. 知乎讨论：https://www.zhihu.com/question/2048003050531558553
2. OpenClaw 文档：https://github.com/OpenClaw/openclaw
3. Claude Code Loop：https://docs.claude.com/en/docs/claude-code/loops
4. Anthropic Harness 实验：https://www.anthropic.com/engineering
5. 虎嗅报道：大人，AI 编程又变天了
6. CSDN 文章：Agent 工程范式四代演进
7. 掘金文章：Loop Engineering 完整解析
