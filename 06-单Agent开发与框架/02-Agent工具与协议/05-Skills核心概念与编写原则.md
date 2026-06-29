# Skills 核心概念与编写原则

> 📅 **更新时间**: 2026-06-29  
> 📚 **参考资料**: [Anthropic Skills 官方文档](https://platform.claude.com/docs/zh-CN/agents-and-tools/agent-skills/best-practices)、[Agent Skills 规范](https://agentskills.io/)、[awesome-agent-skills 社区](https://github.com/libukai/awesome-agent-skills)

---

## 目录

- [1. 什么是 Skills](#1-什么是-skills)
- [2. Skill 标准文件结构](#2-skill-标准文件结构)
- [3. Skill 编写六大原则](#3-skill-编写六大原则)
- [4. 真实示例解析：find-skills](#4-真实示例解析find-skills)
- [5. 常见反模式](#5-常见反模式)

---

## 1. 什么是 Skills

### 1.1 Skill 的本质

**Agent Skill（代理技能）** 是**一个包含 `SKILL.md` 文件的文件夹**，AI Agent 在执行任务时动态加载，用于提升特定领域的表现。

> **Skill 不是可执行代码，而是教 AI 如何做事的结构化知识文档。**

AI 读取 `SKILL.md` 后，利用自身已有的工具（Bash、文件读写、搜索等）独立完成任务，不依赖特定 MCP 服务。

### 1.2 为什么需要 Skills

| 问题 | 没有 Skill | 有 Skill |
|------|-----------|----------|
| 上下文浪费 | 每次对话重复发送规范 | 按需加载，节省 token |
| 输出不一致 | 每次生成的结构不同 | 固定模板，保证一致性 |
| 领域知识缺失 | AI 靠训练数据猜测 | 注入团队特定规范 |
| 难以复用 | 散落在对话记录中 | 文件夹即插即用 |

### 1.3 Skill 与 Tool、Prompt 的区别

| 维度 | Prompt | Tool | Skill |
|------|--------|------|-------|
| **形态** | 一段文字提示 | 一个函数/接口 | 一个文件夹（含 SKILL.md） |
| **作用** | 单次引导 AI 行为 | 执行特定操作 | 封装完整工作流 |
| **复用性** | 低 | 中 | 高 |
| **依赖** | 无 | 需要代码实现 | 无，纯 Markdown |
| **生命周期** | 单次对话 | 持久化服务 | 按需加载 |

---

## 2. Skill 标准文件结构

### 2.1 最小结构

一个 Skill 最少只需要一个文件：

```
my-skill/
└── SKILL.md          # 必须：主指令文件
```

### 2.2 完整结构

复杂 Skill 可包含参考文档、脚本、模板：

```
my-skill/
├── SKILL.md              # 主指令：工作流、决策树、规则
├── references/           # 参考文档：按需加载，减少 token
│   ├── conventions.md    # 领域知识/编码规范
│   ├── checklist.md      # 审查清单
│   └── style-guide.md    # 风格指南
├── scripts/              # 实用脚本：Agent 直接执行
│   ├── validate.py
│   └── helper.sh
├── templates/            # 输出模板
│   └── report.md
└── examples/             # 使用示例
    └── sample.md
```

### 2.3 SKILL.md 基本结构

```markdown
---
name: my-skill-name
description: 一句话描述何时使用此 Skill
---

# My Skill Name

触发条件说明...

## 工作流程
1. 第一步
2. 第二步
3. ...

## 输出格式
（描述输出结构）

## 注意事项
- 规则 1
- 规则 2
```

**必需元数据字段**：

| 字段 | 要求 | 说明 |
|------|------|------|
| `name` | 最多 64 字符，仅小写字母/数字/连字符 | Skill 的唯一标识符 |
| `description` | 最多 1024 字符，非空 | 帮助 Agent 决定何时触发此 Skill |

---

## 3. Skill 编写六大原则

### 原则一：写好元数据，description 是触发关键

**description 决定了 AI 是否会加载你的 Skill。** 它会被注入到系统提示中，用于匹配用户意图。

#### 三要

| 要点 | 说明 | 示例 |
|------|------|------|
| 第三人称 | 描述会被注入系统提示，人称不一致会导致触发失败 | ✅ "Processes PDF files..." |
| 包含 WHAT | 描述 Skill 做什么 | ✅ "Extract text and tables from PDF files..." |
| 包含 WHEN | 描述何时触发 | ✅ "Use when working with PDF files or when the user mentions PDFs..." |

#### 三不要

| 错误 | 说明 | 示例 |
|------|------|------|
| ❌ 第一/第二人称 | 系统提示中视角混乱 | "I can help you..." / "You can use this..." |
| ❌ 过于简短 | AI 无法判断触发场景 | "代码审查" |
| ❌ 过于宽泛 | 会错误匹配到无关任务 | "帮助处理文档" |

#### 优秀 description 示例

```yaml
# PDF 处理
description: Extract text and tables from PDF files, fill forms, merge documents. Use when working with PDF files or when the user mentions PDFs, forms, or document extraction.

# 代码审查
description: Review code for quality, security, and maintainability following team standards. Use when reviewing pull requests, examining code changes, or when the user asks for a code review.

# 搜索技能
description: Helps users discover and install agent skills when they ask questions like "how do I do X", "find a skill for X", "is there a skill that can...", or express interest in extending capabilities.
```

**关键结论：写好 description 比写好指令主体更重要。** 描述是触发的关键，应该同时包含功能和触发场景。

### 原则二：一个 Skill 只做一件事

遵循单一职责原则，每个 Skill 聚焦一个领域或工作流。

#### 错误示例：什么都做的 MegaSkill

```markdown
---
name: dev-assistant
description: 帮助开发者完成各种任务
---

# 开发助手

我可以帮你：
- 写代码
- 审查代码
- 生成测试
- 写文档
- 部署应用
- 监控日志
```

问题：
- 描述太模糊，AI 无法判断何时触发
- 指令互相干扰，容易遗漏关键步骤
- 上下文窗口被大量无关内容占用

#### 正确示例：拆分为多个 Skill

```
code-reviewer/          # 只负责代码审查
└── SKILL.md

test-generator/         # 只负责生成测试
└── SKILL.md

doc-writer/             # 只负责写文档
└── SKILL.md
```

**组合优于堆砌**：通过 Skill 组合实现复杂工作流，比塞进一个 Skill 更可靠。

### 原则三：SKILL.md 控制在 500 行以内

上下文窗口要和对话历史、其他 Skill 共享。每一行都在消耗 token。

#### 默认假设：AI 已经很聪明了

只添加 AI 不知道的上下文。每个内容片段都应该被质疑：
- "AI 真的需要这个解释吗？"
- "能假设 AI 已经知道这个吗？"
- "这段文字值得花费 token 吗？"

#### 精简示例

**✅ 简洁（推荐）**：
```markdown
## 提取 PDF 文本

使用 pdfplumber 提取文本：

```python
import pdfplumber
with pdfplumber.open("file.pdf") as pdf:
    text = pdf.pages[0].extract_text()
```
```

**❌ 冗长（避免）**：
```markdown
## 提取 PDF 文本

PDF（Portable Document Format）是一种常见的文件格式，包含文本、图像等内容。
要提取 PDF 中的文本，需要使用专门的库。Python 有很多库可以处理 PDF，
但我们推荐 pdfplumber，因为它易于使用且能处理大多数情况...
```

#### 超过 500 行怎么办？使用渐进式披露

把核心指令放在 `SKILL.md`，详细内容放在 `references/` 目录，通过链接引用：

```markdown
## 编码规范

加载 `references/conventions.md` 获取完整的 FastAPI 最佳实践列表。
```

**保持引用只有一层深度** — 直接从 SKILL.md 链接到参考文件。深层嵌套引用可能导致部分内容无法加载。

### 原则四：区分开放任务和确定性任务

根据任务的"脆弱程度"选择指令的确定性级别：

| 自由度 | 适用场景 | 指令形式 | 示例 |
|--------|---------|---------|------|
| **高** | 多种有效方案，依赖上下文 | 文字指导 | 代码审查指南 |
| **中** | 有推荐模式，允许变通 | 伪代码/模板 | 报告生成 |
| **低** | 操作脆弱，一致性关键 | 具体脚本 | 数据库迁移 |

#### 开放任务示例（高自由度）

```markdown
## 代码审查

审查代码时注意：
- 逻辑是否正确，是否处理了边界情况
- 是否存在安全漏洞（SQL 注入、XSS 等）
- 代码是否遵循项目风格约定
- 函数大小是否合理
- 错误处理是否完整
```

#### 确定性任务示例（低自由度）

```markdown
## 数据库迁移

严格执行以下步骤，不得跳过或改变顺序：

1. 备份当前数据库：`python scripts/backup_db.py`
2. 运行迁移：`python scripts/migrate.py --up`
3. 验证迁移：`python scripts/verify_migration.py`
4. 如果验证失败：`python scripts/rollback.py` 并停止
```

### 原则五：写 AI 不知道的东西

**只教 AI 不知道或不确定的内容**，不要教它已经知道的基础知识。

#### 错误：教 AI 已知知识

```markdown
## 什么是 JSON

JSON（JavaScript Object Notation）是一种轻量级数据交换格式...
```

AI 已经知道 JSON 是什么，不需要你解释。

#### 正确：写 AI 不知道的

```markdown
## 我们的 JSON 输出格式

必须按照以下结构输出，这是我们的系统要求的：

```json
{
  "user_id": "必须是 UUID 格式",
  "action": "只能是 create/update/delete 之一",
  "timestamp": "必须使用 ISO 8601 格式"
}
```
```

**核心区分**：
- **百科式写法** ❌："JSON 是一种数据格式..."（AI 已知）
- **可执行写法** ✅："1. 使用 UUID 格式 2. 验证字段 3. 输出到 output.json"（可操作步骤）

### 原则六：使用渐进式披露

把关键信息放在 `SKILL.md`，详细参考放在单独文件中，AI 按需读取。

#### 示例结构

```
api-expert/
├── SKILL.md                # 150 行：核心工作流
└── references/
    ├── conventions.md      # 400 行：详细编码规范
    ├── error-handling.md   # 200 行：错误处理指南
    └── security.md         # 250 行：安全清单
```

**SKILL.md 中引用**：

```markdown
## 审查代码时

1. 加载 `references/conventions.md` 获取编码规范
2. 对照规范检查代码
3. 加载 `references/security.md` 检查安全问题
```

这样 AI 只有在需要时才加载详细参考，不会一次性消耗大量 token。

---

## 4. 真实示例解析：skill-creator（官方教科书级）

本示例来自 [Anthropic 官方 skills 仓库](https://github.com/anthropics/skills)，是**教 AI 如何创建 Skill 的 Skill**，堪称教科书级实现。

### 4.1 完整目录结构

```
skill-creator-example/
├── SKILL.md                      # 485 行，完整主指令（接近 500 行上限）
├── LICENSE.txt                   # Apache 2.0 许可证
├── references/
│   └── schemas.md                # YAML 元数据 schema 定义
├── scripts/                      # 10 个实用脚本
│   ├── run_loop.py               # 核心循环：迭代改进 Skill
│   ├── run_eval.py               # 运行评估测试
│   ├── quick_validate.py         # 快速验证 Skill 格式
│   ├── improve_description.py    # 自动优化 description
│   ├── package_skill.py          # 打包 Skill 为可分发格式
│   ├── generate_report.py        # 生成评估报告
│   ├── aggregate_benchmark.py    # 聚合基准测试结果
│   └── ...
├── agents/                       # 3 个 Agent 角色定义
│   ├── analyzer.md               # 分析器：评估 Skill 质量
│   ├── comparator.md             # 比较器：对比多个 Skill 版本
│   └── grader.md                 # 评分器：打分和反馈
├── assets/
│   └── eval_review.html          # 评估报告可视化
└── eval-viewer/                  # 评估查看器
    ├── viewer.html
    └── generate_review.py
```

**目录数量级**：1 个主文件 + 1 个参考 + 10 个脚本 + 3 个 Agent 角色 + 2 个查看器文件

[查看完整目录](./skill-creator-example/)

### 4.2 元数据解析

```yaml
---
name: skill-creator
description: Create new skills, modify and improve existing skills, and measure skill 
  performance. Use when users want to create a skill from scratch, edit, or optimize 
  an existing skill, run evals to test a skill, benchmark skill performance with 
  variance analysis, or optimize a skill's description for better triggering accuracy.
---
```

**优点分析**：
1. ✅ **第三人称**："Create new skills..." 不是 "I can help you create..."
2. ✅ **包含 WHAT**："create, modify, improve, measure performance"
3. ✅ **包含 WHEN**：列举了所有触发场景：
   - "create a skill from scratch"
   - "edit, or optimize an existing skill"
   - "run evals to test a skill"
   - "benchmark skill performance"
   - "optimize a skill's description"
4. ✅ **"pushy" 策略**：官方明确说要"pushy"一点，宁可多触发也不要少触发
5. ✅ **长度适中**：description 约 120 字符，覆盖所有场景但不冗长

### 4.3 指令设计解析

#### 4.3.1 清晰的高阶流程

skill-creator 开头用简单的语言描述了整个创建流程：

```markdown
从高层次来看，创建 Skill 的过程是这样的：

- 确定这个 Skill 要做什么，大概怎么做
- 撰写 Skill 草稿
- 创建几个测试提示词，在启用此 Skill 的情况下运行 Claude
- 帮助用户定性和定量地评估结果
  - 在后台运行测试时，起草定量评估（如果还没有的话）
  - 使用评估查看器向用户展示结果
- 根据用户反馈重写 Skill
- 重复直到满意
- 扩大测试集，进行更大规模测试
```

**关键设计**：
- 用简单的 bullet points 描述流程，不是复杂的决策树
- AI 的职责是"判断用户处于哪个阶段，然后帮助他们推进"
- 灵活可调整："始终灵活，如果用户说'我不需要跑评估，随便聊聊就行'，你也可以那样做"

#### 4.3.2 用户沟通适配

一个经常被忽略但极其重要的设计：

```markdown
## 与用户沟通

Skill Creator 的使用者技术水平差异很大。现在的趋势是，Claude 的强大能力
正在鼓励水管工打开终端，父母和祖父母去搜索"如何安装 npm"。

所以请注意上下文线索，理解如何调整你的沟通方式！

- "评估"和"基准测试"勉强可以，但要注意
- 对于"JSON"和"断言"，你需要看到用户明确表示他们知道这些是什么，
  才能在不解释的情况下使用它们
```

**学习要点**：
- 考虑用户的多样性，不要假设技术知识水平
- 给出具体的词汇示例（评估/JSON/断言），不是空泛的"注意措辞"
- 允许 AI 根据上下文调整沟通方式

#### 4.3.3 渐进式披露的完美实践

skill-creator 展示了如何组织多层次的内容：

```markdown
## 渐进式披露

Skill 使用三层加载系统：
1. 元数据（名称 + 描述）- 始终在上下文中（约 100 字）
2. SKILL.md 正文 - 仅在 Skill 触发时加载（理想情况下 <500 行）
3. 捆绑资源 - 按需加载（无限制，脚本可执行无需加载到上下文）

关键模式：
- 保持 SKILL.md 在 500 行以内
- 从 SKILL.md 中清晰引用参考文件，说明何时读取
- 对于大型参考文件（>300 行），包含目录
```

**实际文件数量级对比**：

| 文件/目录 | 行数/数量 | 加载时机 |
|----------|----------|---------|
| SKILL.md | 485 行 | 触发时加载 |
| references/schemas.md | ~200 行 | 需要写 eval 时加载 |
| scripts/ | 10 个脚本 | 黑盒执行，不加载到上下文 |
| agents/analyzer.md | ~150 行 | 需要分析结果时加载 |
| agents/comparator.md | ~120 行 | 需要盲对比时加载 |
| agents/grader.md | ~100 行 | 需要评分时加载 |

**关键**：AI 只加载需要的部分，不会一次性加载 1000+ 行。

#### 4.3.4 完整的评估体系（不只是教创建）

这是 skill-creator 最独特的部分——它不只是教怎么写 Skill，还教怎么**测试和评估**：

**Step 1: 同时启动对照实验**

```markdown
### 步骤 1：在同一轮次中启动所有运行（含 Skill 和基线）

对于每个测试用例，在同一轮次中启动两个子代理——一个启用 Skill，一个不启用。

**含 Skill 的运行：**
执行此任务：
- Skill 路径：<path-to-skill>
- 任务：<eval prompt>
- 保存输出到：<workspace>/iteration-<N>/eval-<ID>/with_skill/outputs/

**基线运行：**
- 创建新 Skill：不使用任何 Skill
- 改进现有 Skill：使用旧版本（编辑前快照）
```

**学习要点**：
- 有对照组（with/without skill），不是空谈"要测试"
- 新旧版本对比用 snapshot，确保公平
- 同时启动，避免时间差异

**Step 2: 运行期间并行工作**

```markdown
### 步骤 2：在运行期间起草断言

不要只是等待运行完成——你可以高效利用这段时间。

好的断言是客观可验证的，并且有描述性名称——
它们应该在基准查看器中清晰可读，这样浏览结果的人
能立即理解每个检查项的内容。
```

**学习要点**：利用等待时间做准备工作，提高效率

**Step 4: 评分、聚合、启动查看器**

```markdown
### Step 4: Grade, aggregate, and launch the viewer

1. Grade each run — spawn a grader subagent that reads agents/grader.md
2. Aggregate into benchmark — run the aggregation script:
   python -m scripts.aggregate_benchmark <workspace>/iteration-N
3. Do an analyst pass — read benchmark data and surface patterns
4. Launch the viewer:
   nohup python eval-viewer/generate_review.py <workspace>/iteration-N
```

**学习要点**：
- 有自动化评分（grader agent）
- 有聚合分析（benchmark.json）
- 有可视化查看器（HTML viewer）
- 有分析师视角（发现隐藏模式）

#### 4.3.5 改进策略：如何从反馈中学习

```markdown
## 如何思考改进

1. 从反馈中泛化。
   与其添加繁琐的过拟合更改，或压迫性的强制规则，
   如果遇到顽固问题，可以尝试使用不同的比喻，
   或推荐不同的工作模式。

2. 保持提示词精简。
   删除不起作用的内容。确保阅读的是对话记录，
   而不仅仅是最终输出。

3. 解释为什么。
   努力解释要求模型做每件事的**原因**。
   如果你发现自己在全部大写字母中写 ALWAYS 或 NEVER，那是黄色警告——
   重新组织并解释推理过程。

4. 寻找跨测试用例的重复工作。
   如果所有 3 个测试用例都导致子代理编写了 `create_docx.py`，
   那是强烈信号，表明 Skill 应该捆绑这个脚本。
```

**关键理念**：
1. **泛化而非过拟合**：不要写死特定案例，要找到通用模式
2. **保持精简**：删除不起作用的内容
3. **解释为什么**：用"解释原因"替代"强制规则"（ALWAYS/NEVER 是黄色警告）
4. **发现重复模式**：从测试用例中提取共性，封装成脚本

### 4.4 描述优化：独特的触发优化流程

skill-creator 有一个**独特的描述优化流程**，这是其他 Skill 没有的：

```markdown
## 描述优化

### 步骤 1：生成触发评估查询

创建 20 个评估查询——包含应该触发和不应该触发的混合。

错误示例："格式化这个数据"、"从 PDF 提取文本"、"创建图表"

正确示例："好的，我老板刚给我发了这个 xlsx 文件（在我的下载文件夹里，
叫什么'Q4 销售最终版 FINAL v2.xlsx'之类的），她想让我加一列显示
利润率百分比..."
```

**关键设计**：
- 包含应该触发和不应该触发的查询（正负样本）
- 查询要真实，包含文件路径、URL、个人上下文
- 包含小写、缩写、拼写错误、随意用语
- 负样本要是"近失"案例，不是明显无关的

**Step 3: 自动优化循环**

```bash
python -m scripts.run_loop \
  --eval-set <path-to-trigger-eval.json> \
  --skill-path <path-to-skill> \
  --max-iterations 5
```

这个脚本会：
1. 将 eval 集分为 60% 训练 + 40% 测试
2. 评估当前描述（每个查询运行 3 次获取可靠触发率）
3. 调用 Claude 提议改进
4. 重新评估，迭代最多 5 次
5. 选择**测试集分数最高**的描述（避免过拟合训练集）

**学习要点**：这是 Skill 工程中非常先进的做法，用数据驱动优化 description！

### 4.5 总结

| 维度 | 评价 | 说明 |
|------|------|------|
| 元数据 | ⭐⭐⭐⭐⭐ | "pushy" 策略，覆盖所有触发场景 |
| 流程设计 | ⭐⭐⭐⭐⭐ | 高阶流程清晰，灵活可调整 |
| 用户适配 | ⭐⭐⭐⭐⭐ | 考虑用户多样性，具体措辞指导 |
| 渐进式披露 | ⭐⭐⭐⭐⭐ | 3 层加载系统，485 行主文件 + 大量参考 |
| 评估体系 | ⭐⭐⭐⭐⭐ | 对照实验 + 自动评分 + 可视化 + 分析师 |
| 改进策略 | ⭐⭐⭐⭐⭐ | 泛化/精简/解释为什么/发现重复模式 |
| 描述优化 | ⭐⭐⭐⭐⭐ | 数据驱动优化，避免过拟合 |

**核心学习要点**：
1. description 要"pushy"，宁可多触发也不要少触发
2. 流程描述用简单 bullet points，不是复杂决策树
3. 考虑用户多样性，给出具体措辞指导
4. 渐进式披露的极致实践（485 行 + 10 脚本 + 3 Agent）
5. 完整的评估体系（对照实验 + 自动评分 + 可视化）
6. 改进策略：泛化而非过拟合，解释为什么而非强制规则
7. 独特的描述优化流程（20 个查询 + 自动迭代 + 避免过拟合）

---

**下一步**：学习 [五种设计模式](./02-Skills五种设计模式实战.md)，了解如何根据任务类型选择合适的 Skill 结构。
