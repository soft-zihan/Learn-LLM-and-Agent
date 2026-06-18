# LLMInternSkill - Codex Skill 实战学习指南

## 项目概述

**一句话总结价值**：LLMInternSkill 是一个基于 Codex Skill 的求职工具箱，它演示了如何编写结构化、可执行的 AI 技能，教学习者完整的 Skill 架构设计。

**核心亮点**：
- ✅ **完整的 Skill 文件结构**：SKILL.md + skill-references + templates + evals
- ✅ **证据边界机制**：truth-boundary、evidence-contract，真实工程中的 Skill 安全设计
- ✅ **模块化 Skill 设计**：10 个独立模块可单独调用或组合执行
- ✅ **生产级 Skill 范例**：MIT 开源、完整测试、LaTeX 导出、多角色支持

**适合谁学**：
- 想学习如何编写高质量 Codex/Claude Code Agent Skills 的开发者
- 想理解 Skill 项目工程化实践的 AI 求职者
- 对 Agent 技能系统设计感兴趣的学习者

---

## 核心架构解析

### Skill 整体架构

```
SKILL.md (入口定义)
  ├── name: 技能标识
  ├── description: 何时使用
  └── 触发条件
  │
  ├── skill-references/ (技能引用库)
  │   ├── resume-polish.md       ← 润色模块
  │   ├── resume-tailoring.md    ← JD 定制模块
  │   ├── jd-analysis.md         ← JD 分析模块
  │   ├── truth-boundary.md      ← 真实性边界
  │   ├── evidence-contract.md   ← 证据契约
  │   ├── materials-audit.md     ← 材料审计
  │   ├── interview-grilling.md  ← 面试拷打
  │   ├── answer-cards.md        ← 回答卡
  │   ├── upgrade-plan.md        ← 补证据计划
  │   ├── project-scout.md       ← 开源项目推荐
  │   └── roles/                 ← 角色专用配置
  │       ├── rag.md
  │       ├── agent.md
  │       ├── post-training.md
  │       └── ... (12 个岗位)
  │
  ├── templates/ (输出模板)
  │   ├── final-pack.md
  │   └── resume-latex/
  │
  ├── examples/ (完整示例)
  ├── evals/ (评估测试)
  └── agents/ (Agent 配置)
```

### SKILL.md 设计规范

这是项目的核心，演示了 Agent Skill 的标准写法：

```markdown
---
name: llm-intern-skill
description: Use when polishing, diagnosing, tailoring... (何时触发)
---

# LLMInternSkill

## 触发条件
Use this Skill when the user wants resume polish...

## 核心规则
Do not fabricate. Diagnose first, polish second.

## 输入格式
materials/
├── target_jd.txt
├── resume.md
└── ...

## 主工作流 (11 步)
1. Decide the mode
2. Read the target JD
3. Audit the materials
...
11. Assemble final pack

## 输出文件结构
output/
├── 01_jd_analysis.md
├── 05_resume_polish.md
└── ...

## 不可违反的规则
- Never invent internships
- Do not write "主导" when evidence only supports "参与"
```

**关键设计要点**：
1. **YAML frontmatter** - 定义 name 和 description（Agent 匹配技能的依据）
2. **明确触发条件** - 告诉 Agent "什么时候用我"
3. **主工作流** - 结构化步骤，可拆解可组合
4. **输出文件约定** - 统一输出格式，保证可重复性
5. **不可违反规则** - Skill 的安全边界

### skill-references/ 模块化设计

每个模块是独立的参考文档，Skill 根据模式调用对应模块：

| 模块 | 功能 | 触发场景 |
|------|------|---------|
| resume-polish.md | 简历润色 | "帮我润色简历" |
| jd-analysis.md | JD 分析 | 提供目标岗位描述 |
| materials-audit.md | 材料审计 | 提供 materials/ 文件夹 |
| truth-boundary.md | 真实性边界 | 任何模式都需执行 |
| evidence-contract.md | 证据契约 | 材料充足时 |
| interview-grilling.md | 面试拷打 | "帮我准备面试" |
| project-scout.md | 项目推荐 | 证据不足时 |

**模块解耦的优势**：
- Agent 可以根据用户输入动态选择执行路径
- 每个模块可独立测试和更新
- 新模块可随时添加，不影响现有流程

### 数据流向

```
用户输入 (raw resume + JD + materials/)
         ↓
   SKILL.md 解析输入
         ↓
   判断执行模式 (polish / tailoring / full / interview / scout)
         ↓
   读取对应 skill-references 模块
         ↓
   按工作流步骤执行 (11 步)
         ↓
   生成 output/ 目录 (11 个文件)
         ↓
   可选 LaTeX 导出 (PDF-ready)
```

---

## 代码逻辑主线

### Skill 运行流程

1. **Agent 接收用户请求** → 匹配 SKILL.md 的 description
2. **Agent 读取 SKILL.md** → 理解触发条件、核心规则、工作流
3. **解析输入** → 判断是 simple mode (仅简历文本) 还是 full mode (materials/ 文件夹)
4. **读取对应 skill-references** → 按需加载参考文档
5. **按工作流执行** → 11 步依次或跳过执行
6. **生成输出文件** → 写入 output/ 目录
7. **可选 LaTeX 导出** → 使用 templates/resume-latex/ 生成 PDF-ready 简历

### 关键工作流步骤

```text
Step 1: Decide the mode
  → 根据用户输入判断执行路径
  → polish only → 轻量模式
  → full materials → 完整模式
  
Step 2: Read target JD
  → 检测岗位类型 (RAG/Agent/RL/预训练/...)
  → 加载对应 roles/ 配置
  
Step 4: Set truth boundaries
  → 每条 claim 分类:
    可以写 / 谨慎写 / 补证据后写 / 不能写 / 无法判断
    
Step 6: Generate polished resume
  → 三级输出:
    conservative (保守) → 安全写法
    standard (标准) → 增强技术表达
    stronger-after-evidence (证据增强) → 补齐证据后的版本
```

### 核心机制：证据契约 (Evidence Contract)

```text
每条强声明必须包含:
├── evidence: 支撑证据 (代码、日志、论文、截图)
├── risk: 风险点 (面试可能被问穿的地方)
├── safe_wording: 安全写法 (降级的表达)
└── interview_proof: 面试如何证明

示例:
claim: "优化 RAG 检索效果"
evidence: query-doc 样例、bad case 分析、评测集
risk: 没有 NDCG/MRR 指标，面试官会追问具体优化方法
safe_wording: "围绕搜索相关性场景整理 query-doc 样例与长尾查询 bad case"
interview_proof: 展示 bad case 分析文档和评测集
```

---

## 快速上手实践

### 环境配置步骤

```bash
# 1. 确保有 Codex CLI 或 Claude Code 环境
# Codex CLI: https://github.com/openai/codex
# Claude Code: 已安装即可

# 2. Clone 项目
git clone https://github.com/couragec/llm-intern-skill.git ~/.codex/skills/llm-intern-skill

# 3. 验证安装
ls ~/.codex/skills/llm-intern-skill/SKILL.md
```

### 运行第一个示例

**方式一：简单模式（仅简历文本）**

```
使用 LLMInternSkill。
请帮我做简历润色，但不要编造经历。

我的原始简历：
[贴你的简历内容]
```

**方式二：完整模式（materials/ 文件夹）**

```bash
# 1. 准备材料
mkdir -p materials/projects
echo "你的目标岗位JD" > materials/target_jd.txt
echo "你的原始简历" > materials/resume.md

# 2. 使用 Skill
Use LLMInternSkill on ./materials.
Generate resume polish, JD fit verdict, targeted resume, interview grilling.
```

**方式三：学习 Skill 结构**

```bash
# 直接研究项目结构
cd LLMInternSkill
cat SKILL.md                    # 学习 Skill 入口写法
cat skill-references/truth-boundary.md  # 学习安全边界设计
cat skill-references/evidence-contract.md  # 学习证据契约
```

### 预期输出与验证方法

```
output/
├── 01_jd_analysis.md          ← 岗位匹配度分析
├── 02_materials_audit.md      ← 材料审计报告
├── 03_truth_boundary.md       ← 真实性边界
├── 05_resume_polish.md        ← 润色后的简历
├── 06_targeted_resume.md      ← JD 定制版简历
├── 07_interview_grilling.md   ← 面试问题清单
├── 08_answer_cards.md         ← 回答卡（危险/及格/强回答）
└── 09_upgrade_plan.md         ← 补证据计划
```

**验证方法**：
1. 检查每个文件是否包含具体的分析和输出
2. truth_boundary.md 是否标注了"可以写/谨慎写/不能写"
3. resume_polish.md 是否有 Before/After 对比
4. interview_grilling.md 是否生成了追问链

---

## 核心知识点总结

### 1. SKILL.md 标准格式

```yaml
---
name: 技能唯一标识
description: 何时使用此技能（Agent 匹配依据）
---
```

**为什么重要**：这是 Agent Skills 的元数据层，决定了 Agent 何时识别并调用你的 Skill。

### 2. Skill 工作流设计

```markdown
## Main Workflow
1. Step 1
2. Step 2
...
N. Step N
```

**为什么重要**：结构化步骤让 Agent 可以逐步执行，而不是盲目生成内容。

### 3. 模块化 skill-references

每个 `.md` 文件是一个独立参考模块，Skill 按需调用。

**为什么重要**：
- 实现"条件分支"逻辑（根据输入选择不同模块）
- 每个模块可独立测试和更新
- 新模块可随时添加

### 4. 真实性边界 (Truth Boundary)

```text
可以写 / 谨慎写 / 补证据后写 / 不能写 / 无法判断
```

**为什么重要**：这是 Skill 的安全机制，防止 Agent 编造内容。

### 5. 证据契约 (Evidence Contract)

每条强声明必须包含证据、风险、安全写法、面试证明。

**为什么重要**：工程化设计，确保输出可追溯、可验证。

### 6. 角色配置 (roles/)

针对不同岗位（RAG、Agent、预训练...）的专用配置。

**为什么重要**：Skill 的可扩展性设计，新角色只需添加新文件。

### 7. 三级输出设计

```text
conservative (保守) → safe but weak
standard (标准) → balanced
stronger-after-evidence (证据增强) → strong but needs evidence
```

**为什么重要**：给用户选择权，同时保持安全边界。

### 8. Skill 不可违反规则

```markdown
## Non-Negotiables
- Never invent internships...
- Do not write "主导" when evidence only supports "参与"...
```

**为什么重要**：这是 Skill 的"硬约束"，类似代码中的 assert()。

---

## 常见疑问解答

### Q1: Skill 和 MCP 有什么区别？

**MCP** = Model Context Protocol，是 Agent 连接外部工具的**通信协议**（怎么连接）。
**Skill** = 结构化指令文件，是教 Agent 如何做某类任务的**操作指南**（怎么做）。

两者互补：MCP 让 Agent 调用工具，Skill 教 Agent 怎么用工具完成任务。

### Q2: 为什么 SKILL.md 里不写代码？

Agent Skills 是**指令文件**，不是可执行代码。它们教 Agent：
- 何时触发
- 如何分析
- 输出什么格式

实际执行由 Agent 的 LLM 推理完成。

### Q3: skill-references/ 和 templates/ 的区别？

- **skill-references/** = 给 Agent 看的参考指南（输入侧）
- **templates/** = 给输出文件用的模板（输出侧）

### Q4: 如何编写自己的 Skill？

**最小 Skill 结构**：
```
my-skill/
├── SKILL.md          ← 必须，入口定义
└── references/       ← 可选，参考文档
```

**SKILL.md 最小内容**：
```markdown
---
name: my-skill
description: Use when the user wants to do X
---

# My Skill

## Main Workflow
1. Step 1
2. Step 2
```

### Q5: evals/ 目录是做什么的？

用于测试 Skill 的输出质量。类似单元测试：
- 输入一组测试简历
- 验证输出是否包含必需章节
- 检查是否违反了 Non-Negotiables 规则

---

## 学习路径建议

### 第一步：理解 Skill 概念（30 分钟）

1. 阅读 README.md 了解项目定位
2. 阅读 SKILL.md 学习入口格式
3. 理解"何时触发 → 如何执行 → 输出什么"的流程

### 第二步：分析模块化设计（1 小时）

1. 逐个阅读 skill-references/ 下的模块
2. 理解每个模块的功能和触发条件
3. 观察模块之间的调用关系

### 第三步：实践编写自己的 Skill（2 小时）

1. 复制一个简单模块（如 resume-polish.md）
2. 修改为你自己的任务场景
3. 编写对应的 SKILL.md
4. 用 Codex/Claude Code 测试

### 第四步：理解安全机制（1 小时）

1. 阅读 truth-boundary.md 理解真实性边界
2. 阅读 evidence-contract.md 理解证据契约
3. 思考如何在你的 Skill 中应用安全机制

---

## 与教程目录的关联

本项目与以下教程内容强关联：

| 教程章节 | 关联点 |
|---------|-------|
| `05-工具调用与RAG/01-工具调用技术` | Skill 是工具调用的上层抽象 |
| `07-多Agent与Agent工程化/03-MCP协议与Skills` | MCP + Skills 是完整工具系统 |
| `08-多模态与前沿技术/04-学习路线与前沿展望` | Agent Skills 是 2026 前沿技术 |

---

## 总结

**LLMInternSkill 不是简历工具，而是 Skill 工程教科书**。它展示了：

1. ✅ **SKILL.md 标准写法** - 元数据 + 触发条件 + 工作流
2. ✅ **模块化 Skill 设计** - skill-references 按需调用
3. ✅ **安全机制设计** - truth-boundary + evidence-contract
4. ✅ **生产级 Skill 实践** - 测试 + 模板 + 多角色支持

**面试中可以讲的亮点**：
- 我研究过一个完整的 Codex Skill 项目，理解 Skill 的架构设计
- 我知道如何编写结构化的 Agent Skills，而不是简单写 Prompt
- 我理解 Skill 的安全机制，能设计证据边界和真实性校验
