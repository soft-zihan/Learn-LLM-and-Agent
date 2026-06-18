# Nanobot 智能体设计实战 - 超详细笔记

>
> 📅 **最后更新**：2026-06-15 | 最新版本：v0.2.1 (Workbench Release) | GitHub Stars：44K+

---

## 目录

### 第一部分：AI Agent 基础与 Nanobot 概览
- [1. 什么是 AI Agent](#1-什么是-ai-agent)
  - [1.1 LLM vs Agent 的本质区别](#11-llm-vs-agent-的本质区别)
  - [1.2 Agent 核心三要素](#12-agent-核心三要素)
  - [1.3 ReAct 框架详解](#13-react-框架详解)
  - [1.4 主流 Agent 框架对比](#14-主流-agent-框架对比)
- [2. Nanobot 项目概览](#2-nanobot-项目概览)
  - [2.1 定位与核心优势](#21-定位与核心优势)
  - [2.2 快速开始](#22-快速开始)
  - [2.3 配置驱动架构](#23-配置驱动架构)
  - [2.4 2026 年新特性总览](#24-2026-年新特性总览)

### 第二部分：架构与源码深度解析
- [3. 五层架构深入解析](#3-五层架构深入解析)
  - [3.1 五层架构总览](#31-五层架构总览)
  - [3.2 四大核心模块](#32-四大核心模块)
  - [3.3 10 个设计模式](#33-10-个设计模式)
- [4. 源码逐行解读](#4-源码逐行解读)
  - [4.1 AgentLoop 核心循环](#41-agentloop-核心循环)
  - [4.2 AgentRunner 执行器](#42-agentrunner-执行器)
  - [4.3 MemoryStore 记忆存储](#43-memorystore-记忆存储)
  - [4.4 MessageBus 消息总线](#44-messagebus-消息总线)

### 第三部分：协议与系统集成
- [5. MCP 协议深度集成](#5-mcp-协议深度集成)
  - [5.1 MCP 协议概述](#51-mcp-协议概述)
  - [5.2 M×N 问题与三角架构](#52-mn-问题与三角架构)
  - [5.3 MCP 三大原语](#53-mcp-三大原语)
  - [5.4 MCPToolWrapper 实现](#54-mcptoolwrapper-实现)
- [6. 安装与上手实战](#6-安装与上手实战)
  - [6.1 环境配置](#61-环境配置)
  - [6.2 第一次对话](#62-第一次对话)
  - [6.3 Bot 定制](#63-bot-定制)

### 第四部分：核心系统详解
- [7. 记忆系统设计实战](#7-记忆系统设计实战)
  - [7.1 为什么 Agent 需要记忆](#71-为什么-agent-需要记忆)
  - [7.2 双层记忆架构](#72-双层记忆架构)
  - [7.3 MEMORY.md 长期记忆](#73-memorymd-长期记忆)
  - [7.4 HISTORY.md 历史时间线](#74-historymd-历史时间线)
  - [7.5 MemoryConsolidator 压缩机制](#75-memoryconsolidator-压缩机制)
  - [7.6 Session JSONL 短期记忆](#76-session-jsonl-短期记忆)
  - [7.7 与其他框架对比](#77-与其他框架对比)
- [8. Skills 与 Tools 系统](#8-skills-与-tools-系统)
  - [8.1 Skill vs Tool 区别](#81-skill-vs-tool-区别)
  - [8.2 SKILL.md 格式规范](#82-skillmd-格式规范)
  - [8.3 渐进披露机制](#83-渐进披露机制)
  - [8.4 ToolRegistry 统一注册](#84-toolregistry-统一注册)
  - [8.5 内置工具完整列表](#85-内置工具完整列表)
  - [8.6 MCP 工具集成](#86-mcp-工具集成)
  - [8.7 工具安全机制](#87-工具安全机制)

### 第五部分：高级特性与多平台
- [9. 多平台接入架构](#9-多平台接入架构)
  - [9.1 MessageBus 双队列](#91-messagebus-双队列)
  - [9.2 BaseChannel 适配器模式](#92-basechannel-适配器模式)
  - [9.3 session_key 会话隔离](#93-session_key-会话隔离)
  - [9.4 流式输出优化](#94-流式输出优化)
  - [9.5 8+ 平台接入指南](#95-8-平台接入指南)
- [10. SubAgent 与 Cron 定时任务](#10-subagent-与-cron-定时任务)
  - [10.1 SubAgent 后台任务机制](#101-subagent-后台任务机制)
  - [10.2 Cron 定时调度系统](#102-cron-定时调度系统)
  - [10.3 Heartbeat 心跳服务](#103-heartbeat-心跳服务)
  - [10.4 防递归机制](#104-防递归机制)

### 第六部分：生产部署与实战
- [11. 安全与生产部署](#11-安全与生产部署)
  - [11.1 安全防御体系](#111-安全防御体系)
  - [11.2 Docker 部署方案](#112-docker-部署方案)
  - [11.3 多实例架构](#113-多实例架构)
  - [11.4 密钥管理](#114-密钥管理)
  - [11.5 监控与备份](#115-监控与备份)
- [12. 真实项目案例](#12-真实项目案例)
  - [12.1 智能运维助手](#121-智能运维助手)
  - [12.2 多平台智能客服](#122-多平台智能客服)
  - [12.3 自定义 MCP Server](#123-自定义-mcp-server)
- [13. 从零手写教学版 Agent](#13-从零手写教学版-agent)
  - [13.1 6 章渐进式教程](#131-6-章渐进式教程)
  - [13.2 核心设计决策](#132-核心设计决策)
- [14. 面试高频考点总结](#14-面试高频考点总结)
  - [14.1 架构设计题](#141-架构设计题)
  - [14.2 技术深度题](#142-技术深度题)
  - [14.3 系统设计题](#143-系统设计题)
  - [14.4 实战项目题](#144-实战项目题)

---

## 1. 什么是 AI Agent

### 1.1 LLM vs Agent 的本质区别

**核心区别**：LLM 是被动的语言模型,Agent 是主动的决策系统。

```
LLM (Large Language Model):
  输入 Prompt → 预测下一个 token → 输出文本
  特点：被动响应,无状态,无外部能力

Agent (AI Agent):
  目标 → 规划 → 执行工具 → 观察结果 → 调整策略 → 达成目标
  特点：主动决策,有状态,有工具使用能力,能循环迭代
```

**关键差异对比表**：

| 维度 | LLM | Agent |
|------|-----|-------|
| **交互模式** | 单次问答 | 多轮循环 (ReAct) |
| **状态管理** | 无状态 (每次独立) | 有状态 (跨轮记忆) |
| **外部能力** | 仅文本生成 | 工具调用 (API/文件/Shell) |
| **决策能力** | 无 (被动生成) | 有 (主动规划) |
| **错误恢复** | 无法自我纠正 | 可基于工具结果调整 |
| **任务复杂度** | 简单问答 | 复杂多步骤任务 |
| **典型应用** | 聊天机器人 | 智能助手/自动化运维/客服系统 |

**面试话术**：
> "LLM 就像一本百科全书——你问它答,但它不会主动做任何事情。Agent 则是一个有手有脑的助手——它能理解你的目标,制定计划,使用工具,观察结果,并根据反馈调整策略。关键区别在于**主动决策**和**工具使用能力**。"

### 1.2 Agent 核心三要素

所有 AI Agent 都基于三个核心要素构建：

```
┌────────────────────────────────────────────────────┐
│              AI Agent 核心三要素                     │
│                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │  Planning     │  │  Memory      │  │  Tool    │ │
│  │  (规划)       │  │  (记忆)      │  │  Use     │ │
│  │              │  │              │  │  (工具)   │ │
│  │ · 目标分解    │  │ · 长期记忆    │  │ · 工具发现│ │
│  │ · 任务排序    │  │ · 短期记忆    │  │ · 参数构造│ │
│  │ · 策略选择    │  │ · 上下文维护  │  │ · 结果解析│ │
│  │ · 反思调整    │  │ · 历史检索    │  │ · 错误处理│ │
│  └──────────────┘  └──────────────┘  └──────────┘ │
│        │                  │                │       │
│        └──────────────────┼────────────────┘       │
│                           │                        │
│                   三者协作实现智能行为               │
└────────────────────────────────────────────────────┘
```

#### 1.2.1 Planning (规划)

Planning 是 Agent 的"大脑"——决定做什么、按什么顺序做、失败了怎么办。

**规划的核心机制**：

```python
# 简化的规划循环（伪代码）
def plan_and_execute(goal: str) -> str:
    """Agent 的规划与执行循环"""
    
    # 1. 目标分解
    subtasks = decompose_goal(goal)  # LLM 将复杂目标拆成子任务
    
    # 2. 执行循环
    for task in subtasks:
        while not task.is_completed:
            # 思考下一步
            next_action = llm.think(current_state, task)
            
            # 执行工具
            result = tool_registry.execute(next_action.tool, next_action.args)
            
            # 观察结果并调整
            if result.success:
                task.update_progress(result)
            else:
                # 反思并调整策略
                strategy = llm.reflect(result.error)
                next_action = strategy.adjusted_action()
    
    return "任务完成"
```

**规划的设计模式**：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **单步规划** | 每次只决定下一步 | 简单任务,工具少 |
| **分层规划** | 先高层计划,再细化执行 | 复杂多步骤任务 |
| **反思规划** | 执行后反思,调整后续计划 | 不确定性高的任务 |
| **并行规划** | 识别可并行的子任务 | 独立子任务多的场景 |

#### 1.2.2 Memory (记忆)

Memory 是 Agent 的"记忆系统"——记住用户偏好、历史对话、关键决策。

**记忆的层次结构**：

```
记忆系统 = 短期记忆 + 长期记忆 + 工作记忆

短期记忆 (Session):
  · 当前会话的完整对话历史
  · 生命周期：会话结束即清除
  · 存储格式：JSONL (JSON Lines)
  · 容量限制：上下文窗口大小 (128K tokens)

长期记忆 (MEMORY.md):
  · 跨会话的核心信息（用户偏好、重要决策）
  · 生命周期：永久保存,持续更新
  · 存储格式：Markdown 文件
  · 注入方式：始终注入 System Prompt

工作记忆 (Context):
  · 当前思考的上下文
  · 包含：System Prompt + 工具定义 + 技能摘要
  · 每轮对话都重建
```

**记忆的核心矛盾**：

```
      完整性               成本
   (记住所有)          (节省 token)
       │                   │
       └────── 矛盾 ──────┘
               │
            实时性
        (快速检索)

1. 完整性 vs 成本：记住所有信息需要巨大的上下文窗口,费用高昂
2. 完整性 vs 实时性：存储所有信息使检索变慢
3. 成本 vs 实时性：压缩信息虽然省钱,但可能丢失重要细节
```

#### 1.2.3 Tool Use (工具使用)

Tool Use 是 Agent 的"手脚"——让 LLM 能够执行实际操作。

**工具使用的完整流程**：

```
用户消息 → Agent 构建 Prompt → LLM 决定调用工具
                                    │
                                    ▼
                          解析 tool_calls JSON
                                    │
                                    ▼
                          ToolRegistry 查找 handler
                                    │
                                    ▼
                          执行工具 (文件/Shell/API)
                                    │
                                    ▼
                          将结果封装为 tool message
                                    │
                                    ▼
                          返回给 LLM 继续推理
                                    │
                                    ▼
                          LLM 基于结果决定下一步
                          (继续调用工具 / 生成回复)
```

**工具调用示例**：

```python
# LLM 返回的工具调用（JSON 格式）
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": '{"path": "src/main.py"}'
      }
    }
  ]
}

# Agent 执行工具
result = await tool_registry.execute(
    name="read_file",
    arguments={"path": "src/main.py"}
)

# 将结果返回给 LLM
{
  "role": "tool",
  "content": "import os\nimport sys\n\ndef main():...",
  "tool_call_id": "call_abc123"
}
```

### 1.3 ReAct 框架详解

**ReAct** = **Re**asoning + **Act**ing —— 推理与行动交替进行的框架。

#### 1.3.1 ReAct 的核心思想

ReAct 框架的核心是**交替进行推理和行动**：

```
思考 (Thought) → 行动 (Action) → 观察 (Observation) → 思考 → 行动 → 观察 → ... → 最终答案

Thought:  我需要先了解用户的代码结构
Action:   使用 list_dir 查看项目目录
Observation: 返回了 src/, tests/, config/ 三个目录

Thought:  现在我需要查看主文件的内容
Action:   使用 read_file 读取 src/main.py
Observation: 返回了文件内容...

Thought:  我发现代码中缺少错误处理,我来添加
Action:   使用 edit_file 修改代码
Observation: 文件已成功修改

Thought:  任务完成,我可以给用户回复了
Action:   生成最终回复
```

#### 1.3.2 ReAct 循环的源码实现

```python
# 简化的 ReAct 循环（Nanobot 核心逻辑）
class AgentLoop:
    async def run(self):
        """主循环：持续处理消息"""
        while True:
            # 1. 从入站队列获取消息
            msg = await self.bus.inbound.get()
            
            # 2. 异步处理（不阻塞其他消息）
            asyncio.create_task(self._handle_message(msg))
    
    async def _handle_message(self, msg: InboundMessage):
        """处理单条消息的 ReAct 循环"""
        
        # 构建初始 Prompt
        messages = self._build_prompt(msg)
        
        # ReAct 循环：最多 40 次迭代
        for iteration in range(self.max_iterations):
            # ─── Thought（思考）+ Action（行动）───
            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=self.tool_registry.get_definitions()
            )
            
            # 检查是否有工具调用
            if response.has_tool_calls:
                # ─── Observation（观察）───
                results = []
                for tool_call in response.tool_calls:
                    result = await self.tool_registry.execute(
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments)
                    )
                    results.append(result)
                
                # 将工具结果加入消息历史
                messages.append(tool_results_to_message(results))
                
            else:
                # 没有工具调用,生成最终回复
                await self._send_reply(response.content)
                break
        
        # 保存会话历史
        self._save_session(messages)
```

**关键设计决策**：

| 决策点 | Nanobot 的选择 | 理由 |
|--------|---------------|------|
| **最大迭代次数** | 40 次 | 足够完成复杂任务,防止失控 |
| **停止条件** | 无 tool_calls | LLM 判断任务完成 |
| **错误处理** | 工具结果包含错误信息 | 让 LLM 自我纠正 |
| **并发控制** | asyncio.Semaphore(3) | 限制同时运行的会话数 |

#### 1.3.3 ReAct vs 其他框架

| 框架 | 核心思想 | 优点 | 缺点 |
|------|---------|------|------|
| **ReAct** | 推理+行动交替 | 简单直观,可解释性强 | 可能陷入循环 |
| **ReWOO** | 先规划再并行执行 | 减少 LLM 调用次数 | 灵活性差 |
| **Plan-and-Execute** | 显式规划+执行分离 | 适合复杂任务 | 实现复杂 |
| **Reflexion** | 自我反思+重试 | 错误恢复能力强 | token 消耗大 |
| **LATS** | 树搜索+回溯 | 全局最优 | 计算成本高 |

### 1.4 主流 Agent 框架对比

#### 1.4.1 框架对比总览

| 框架 | 语言 | 代码量 | Stars | 定位 | 学习曲线 |
|------|------|--------|-------|------|---------|
| **Nanobot** | Python | ~6000 行 | 44K+ | 超轻量个人助手 + WebUI | ⭐ 低 |
| **LangChain** | Python/JS | 50 万+ 行 | 90K+ | 企业级全功能 | ⭐⭐⭐ 高 |
| **AutoGPT** | Python | 2 万+ 行 | 160K+ | 自主任务执行 | ⭐⭐ 中 |
| **CrewAI** | Python | 1 万+ 行 | 20K+ | 多 Agent 协作 | ⭐⭐ 中 |
| **OpenAI Agents** | Python | 官方 SDK | N/A | OpenAI 生态 | ⭐ 低 |
| **MetaGPT** | Python | 2 万+ 行 | 45K+ | 软件工程模拟 | ⭐⭐⭐ 高 |

#### 1.4.2 Nanobot 的核心优势

```
Nanobot 的设计哲学：Less is More

✅ ~6000 行代码 —— 一个周末就能读完理解（核心逻辑仍保持极简）
✅ 零外部依赖 —— 不需要数据库、向量库、消息队列
✅ 配置驱动 —— config.json + AGENTS.md + SOUL.md
✅ 文件存储 —— MEMORY.md, HISTORY.md, 人类可读可编辑
✅ MCP 原生支持 —— 标准化工具集成协议
✅ 多平台接入 —— Telegram/Discord/飞书/钉钉/微信/Slack/Email/Signal 等 12+ 平台
✅ 内置 WebUI —— 2026 年新增 Agent Workbench，可视化交互
✅ 生产就绪 —— Docker 部署、安全沙箱、监控备份、Langfuse 可观测性
✅ Dream Memory —— 两阶段记忆系统，自动学习与记忆压缩
✅ Goal 系统 —— 长期目标跨轮次保持，支持持续任务
✅ Plugin SDK —— 2026 年新架构，支持第三方插件扩展

❌ 不适合的场景：
   - 海量文档检索（1000+ 页,应用向量库）
   - 多用户共享知识库（应用数据库）
   - 需要复杂关系推理（应用知识图谱）
   - 实时高并发场景（应用消息队列）
   - 企业级多租户 SaaS（应用 LangChain/AutoGen）
```

#### 1.4.3 面试话术：为什么选择 Nanobot

> "在选择 Agent 框架时,我对比了 LangChain、AutoGPT 和 Nanobot。LangChain 功能强大但学习曲线陡峭,50 万行代码意味着出了问题很难排查。AutoGPT 自主性强但可控性差,容易失控消耗大量 API 费用。
>
> 我最终选择 Nanobot 是因为它的**极简设计哲学**——~6000 行代码,一个周末就能完全理解；零外部依赖,纯文件存储,完全透明可调试；配置驱动,通过 Markdown 文件定义 Agent 行为,而非写代码。
>
> 对于个人助手和中小团队场景,Nanobot 的简单性反而是最大的优势——快速上手、易于定制、维护成本低。而且它原生支持 MCP 协议和多平台接入,2026 年还新增了 WebUI 工作台、Dream Memory 和 Goal 系统,这在实际项目中非常重要。"

---

## 2. Nanobot 项目概览

### 2.1 定位与核心优势

#### 2.1.1 项目定位

```
Nanobot = Nano (超小) + Bot (机器人)

定位：~6000 行代码的超轻量级 AI Agent 框架（v0.2.1）
目标：让每个人都能理解和定制 AI Agent，拥有自己的个人助手
GitHub：HKUDS/nanobot (44K+ Stars)
最新版本：v0.2.1 - The Workbench Release (2026-06-01)
```

#### 2.1.2 六大核心特性

| 特性 | 说明 | 技术实现 |
|------|------|---------|
| **超轻量** | ~6000 行核心代码 | 无冗余抽象,每个函数都有用 |
| **零依赖** | 不需要数据库/向量库 | 纯文件存储 (Markdown + JSONL) |
| **配置驱动** | 改配置而非改代码 | config.json + AGENTS.md + SOUL.md |
| **MCP 原生** | 标准化工具集成 | MCP 协议支持,工具即插即用 |
| **多平台** | 12+ 平台同时接入 | MessageBus + BaseChannel 适配器 |
| **生产就绪** | 安全/部署/监控 | 沙箱、Docker、备份、日志、Langfuse |

#### 2.1.3 2026 年新增特性

| 特性 | 版本 | 说明 |
|------|------|------|
| **WebUI Workbench** | v0.2.1 | 内置浏览器 UI，Thought/Response 时间线，文件编辑活动可视化 |
| **Goal 系统** | v0.2.0 | `/goal` 命令保持长期目标跨轮次，支持持续任务 |
| **Dream Memory** | v0.1.5 | 两阶段记忆系统，自动学习与记忆压缩 |
| **Plugin SDK** | v0.1.4+ | 第三方插件扩展，Channel/Tool/Provider 可插拔 |
| **Embeddable Core** | v0.1.4+ | 可作为库导入 `from nanobot import Agent` |
| **Langfuse 集成** | v0.1.5 | 生产级可观测性与追踪 |
| **Model Fallback** | v0.2.0 | 主模型失败自动切换到备用模型 |
| **Image Generation** | v0.2.0 | 支持 Gemini/MiniMax/Zhipu 等多平台图像生成 |
| **Step Plan** | v0.1.5 | 多步骤规划与进度可视化 |
| **CLI Apps + MCP** | v0.1.5 | 统一 CLI 应用与 MCP 扩展 |

### 2.2 快速开始

#### 2.2.1 安装与配置

```bash
# 方式 1: 一键安装脚本（推荐）
# macOS / Linux
sh -c "$(curl -fsSL https://raw.githubusercontent.com/HKUDS/nanobot/main/scripts/install.sh)"

# Windows PowerShell
irm https://raw.githubusercontent.com/HKUDS/nanobot/main/scripts/install.ps1 | iex

# 方式 2: pip 安装（稳定版）
pip install nanobot-ai

# 方式 3: 从源码安装（最新版）
git clone https://github.com/HKUDS/nanobot.git
cd nanobot
pip install -e .

# 创建 workspace
mkdir my-bot && cd my-bot

# 初始化配置（交互式引导）
nanobot onboard --wizard
# 选择 LLM Provider、输入 API Key、配置模型

# 启动 Agent（CLI 模式）
nanobot

# 启动 WebUI（v0.2.1 新增）
nanobot --webui
# 访问 http://localhost:8080
```

#### 2.2.2 配置文件详解

```json
// config.json - Nanobot 的核心配置
{
  "agents": {
    "defaults": {
      "workspace": "/path/to/workspace",
      "model": "gpt-4o",
      "provider": "openai",
      "max_tokens": 8192,
      "context_window_tokens": 64000,
      "temperature": 0.3,
      "max_tool_iterations": 30,
      "restrict_to_workspace": true
    }
  },
  "providers": {
    "openai": {
      "api_key": "${OPENAI_API_KEY}",
      "api_base": "https://api.openai.com/v1"
    },
    "anthropic": {
      "api_key": "${ANTHROPIC_API_KEY}"
    }
  },
  "channels": {
    "telegram": {
      "bot_token": "${TELEGRAM_BOT_TOKEN}"
    }
  },
  "tools": {
    "exec": {
      "allowed": true,
      "timeout": 60
    },
    "web_search": {
      "provider": "brave",
      "api_key": "${BRAVE_API_KEY}"
    }
  }
}
```

**关键配置项说明**：

| 配置项 | 说明 | 推荐值 |
|--------|------|--------|
| `max_tokens` | 单次回复最大 token 数 | 开发: 16384, 生产: 8192 |
| `context_window_tokens` | 上下文窗口大小 | 开发: 128000, 生产: 64000 |
| `temperature` | 输出随机性 | 开发: 0.7, 生产: 0.3 |
| `max_tool_iterations` | 单次对话最大工具调用次数 | 主Agent: 30-40, SubAgent: 15 |
| `restrict_to_workspace` | 启用 workspace 沙箱 | 始终: true |

### 2.3 配置驱动架构

#### 2.3.1 配置文件体系

```
Nanobot 配置 = config.json + AGENTS.md + SOUL.md + USER.md + TOOLS.md

config.json (系统配置):
  · LLM Provider 配置
  · 平台通道配置
  · 工具配置
  · 安全配置

AGENTS.md (Agent 身份):
  · Agent 的角色定义
  · 核心原则
  · 工作流程
  · 可用技能

SOUL.md (Agent 灵魂):
  · 个性特征
  · 语言风格
  · 价值观和边界

USER.md (用户信息):
  · 用户名和背景
  · 偏好设置
  · 特殊需求

TOOLS.md (工具说明):
  · 工具使用指南
  · 安全约束
  · 最佳实践
```

#### 2.3.2 AGENTS.md 示例

```markdown
# 智能运维助手

## 身份
你是一个专业的 Linux 服务器运维专家。

## 核心原则
1. 安全第一：只执行白名单内的命令
2. 记录一切：所有操作都记录到 MEMORY.md
3. 主动告警：发现异常立即通知

## 安全白名单
允许执行的命令：
- 系统状态查看：top, free, df, uptime
- 日志查看：tail, grep (仅限日志目录)
- 网络诊断：ping, curl, netstat

## 告警阈值
| 指标 | 警告 | 严重 |
|------|------|------|
| CPU | > 80% | > 95% |
| 内存 | > 85% | > 95% |
| 磁盘 | > 80% | > 90% |
```

**面试话术**：
> "Nanobot 的配置驱动架构是我认为最优雅的设计之一。它通过 Markdown 文件定义 Agent 的行为,而非写代码。这意味着非程序员也能定制 Agent——只需编辑 AGENTS.md 就能改变 Agent 的角色和原则。这比 LangChain 那种需要写大量 Python 代码的方式灵活得多,而且完全透明可审计。"

---

## 3. 五层架构深入解析

### 3.1 五层架构总览
```
Nanobot 的整体架构可以分为五层（v0.2.1 更新）：

```
┌─────────────────────────────────────────────────────────┐
│                  Nanobot 五层架构                        │
│                                                         │
│  Layer 5: UI 层 (2026 年扩展)                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │Telegram  │ │Discord   │ │WebUI     │ │CLI       │  │
│  │Bot       │ │Bot       │ │Workbench │ │Terminal  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│       │            │            │            │         │
│  ┌────▼────────────▼────────────▼────────────▼─────┐   │
│  │  Layer 4: Gateway 层 (MessageBus + Plugin SDK)   │   │
│  │  ┌──────────────┐    ┌──────────────┐           │   │
│  │  │ Inbound Queue│    │Outbound Queue│           │   │
│  │  │ (入站队列)    │    │(出站队列)     │           │   │
│  │  └──────┬───────┘    └──────▲───────┘           │   │
│  └─────────┼───────────────────┼───────────────────┘   │
│            │                   │                       │
│  ┌─────────▼───────────────────┴──────────────────┐    │
│  │  Layer 3: Core Agent 层 (+ Goal/Dream Memory)   │    │
│  │  ┌──────────────┐  ┌──────────────────────┐    │    │
│  │  │  AgentLoop   │  │  AgentRunner         │    │    │
│  │  │  (消息循环)   │  │  (ReAct 执行器)       │    │    │
│  │  └──────────────┘  └──────────────────────┘    │    │
│  └───────────────────────┬────────────────────────┘    │
│                          │                             │
│  ┌───────────────────────▼────────────────────────┐    │
│  │  Layer 2: Provider 层 (25+ 模型支持)            │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐   │    │
│  │  │ OpenAI   │ │Anthropic │ │DeepSeek      │   │    │
│  │  │ GPT-4o/5 │ │Claude 4  │ │DeepSeek-V4   │   │    │
│  │  └──────────┘ └──────────┘ └──────────────┘   │    │
│  └───────────────────────┬────────────────────────┘    │
│                          │                             │
│  ┌───────────────────────▼────────────────────────┐    │
│  │  Layer 1: Tool 层 (+ MCP + Plugin Tools)        │    │
│  │  ┌──────────────┐  ┌──────────────────────┐    │    │
│  │  │ Built-in Tools│  │ MCP Tools            │    │    │
│  │  │ read_file    │  │ mcp_weather          │    │    │
│  │  │ exec         │  │ mcp_database          │    │    │
│  │  │ web_search   │  │ mcp_github           │    │    │
│  │  └──────────────┘  └──────────────────────┘    │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**数据流说明**：

```
用户在 Telegram 发送消息
  → TelegramChannel 接收
  → 转换为 InboundMessage
  → 发布到 MessageBus Inbound Queue
  → AgentLoop 消费消息
  → AgentRunner 执行 ReAct 循环
  → 调用 Provider (OpenAI/Anthropic)
  → LLM 返回工具调用
  → ToolRegistry 执行工具
  → 工具结果返回给 LLM
  → LLM 生成回复
  → 创建 OutboundMessage
  → 发布到 MessageBus Outbound Queue
  → ChannelManager 路由到 TelegramChannel
  → 发送到 Telegram
```

### 3.2 四大核心模块

Nanobot 的核心功能由四个模块实现：

#### 3.2.1 AgentLoop (消息循环)

AgentLoop 是 Nanobot 的心脏——持续从消息队列中消费消息并触发处理。

```python
class AgentLoop:
    """Agent 主循环 - 消息处理引擎"""
    
    def __init__(self, bus, provider, workspace, tool_registry, 
                 memory_store, context_builder):
        self.bus = bus                    # MessageBus 消息总线
        self.provider = provider          # LLM Provider
        self.workspace = workspace        # Workspace 路径
        self.tool_registry = tool_registry # 工具注册表
        self.memory_store = memory_store  # 记忆存储
        self.context_builder = context_builder  # 上下文构建器
        
        # 并发控制
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._concurrency_gate = asyncio.Semaphore(3)  # 最多3个并发会话
    
    async def run(self):
        """主循环：持续处理入站消息"""
        while True:
            # 1. 从入站队列阻塞等待消息
            msg = await self.bus.inbound.get()
            
            # 2. 异步处理（不阻塞后续消息）
            asyncio.create_task(self._handle_message(msg))
    
    async def _handle_message(self, msg: InboundMessage):
        """处理单条入站消息"""
        
        # 获取会话锁（防止同一会话并发处理）
        session_key = msg.session_key
        if session_key not in self._session_locks:
            self._session_locks[session_key] = asyncio.Lock()
        
        async with self._session_locks[session_key]:
            # 并发门控（限制全局并发数）
            async with self._concurrency_gate:
                await self._process_message(msg)
    
    async def _process_message(self, msg: InboundMessage):
        """实际的消息处理逻辑"""
        
        # 1. 构建完整 Prompt
        messages = await self.context_builder.build(
            session_key=msg.session_key,
            user_message=msg.content,
            media=msg.media
        )
        
        # 2. 创建 AgentRunner 并执行
        runner = AgentRunner(
            provider=self.provider,
            tool_registry=self.tool_registry,
            memory_store=self.memory_store,
            messages=messages,
            session_key=msg.session_key
        )
        
        # 3. 执行 ReAct 循环
        response = await runner.run()
        
        # 4. 发送回复
        outbound = OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=response,
            reply_to=msg.metadata.get("message_id")
        )
        await self.bus.publish_outbound(outbound)
```

**关键设计决策**：

| 设计点 | 实现 | 理由 |
|--------|------|------|
| **会话锁** | `asyncio.Lock` per session | 防止同一会话的消息乱序 |
| **并发门控** | `asyncio.Semaphore(3)` | 限制同时运行的会话数,防止资源耗尽 |
| **异步处理** | `asyncio.create_task` | 不阻塞后续消息,提高吞吐量 |
| **无限循环** | `while True` | 持续监听消息队列 |

#### 3.2.2 AgentRunner (ReAct 执行器)

AgentRunner 是 ReAct 框架的具体实现——执行思考-行动-观察循环。

```python
class AgentRunner:
    """ReAct 循环执行器"""
    
    def __init__(self, provider, tool_registry, memory_store, 
                 messages, session_key, max_iterations=40):
        self.provider = provider
        self.tool_registry = tool_registry
        self.memory_store = memory_store
        self.messages = messages
        self.session_key = session_key
        self.max_iterations = max_iterations
    
    async def run(self) -> str:
        """执行 ReAct 循环,返回最终回复"""
        
        for iteration in range(self.max_iterations):
            # ─── Step 1: Thought + Action ───
            # 调用 LLM,传入工具定义
            response = await self.provider.chat_with_retry(
                messages=self.messages,
                tools=self.tool_registry.get_definitions()
            )
            
            # ─── Step 2: 检查是否有工具调用 ───
            if response.has_tool_calls:
                # ─── Step 3: Observation ───
                tool_results = []
                for tool_call in response.tool_calls:
                    # 执行工具
                    result = await self.tool_registry.execute(
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments)
                    )
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                
                # 将工具结果加入消息历史
                self.messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": response.tool_calls
                })
                
                for tool_result in tool_results:
                    self.messages.append({
                        "role": "tool",
                        "content": tool_result["content"],
                        "tool_call_id": tool_result["tool_call_id"]
                    })
                
                # 检查是否需要记忆压缩
                if self.memory_store.should_consolidate(self.messages):
                    await self.memory_store.consolidate(self.messages)
            
            else:
                # ─── Step 4: 最终回复 ───
                # LLM 没有工具调用,说明任务完成
                final_response = response.content
                
                # 保存会话历史
                await self.memory_store.save_session(
                    self.session_key, self.messages
                )
                
                return final_response
        
        # 达到最大迭代次数
        return "已达到最大处理次数,请稍后再试。"
```

**ReAct 循环流程图**：

```
用户消息
  │
  ▼
┌─────────────────────────┐
│ 构建完整 Prompt          │
│ = System Prompt          │
│   + MEMORY.md            │
│   + Session History      │
│   + 工具定义              │
│   + 技能摘要              │
│   + 当前用户消息          │
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│ 调用 LLM API             │
│ (传入工具定义)            │
└─────────────────────────┘
  │
  ├─────────────────┐
  │                 │
  ▼                 ▼
有 tool_calls    无 tool_calls
  │                 │
  ▼                 ▼
┌─────────────┐  ┌─────────────┐
│ 执行工具     │  │ 返回最终回复 │
│ (Observation)│  │ 保存会话    │
└─────────────┘  └─────────────┘
  │
  ▼
将结果加入消息历史
  │
  ▼
检查是否需压缩记忆
  │
  ▼
循环继续 (回到调用 LLM)
```

#### 3.2.3 MemoryStore (记忆存储)

MemoryStore 管理所有记忆相关操作——加载、保存、压缩。

```python
class MemoryStore:
    """记忆存储管理器"""
    
    def __init__(self, workspace: str, context_window_tokens: int):
        self.workspace = workspace
        self.context_window_tokens = context_window_tokens
        self.consolidator = MemoryConsolidator()
    
    def load_memory(self) -> str:
        """加载 MEMORY.md (长期记忆)"""
        path = os.path.join(self.workspace, "memory", "MEMORY.md")
        if os.path.exists(path):
            return read_file(path)
        return ""
    
    def load_session(self, session_key: str) -> list:
        """加载 Session JSONL (短期记忆)"""
        path = os.path.join(self.workspace, "sessions", 
                           f"{session_key}.jsonl")
        messages = []
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        messages.append(json.loads(line))
        return messages
    
    def save_session(self, session_key: str, messages: list):
        """保存会话历史到 JSONL"""
        path = os.path.join(self.workspace, "sessions",
                           f"{session_key}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "a") as f:
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    
    def should_consolidate(self, messages: list) -> bool:
        """检查是否需要压缩记忆"""
        total_tokens = estimate_prompt_tokens_chain(messages)
        return total_tokens > self.context_window_tokens
    
    async def consolidate(self, messages: list):
        """执行记忆压缩"""
        await self.consolidator.consolidate(
            current_memory=self.load_memory(),
            messages=messages,
            workspace=self.workspace
        )
```

#### 3.2.4 MessageBus (消息总线)

MessageBus 实现生产者/消费者解耦——双队列架构。

```python
class MessageBus:
    """异步消息总线 - 双队列设计"""
    
    def __init__(self):
        self._inbound_queue = asyncio.Queue()   # 入站队列
        self._outbound_queue = asyncio.Queue()  # 出站队列
    
    # ─── 入站 API ───
    
    async def publish_inbound(self, message: InboundMessage):
        """Channel 将用户消息发布到入站队列"""
        await self._inbound_queue.put(message)
    
    async def consume_inbound(self) -> InboundMessage:
        """Agent 从入站队列消费消息"""
        return await self._inbound_queue.get()
    
    # ─── 出站 API ───
    
    async def publish_outbound(self, message: OutboundMessage):
        """Agent 将回复发布到出站队列"""
        await self._outbound_queue.put(message)
    
    async def consume_outbound(self) -> OutboundMessage:
        """Channel 从出站队列消费并发送"""
        return await self._outbound_queue.get()
```

**为什么用队列而非直接调用**：

| 对比维度 | 直接调用 | 队列解耦 |
|---------|---------|---------|
| 耦合度 | Agent 直接依赖 Channel | 完全解耦 |
| 并发 | 需要手动管理 | 队列天然支持 |
| 背压 | 无 | 队列满时自动等待 |
| 扩展性 | 添加平台需修改 Agent | 只需添加 Channel |
| 测试 | 需要 Mock 平台 | 直接操作队列 |

### 3.3 10 个设计模式

Nanobot 的源码中应用了 10 个经典设计模式：

#### 3.3.1 模式 1: 生产者-消费者 (Producer-Consumer)

```
应用位置: MessageBus 双队列

生产者:
  · Channel 生产 InboundMessage
  · AgentRunner 生产 OutboundMessage

消费者:
  · AgentLoop 消费 InboundMessage
  · ChannelManager 消费 OutboundMessage

优势:
  ✅ 解耦生产者和消费者
  ✅ 天然支持异步和背压
  ✅ 可独立扩展生产/消费端
```

#### 3.3.2 模式 2: 注册表 (Registry)

```
应用位置: ToolRegistry

实现:
  class ToolRegistry:
      def register(name, definition, handler):
          self._tools[name] = ToolDefinition(...)
      
      def execute(name, arguments):
          return self._tools[name].handler(**arguments)

优势:
  ✅ 工具即插即用
  ✅ 统一执行入口
  ✅ 支持动态注册 (MCP 工具)
```

#### 3.3.3 模式 3: 适配器 (Adapter)

```
应用位置: BaseChannel 及其子类

实现:
  class BaseChannel(ABC):
      @abstractmethod
      async def send(chat_id, content): pass
      
      @abstractmethod
      def _convert_to_inbound(platform_message): pass
  
  class TelegramChannel(BaseChannel):
      async def send(chat_id, content):
          await self.bot.send_message(chat_id, content)
      
      def _convert_to_inbound(update):
          return InboundMessage(...)

优势:
  ✅ 统一多平台接口
  ✅ 添加新平台无需修改核心代码
  ✅ 符合开闭原则
```

#### 3.3.4 模式 4: 包装器 (Wrapper)

```
应用位置: MCPToolWrapper

实现:
  class MCPToolWrapper(BaseTool):
      def __init__(self, server_name, tool_spec, client):
          self.name = f"mcp_{server_name}_{tool_spec.name}"
          self.parameters_schema = self._normalize_schema(tool_spec.input_schema)
      
      async def run(self, args):
          result = await self.client.call_tool(name=self.original_name, arguments=args)
          return self._format_result(result)

优势:
  ✅ MCP 工具与内置工具统一格式
  ✅ Agent 无需区分工具来源
  ✅ 标准化集成流程
```

#### 3.3.5 模式 5: 配置驱动 (Configuration-Driven)

```
应用位置: config.json + AGENTS.md + SOUL.md

实现:
  # 改配置而非改代码
  config.json → LLM Provider, 平台, 工具
  AGENTS.md → Agent 身份, 原则, 工作流
  SOUL.md → 个性, 风格, 边界

优势:
  ✅ 非程序员也能定制 Agent
  ✅ 快速迭代,无需重新部署
  ✅ 完全透明可审计
```

#### 3.3.6 模式 6: Workspace 中心 (Workspace-Centric)

```
应用位置: 所有文件操作围绕 workspace

实现:
  workspace/
  ├── memory/
  │   ├── MEMORY.md
  │   └── HISTORY.md
  ├── sessions/
  │   └── cli:default.jsonl
  ├── skills/
  │   └── github/SKILL.md
  ├── AGENTS.md
  └── config.json

优势:
  ✅ 所有数据集中管理
  ✅ 便于备份和迁移
  ✅ 沙箱隔离 (restrict_to_workspace)
```

#### 3.3.7 模式 7: 渐进披露 (Progressive Disclosure)

```
应用位置: Skill 系统

实现:
  Tier 1: 摘要目录 (始终注入)
    · 所有技能的 name + description
    · 消耗: 100-500 tokens
  
  Tier 2: SKILL.md 全文 (按需加载)
    · Agent 主动 read_file 读取
    · 消耗: 200-1000 tokens
  
  Tier 3: scripts/references (深度使用)
    · 访问辅助脚本和参考文档
    · 消耗: 按需

优势:
  ✅ 节省 token (默认只注入摘要)
  ✅ 按需深入,不浪费资源
  ✅ 支持无限扩展技能数量
```

#### 3.3.8 模式 8: 虚拟工具 (Virtual Tool)

```
应用位置: save_memory (记忆压缩)

实现:
  save_memory 是一个虚拟工具:
  · 不在工具注册表中暴露给 Agent 正常使用
  · 仅在记忆压缩时通过 tool_choice 强制 LLM 调用
  · 返回两个参数: history_entry + memory_update

优势:
  ✅ 控制 LLM 行为 (强制调用特定工具)
  ✅ 不污染正常工具列表
  ✅ 实现特殊逻辑 (记忆压缩)
```

#### 3.3.9 模式 9: 会话并发控制 (Session Concurrency Control)

```
应用位置: AgentLoop 的会话锁

实现:
  class AgentLoop:
      self._session_locks: dict[str, asyncio.Lock] = {}
      self._concurrency_gate = asyncio.Semaphore(3)
  
      async def _handle_message(self, msg):
          # 会话锁: 同一会话串行处理
          async with self._session_locks[msg.session_key]:
              # 并发门: 全局最多3个会话并行
              async with self._concurrency_gate:
                  await self._process_message(msg)

优势:
  ✅ 防止同一会话消息乱序
  ✅ 控制全局资源消耗
  ✅ 支持多用户并发
```

#### 3.3.10 模式 10: Prompt Cache (提示词缓存)

```
应用位置: System Prompt 构建

实现:
  System Prompt = 基础指令 + MEMORY.md + 技能摘要 + 工具定义
  
  优化:
  · 基础指令: 缓存 (不变化)
  · 工具定义: 缓存 (注册后不变)
  · MEMORY.md: 每轮加载 (动态)
  · 技能摘要: 缓存 (技能不变则不更新)

优势:
  ✅ 减少重复计算
  ✅ 降低 API 延迟
  ✅ 节省 token
```

---

## 4. 源码逐行解读

### 4.1 AgentLoop 核心循环

#### 4.1.1 初始化

```python
class AgentLoop:
    def __init__(self, bus, provider, workspace, tool_registry, 
                 memory_store, context_builder):
        """初始化 AgentLoop"""
        
        # 依赖注入
        self.bus = bus                    # MessageBus
        self.provider = provider          # LLM Provider (OpenAI/Anthropic)
        self.workspace = workspace        # Workspace 路径
        self.tool_registry = tool_registry # 工具注册表
        self.memory_store = memory_store  # 记忆存储
        self.context_builder = context_builder  # 上下文构建器
        
        # 并发控制
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._concurrency_gate = asyncio.Semaphore(3)
        
        # 运行状态
        self._running = False
        self._task: Optional[asyncio.Task] = None
```

**关键设计**：
- **依赖注入**：所有依赖通过构造函数传入,便于测试和替换
- **会话锁字典**：每个 session_key 对应一个独立的锁
- **并发门控**：`asyncio.Semaphore(3)` 限制最多 3 个会话并行处理

#### 4.1.2 主循环

```python
async def run(self):
    """启动主循环"""
    self._running = True
    
    while self._running:
        try:
            # 1. 阻塞等待入站消息
            msg = await self.bus.consume_inbound()
            
            # 2. 异步处理（不阻塞后续消息）
            asyncio.create_task(self._handle_message(msg))
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"AgentLoop error: {e}")
            await asyncio.sleep(1)  # 防止错误循环
```

**错误处理策略**：
- `CancelledError`：优雅退出循环
- 其他异常：记录日志,休眠 1 秒后继续 (防止错误风暴)

#### 4.1.3 消息处理

```python
async def _handle_message(self, msg: InboundMessage):
    """处理单条入站消息"""
    
    # 1. 获取会话锁（防止同一会话并发）
    session_key = msg.session_key
    if session_key not in self._session_locks:
        self._session_locks[session_key] = asyncio.Lock()
    
    # 2. 串行处理同一会话的消息
    async with self._session_locks[session_key]:
        
        # 3. 并发门控（限制全局并发数）
        async with self._concurrency_gate:
            
            # 4. 实际处理
            await self._process_message(msg)
```

**为什么需要会话锁**：

```
无锁场景（消息乱序）:
  t1: 用户发送消息A → AgentLoop 开始处理
  t2: 用户发送消息B → AgentLoop 开始处理（并行）
  t3: 消息B 先完成 → 回复B
  t4: 消息A 后完成 → 回复A
  结果：用户先收到B的回复,再收到A的回复（顺序错误！）

有锁场景（串行处理）:
  t1: 用户发送消息A → 获取锁,开始处理
  t2: 用户发送消息B → 等待锁
  t3: 消息A 完成 → 释放锁,回复A
  t4: 消息B 获取锁,开始处理
  t5: 消息B 完成 → 回复B
  结果：用户按顺序收到A和B的回复（正确！）
```

### 4.2 AgentRunner 执行器

#### 4.2.1 ReAct 循环实现

```python
class AgentRunner:
    async def run(self) -> str:
        """执行 ReAct 循环"""
        
        for iteration in range(self.max_iterations):  # 最多 40 次
            # ─── Step 1: 调用 LLM ───
            response = await self.provider.chat_with_retry(
                messages=self.messages,
                tools=self.tool_registry.get_definitions()
            )
            
            # ─── Step 2: 检查工具调用 ───
            if response.has_tool_calls:
                # ─── Step 3: 执行工具 ───
                tool_results = []
                for tool_call in response.tool_calls:
                    result = await self.tool_registry.execute(
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments)
                    )
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                
                # ─── Step 4: 将结果加入消息历史 ───
                self.messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": response.tool_calls
                })
                
                for tool_result in tool_results:
                    self.messages.append({
                        "role": "tool",
                        "content": tool_result["content"],
                        "tool_call_id": tool_result["tool_call_id"]
                    })
                
                # ─── Step 5: 检查记忆压缩 ───
                if self.memory_store.should_consolidate(self.messages):
                    await self.memory_store.consolidate(self.messages)
            
            else:
                # ─── Step 6: 返回最终回复 ───
                final_response = response.content
                await self.memory_store.save_session(
                    self.session_key, self.messages
                )
                return final_response
        
        return "已达到最大处理次数"
```

### 4.3 MemoryStore 记忆存储

#### 4.3.1 会话保存

```python
def _save_turn(self, message: dict):
    """保存一个对话回合到 JSONL"""
    
    # 1. 清理 reasoning/thinking 标签
    content = message.get("content", "")
    content = self._strip_thinking_tags(content)
    
    # 2. 处理大图片（用占位符替换）
    if "media" in message:
        message = self._replace_large_images(message)
    
    # 3. 追加写入 JSONL
    session_path = self._get_session_path()
    with open(session_path, "a") as f:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")

def _strip_thinking_tags(self, content: str) -> str:
    """移除 LLM 内部推理标签"""
    import re
    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
    content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
    return content.strip()
```

**为什么清理 thinking 标签**：
- 某些模型 (如 DeepSeek) 会在回复中包含 `<thinking>...</thinking>` 内部推理
- 这些标签是模型的内部思考过程,不应出现在持久化的会话记录中
- 清理后减少存储空间,避免混淆后续对话

### 4.4 MessageBus 消息总线

MessageBus 的实现非常简洁 (不到 50 行):

```python
class MessageBus:
    """异步消息总线"""
    
    def __init__(self):
        self._inbound_queue = asyncio.Queue()
        self._outbound_queue = asyncio.Queue()
    
    async def publish_inbound(self, message: InboundMessage):
        await self._inbound_queue.put(message)
    
    async def consume_inbound(self) -> InboundMessage:
        return await self._inbound_queue.get()
    
    async def publish_outbound(self, message: OutboundMessage):
        await self._outbound_queue.put(message)
    
    async def consume_outbound(self) -> OutboundMessage:
        return await self._outbound_queue.get()
```

**设计哲学**：
- **极简实现**：核心逻辑不到 50 行,但功能完整
- **asyncio.Queue**：Python 标准库,线程安全,支持异步
- **双队列**：入站和出站完全隔离,避免消息混乱

---

## 5. MCP 协议深度集成

### 5.1 MCP 协议概述

**MCP** (Model Context Protocol) 是 2024 年 AI 领域最重要的标准化协议之一。

```
MCP 的核心价值：
  解决 M×N 连接问题,实现工具/数据的即插即用

传统方式 (M×N 问题):
  M 个 AI 应用 × N 个数据源/工具 = M×N 个集成
  
  应用 A ──→ 数据源 1
  应用 A ──→ 数据源 2
  应用 A ──→ 数据源 3
  应用 B ──→ 数据源 1
  应用 B ──→ 数据源 2
  应用 B ──→ 数据源 3
  ... (指数增长!)

MCP 方式 (M+N 问题):
  M 个 AI 应用 + N 个 MCP Server = M+N 个连接
  
  应用 A ──┐
  应用 B ──┼──→ MCP Protocol ──→ Server 1
  应用 C ──┘                      Server 2
                                Server 3
  (标准化协议,一次集成,到处使用)
```

### 5.2 M×N 问题与三角架构

#### 5.2.1 MCP 的三角架构

```
┌────────────────────────────────────────────────────┐
│                  MCP 三角架构                        │
│                                                    │
│  ┌──────────┐         ┌──────────┐                │
│  │  Client   │◄───────►│  Host    │                │
│  │ (MCP SDK) │         │ (IDE/    │                │
│  │           │         │  Chat)   │                │
│  └─────┬────┘         └──────────┘                │
│        │                                          │
│        │ MCP Protocol                             │
│        │ (JSON-RPC over stdin/stdout)              │
│        │                                          │
│        ▼                                          │
│  ┌──────────┐                                     │
│  │  Server   │                                     │
│  │ (MCP      │                                     │
│  │  Server)  │                                     │
│  │           │                                     │
│  │ · Tools   │ ← 模型可调用                        │
│  │ · Resources│ ← 应用可读取                       │
│  │ · Prompts │ ← 用户可使用                       │
│  └──────────┘                                     │
└────────────────────────────────────────────────────┘
```

#### 5.2.2 通信协议

```
MCP 使用 JSON-RPC 2.0 协议,通过 stdin/stdout 通信:

Client 发送请求 (stdin):
  {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}

Server 返回响应 (stdout):
  {"jsonrpc": "2.0", "id": 1, "result": {"tools": [...]}}

优势:
  ✅ 语言无关 (任何语言都能实现)
  ✅ 进程隔离 (Server 崩溃不影响 Host)
  ✅ 标准协议 (JSON-RPC 2.0)
  ✅ 易于调试 (可打印 JSON)
```

### 5.3 MCP 三大原语

MCP 定义了三个核心原语：

| 原语 | 控制方 | 说明 | 示例 |
|------|--------|------|------|
| **Tools** | 模型控制 | 模型可调用的函数 | 查询天气、搜索文档、执行SQL |
| **Resources** | 应用控制 | 应用可读取的数据 | 文件内容、数据库记录、API响应 |
| **Prompts** | 用户控制 | 用户可使用的模板 | 代码审查模板、邮件模板 |

#### 5.3.1 Tools (工具)

```json
// MCP Server 暴露的工具列表
{
  "tools": [
    {
      "name": "get_weather",
      "description": "获取指定城市的天气信息",
      "inputSchema": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string",
            "description": "城市名称"
          }
        },
        "required": ["city"]
      }
    }
  ]
}

// 模型调用工具
{
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {"city": "Beijing"}
  }
}

// Server 返回结果
{
  "content": [
    {"type": "text", "text": "Beijing: Sunny, 25°C"}
  ]
}
```

#### 5.3.2 Resources (资源)

```json
// MCP Server 暴露的资源
{
  "resources": [
    {
      "uri": "file:///docs/api-reference.md",
      "name": "API Reference",
      "description": "API 参考文档",
      "mimeType": "text/markdown"
    }
  ]
}

// 应用读取资源
{
  "method": "resources/read",
  "params": {
    "uri": "file:///docs/api-reference.md"
  }
}

// Server 返回资源内容
{
  "contents": [
    {
      "uri": "file:///docs/api-reference.md",
      "mimeType": "text/markdown",
      "text": "# API Reference\n\n..."
    }
  ]
}
```

#### 5.3.3 Prompts (提示词)

```json
// MCP Server 提供的提示词模板
{
  "prompts": [
    {
      "name": "code_review",
      "description": "代码审查模板",
      "arguments": [
        {
          "name": "code",
          "description": "要审查的代码",
          "required": true
        }
      ]
    }
  ]
}

// 用户使用提示词
{
  "method": "prompts/get",
  "params": {
    "name": "code_review",
    "arguments": {"code": "def hello():..."}
  }
}

// Server 返回完整的提示词
{
  "messages": [
    {
      "role": "user",
      "content": "请审查以下代码:\n\ndef hello():..."
    }
  ]
}
```

### 5.4 MCPToolWrapper 实现

Nanobot 通过 `MCPToolWrapper` 将 MCP Server 的工具无缝集成到 ToolRegistry：

```python
class MCPToolWrapper(BaseTool):
    """MCP 工具包装器"""
    
    def __init__(self, server_name: str, tool_spec: dict, client: MCPClient):
        # 添加前缀避免命名冲突
        self.name = f"mcp_{server_name}_{tool_spec['name']}"
        self.original_name = tool_spec['name']
        self.description = tool_spec['description']
        self.parameters_schema = self._normalize_schema(tool_spec['inputSchema'])
        self.client = client
    
    def _normalize_schema(self, mcp_schema: dict) -> dict:
        """将 MCP Schema 转换为 OpenAI 兼容格式"""
        # MCP 和 OpenAI 的 Schema 格式略有不同,需要归一化
        normalized = {
            "type": "object",
            "properties": {},
            "required": mcp_schema.get("required", [])
        }
        
        for prop_name, prop_spec in mcp_schema.get("properties", {}).items():
            normalized["properties"][prop_name] = {
                "type": prop_spec.get("type", "string"),
                "description": prop_spec.get("description", "")
            }
        
        return normalized
    
    async def run(self, args: dict) -> str:
        """执行 MCP 工具调用"""
        try:
            # 调用 MCP Server
            result = await self.client.call_tool(
                name=self.original_name,
                arguments=args
            )
            
            # 格式化结果
            return self._format_result(result)
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _format_result(self, result: dict) -> str:
        """格式化 MCP 工具返回结果"""
        content_items = result.get("content", [])
        texts = []
        for item in content_items:
            if item.get("type") == "text":
                texts.append(item["text"])
        return "\n".join(texts)
```

**注册流程**：

```python
# 1. 连接到 MCP Server
client = MCPClient(command="python", args=["-m", "weather_server"])
await client.connect()

# 2. 获取工具列表
tools = await client.list_tools()

# 3. 包装并注册到 ToolRegistry
for tool_spec in tools:
    wrapper = MCPToolWrapper(
        server_name="weather",
        tool_spec=tool_spec,
        client=client
    )
    tool_registry.register(
        name=wrapper.name,
        definition=wrapper.get_definition(),
        handler=wrapper.run
    )

# 4. Agent 可以像使用内置工具一样使用 MCP 工具
#    例如: mcp_weather_get_weather(city="Beijing")
```

**面试话术**：
> "MCP 是 2024-2026 年 AI 领域最重要的标准化协议之一。它解决了 M×N 集成问题——传统方式需要为每个 AI 应用和每个数据源/工具编写集成代码,复杂度是 M×N。MCP 通过标准化协议,将复杂度降到 M+N。
>
> Nanobot 原生支持 MCP,通过 MCPToolWrapper 将 MCP Server 的工具转换为内置工具格式,注册到 ToolRegistry。这样 Agent 无需区分工具来源,使用方式完全一致。这是真正的'即插即用'——添加新工具只需启动一个 MCP Server,无需修改框架代码。"

---

## 6. 安装与上手实战

### 6.1 环境配置

```bash
# 1. 安装 Python 3.11+
python3 --version  # 确保 >= 3.11

# 2. 安装 Nanobot
pip install nanobot-ai

# 3. 创建 workspace
mkdir my-first-bot && cd my-first-bot

# 4. 初始化配置
nanobot onboard
# 交互式引导:
# - 选择 LLM Provider (OpenAI/Anthropic/DeepSeek)
# - 输入 API Key
# - 设置模型名称
# - 配置可选的渠道 (Telegram/Discord等)

# 5. 启动 Agent
nanobot
# 进入 CLI 交互模式
```

### 6.2 第一次对话

```
$ nanobot

Nanobot v0.2.1 - AI Agent Framework
Workspace: /path/to/my-first-bot

You: 你好,请介绍一下你自己

Agent: 你好！我是你的 AI 助手,基于 Nanobot v0.2.1 框架运行。
我可以帮助你:
- 回答问题和提供信息
- 执行文件操作 (读写文件)
- 运行 Shell 命令
- 搜索网页
- 图像生成与分析
- 以及更多通过技能和 MCP 扩展的能力

你也可以访问 WebUI: http://localhost:8080 获得更好的交互体验。

有什么我可以帮你的吗?

You: 帮我创建一个 hello.py 文件,打印 "Hello, World!"

Agent: [调用工具: write_file]
[路径: hello.py]
[内容: print("Hello, World!")]

文件已创建成功！你可以运行 `python hello.py` 来测试。

You: 运行它

Agent: [调用工具: exec]
[命令: python hello.py]

输出:
Hello, World!

程序成功运行！
```

### 6.3 Bot 定制

#### 6.3.1 修改 AGENTS.md

```markdown
# 我的个人助手

## 身份
你是一个友好的个人助手,专门帮助我管理日常任务和项目。

## 核心原则
1. 简洁高效：回答简洁明了,不啰嗦
2. 主动建议：不仅回答问题,还主动给出建议
3. 记住偏好：记住我的工作习惯和偏好

## 我的偏好
- 编程语言：Python, JavaScript
- 编辑器：VS Code
- 操作系统：macOS
- 工作风格：喜欢代码示例,偏好详细解释

## 可用技能
- `skills/github/SKILL.md` - GitHub 管理
- `skills/weather/SKILL.md` - 天气查询
```

#### 6.3.2 添加自定义技能

```bash
# 创建技能目录
mkdir -p skills/my-skill

# 编写 SKILL.md
cat > skills/my-skill/SKILL.md << 'EOF'
---
name: my-skill
description: "我的自定义技能"
always: false
---

# 我的自定义技能

这个技能教会 Agent 如何做 XXX。

## 使用步骤
1. ...
2. ...
3. ...
EOF

# 重启 Agent
nanobot
# Agent 会自动发现并加载新技能
```

---

## 7. 记忆系统设计实战

### 7.1 为什么 Agent 需要记忆

#### 7.1.1 无记忆 Agent 的问题

```
对话 1:
You: 我叫张三,是一名后端工程师
Agent: 你好张三！很高兴认识你。

对话 2（新会话）:
You: 我之前说我叫什么名字？
Agent: 抱歉,我不知道你的名字,这是我们的第一次对话。
```

**这就是纯 LLM 的局限**——没有跨会话记忆,每次对话都是全新的开始。

#### 7.1.2 记忆赋予的能力

| 能力 | 无记忆 | 有记忆 |
|------|--------|--------|
| 跨会话延续 | 每次重新开始 | 记住之前的对话 |
| 用户偏好 | 每次重新询问 | 自动适应 |
| 任务延续 | 无法暂停继续 | 支持长期项目 |
| 知识积累 | 不积累 | 持续学习 |
| 上下文理解 | 仅当前对话 | 理解历史背景 |

### 7.2 双层记忆架构

Nanobot 采用极简但高效的双层记忆架构：

```
┌─────────────────────────────────────────────────────────┐
│                  Nanobot 记忆架构                        │
│                                                         │
│  ┌──────────────────────┐  ┌──────────────────────────┐ │
│  │   MEMORY.md          │  │   HISTORY.md             │ │
│  │   （长期记忆）         │  │   （历史时间线）          │ │
│  │                      │  │                          │ │
│  │ · 关键事实           │  │ · 时间戳 + 事件摘要      │ │
│  │ · 用户偏好           │  │ · 仅追加模式             │ │
│  │ · 重要决策           │  │ · 不注入 System Prompt   │ │
│  │ · 项目状态           │  │ · 通过 read_file 检索    │ │
│  │                      │  │                          │ │
│  │ ✅ 始终注入上下文     │  │ ✅ 节省 token           │ │
│  │ ❌ 大小有限          │  │ ❌ 需要主动查询          │ │
│  └──────────────────────┘  └──────────────────────────┘ │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │   Session JSONL（短期记忆）                       │  │
│  │   · 完整对话历史 · 工具调用记录 · 当前会话上下文  │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**关键设计决策**：

| 设计决策 | Nanobot 的选择 | 理由 |
|---------|---------------|------|
| 存储方式 | 纯文件 (Markdown) | 透明、可编辑、无外部依赖 |
| 长期记忆 | 全量注入 | 确保 Agent 始终知道关键信息 |
| 历史时间线 | 按需检索 | 节省 token,历史通常不需要 |
| 压缩方式 | LLM 驱动摘要 | 利用 LLM 的理解能力 |
| 会话存储 | JSONL 格式 | 追加写入高效,便于流式处理 |

---

## 15. 2026 年新特性详解

### 15.1 WebUI Workbench (v0.2.1)

2026 年 6 月发布的 v0.2.1 版本将 Nanobot 从纯 CLI 工具升级为完整的 Agent 工作台。

#### 15.1.1 WebUI 核心特性

```
┌─────────────────────────────────────────────────────────┐
│              Nanobot WebUI Workbench                     │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Chat Panel  │  │  Thought     │  │  File        │ │
│  │              │  │  Timeline    │  │  Activity    │ │
│  │ · 多会话切换  │  │ · 思考过程    │  │ · 实时编辑    │ │
│  │ · 消息历史    │  │ · 工具调用    │  │ · 变更追踪    │ │
│  │ · 文件上传    │  │ · 决策路径    │  │ · 预览      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Project Workspaces & Model Controls              │  │
│  │  · 项目工作区隔离                                 │  │
│  │  · 模型预设切换                                   │  │
│  │  · 上下文窗口控制                                 │  │
│  │  · 长期目标管理 (/goal)                           │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**关键优势**：
- ✅ **可视化 Thought 过程**：清晰展示 Agent 的思考链路和工具调用序列
- ✅ **实时文件编辑**：在 WebUI 中直接查看和编辑 Agent 操作的文件
- ✅ **项目工作区**：不同项目隔离，避免上下文混乱
- ✅ **模型控制**：动态切换模型、调整上下文窗口、设置温度等参数
- ✅ **i18n 支持**：多语言界面切换

#### 15.1.2 启动 WebUI

```bash
# 启动 Nanobot 并开启 WebUI
nanobot --webui

# 访问地址
# http://localhost:8080

# 配置端口
nanobot --webui --port 3000
```

### 15.2 Dream Memory 两阶段记忆系统 (v0.1.5)

Dream Memory 是 Nanobot 在 v0.1.5 引入的先进记忆系统，比传统的 MEMORY.md 更智能。

#### 15.2.1 Dream Memory 架构

```
┌─────────────────────────────────────────────────────────┐
│                  Dream Memory 架构                       │
│                                                         │
│  Stage 1: 短期记忆 (Visible History)                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │ · 当前会话完整对话                                │  │
│  │ · JSONL 格式存储                                  │  │
│  │ · 自动清理 reasoning 标签                         │  │
│  │ · 原子写入 + 自动修复                             │  │
│  └──────────────────────────────────────────────────┘  │
│                          │                              │
│                          │ 压缩触发                     │
│                          ▼                              │
│  Stage 2: 长期记忆 (Dream Memory)                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │ · LLM 驱动的自动摘要                              │  │
│  │ · 关键信息提取（事实、偏好、决策）                 │  │
│  │ · 记忆老化与淘汰                                  │  │
│  │ · 基于 token 阈值的智能压缩                       │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### 15.2.2 Dream Memory vs 传统记忆

| 特性 | 传统 MEMORY.md | Dream Memory |
|------|---------------|--------------|
| **更新方式** | 手动或简单追加 | LLM 智能摘要与更新 |
| **信息筛选** | 全部保留 | 自动提取关键信息 |
| **记忆老化** | 无 | 自动淘汰过时信息 |
| **压缩策略** | 固定阈值 | 基于 token 和内容重要性 |
| **可靠性** | 可能遗漏 | 原子写入 + 自动修复 |

### 15.3 Goal 系统 - 长期目标管理 (v0.2.0)

Goal 系统解决了 Agent 在长对话中容易丢失原始目标的问题。

#### 15.3.1 使用 Goal 命令

```
# 设置长期目标
User: /goal 帮我重构整个项目的认证模块，使用 JWT 替代 session

Agent: 已设置长期目标：
🎯 重构认证模块，使用 JWT 替代 session

我会在这个目标完成前持续跟踪进度。

# Goal 跨轮次保持
User: 先帮我查看当前的认证代码

Agent: [查看 auth.py]
[分析当前实现]
当前使用 session-based 认证，需要重构为 JWT。

🎯 当前目标：重构认证模块，使用 JWT 替代 session
进度：1/5 - 分析现有实现

User: 继续

Agent: [创建新的 auth_jwt.py]
[实现 JWT 逻辑]

🎯 当前目标：重构认证模块，使用 JWT 替代 session
进度：3/5 - 实现 JWT 认证
```

#### 15.3.2 Goal 系统架构

```python
# Goal 系统核心逻辑（简化版）
class GoalManager:
    def __init__(self):
        self.current_goal: Optional[str] = None
        self.progress_steps: list = []
    
    def set_goal(self, goal: str):
        """设置长期目标"""
        self.current_goal = goal
        self.progress_steps = []
        self.save_to_workspace()
    
    def track_progress(self, step: str, completed: bool):
        """跟踪进度"""
        self.progress_steps.append({
            "step": step,
            "completed": completed,
            "timestamp": datetime.now()
        })
    
    def should_remind(self) -> bool:
        """判断是否需要在回复中提醒目标"""
        return self.current_goal is not None
    
    def get_goal_context(self) -> str:
        """获取目标上下文，注入 System Prompt"""
        if not self.current_goal:
            return ""
        
        return f"""
当前长期目标：{self.current_goal}
已完成步骤：{len([s for s in self.progress_steps if s['completed']])}
总步骤数：{len(self.progress_steps)}
请持续推进这个目标。
"""
```

### 15.4 Plugin SDK - 可扩展架构 (v0.1.4+)

Plugin SDK 是 Nanobot 从"内置一切"到"可扩展平台"的关键转变。

#### 15.4.1 插件类型

```
Nanobot Plugin 类型:

1. Channel Plugins (渠道插件)
   · 添加新的聊天平台支持
   · 示例: pip install nanobot-wechat
   · 实现: BaseChannel 接口

2. Tool Plugins (工具插件)
   · 添加新工具能力
   · 示例: pip install nanobot-database
   · 实现: BaseTool 接口

3. Provider Plugins (模型提供商插件)
   · 添加新的 LLM 支持
   · 示例: pip install nanobot-local-llm
   · 实现: LLMProvider 接口
```

#### 15.4.2 开发 Channel 插件

```python
# nanobot_wechat/__init__.py
from nanobot.channels import BaseChannel
from nanobot.messages import InboundMessage, OutboundMessage

class WeChatChannel(BaseChannel):
    """微信渠道插件"""
    
    name = "wechat"
    
    def __init__(self, config: dict):
        self.token = config["token"]
        self.app_id = config["app_id"]
        # 初始化微信 SDK
    
    async def send(self, chat_id: str, content: str):
        """发送消息到微信"""
        await self.wechat_api.send_message(
            chat_id=chat_id,
            content=content
        )
    
    def _convert_to_inbound(self, wechat_msg) -> InboundMessage:
        """将微信消息转换为 Nanobot 格式"""
        return InboundMessage(
            channel="wechat",
            chat_id=wechat_msg.from_user,
            content=wechat_msg.text,
            session_key=f"wechat:{wechat_msg.from_user}"
        )
    
    async def start(self):
        """启动微信消息监听"""
        # 设置 webhook 或轮询
        pass

# 注册插件
def register(registry):
    registry.register_channel("wechat", WeChatChannel)
```

#### 15.4.3 安装和使用插件

```bash
# 安装微信插件
pip install nanobot-wechat

# 在 config.json 中配置
{
  "channels": {
    "wechat": {
      "token": "your_wechat_token",
      "app_id": "your_app_id"
    }
  }
}

# 启动 Nanobot（自动加载插件）
nanobot
```

### 15.5 Langfuse 可观测性集成 (v0.1.5)

生产环境中，追踪 Agent 的行为和性能至关重要。

#### 15.5.1 配置 Langfuse

```json
{
  "langfuse": {
    "enabled": true,
    "public_key": "pk-lf-...",
    "secret_key": "sk-lf-...",
    "host": "https://cloud.langfuse.com"
  }
}
```

#### 15.5.2 Langfuse 追踪内容

```
Langfuse Dashboard 显示:

1. Trace (追踪)
   · 每次用户请求的完整链路
   · 耗时、token 消耗、模型调用

2. Generations (生成)
   · LLM 调用的输入输出
   · Prompt 模板和参数

3. Spans (跨度)
   · 工具调用详情
   · 记忆加载/保存
   · 消息处理各阶段

4. 指标分析
   · 平均响应时间
   · Token 使用统计
   · 工具调用频率
   · 错误率
```

### 15.6 更新的 Providers 列表 (2026-06)

截至 v0.2.1，Nanobot 支持以下 LLM Providers：

| Provider | 支持模型 | 特色功能 |
|----------|---------|---------|
| **OpenAI** | GPT-4o, o1, o3, GPT-5 | 工具调用、图像生成、Codex |
| **Anthropic** | Claude 3/4 | Adaptive Thinking、Prompt Caching |
| **DeepSeek** | DeepSeek-V3/V4 | 思考模式控制、高性价比 |
| **Google** | Gemini 1.5/2.0 | 图像生成、长上下文 |
| **Qwen** | Qwen-Max/Plus | 中文优化、通义千问 |
| **Moonshot** | Kimi K1/K2.6 | 长文本处理 |
| **MiniMax** | MiniMax-01 | 图像生成、多模态 |
| **Zhipu** | GLM-4 | 图像生成、中文场景 |
| **StepFun** | step-1/2 | 思考模式 |
| **Xiaomi** | MiMo | Chain-of-Thought |
| **LongCat** | LongCat-Chat | 长上下文 |
| **NVIDIA NIM** | 多种模型 | 本地部署 |
| **AWS Bedrock** | Claude/Llama | 企业级 |
| **Azure OpenAI** | GPT 系列 | 企业合规 |
| **Ollama** | 本地模型 | 离线运行 |
| **LM Studio** | 本地模型 | GUI 管理 |
| **VolcEngine** | 豆包 | 字节跳动 |
| **OpenRouter** | 多模型聚合 | 模型路由 |
| **GitHub Copilot** | GPT-4/Claude | IDE 集成 |
| **HuggingFace** | 开源模型 | 模型库 |
| **Novita** | 多模型 | 性价比 |
| **Atomic Chat** | 多模型 | 隐私优先 |
| **Ant Ling** |  ling-1 | 蚂蚁集团 |
| **Skywork** | Skywork-13B | 昆仑万维 |
| **vLLM** | 自定义 | 本地推理 |
| **Local** | 自定义 | OpenAI 兼容 API |

### 15.7 更新的 Channels 列表 (2026-06)

截至 v0.2.1，Nanobot 支持以下聊天平台：

| Channel | 特性 | 状态 |
|---------|------|------|
| **CLI** | 终端交互、WebSocket | ✅ 稳定 |
| **WebUI** | 浏览器工作台、多会话 | ✅ v0.2.1 |
| **Telegram** | Bot API、Webhook、流式 | ✅ 稳定 |
| **Discord** | Bot、Threads、长消息分割 | ✅ 稳定 |
| **Feishu** | 飞书、CardKit 流式、话题线程 | ✅ 稳定 |
| **DingTalk** | 钉钉、富媒体 | ✅ 稳定 |
| **Slack** | Threads、Reactions | ✅ 稳定 |
| **MSTeams** | Teams 集成 | ✅ 稳定 |
| **WhatsApp** | 媒体支持、去重 | ✅ 稳定 |
| **Signal** | 隐私通信 | ✅ v0.1.5+ |
| **Email** | SMTP/IMAP、附件 | ✅ 稳定 |
| **QQ** | QQ 群/私聊 | ✅ 稳定 |
| **WeCom** | 企业微信 | ✅ 稳定 |
| **WeChat** | 微信（插件） | 🔌 插件 |
| **Matrix** | 开放协议 | ✅ 稳定 |
| **WebSocket** | 自定义客户端 | ✅ 稳定 |

### 15.8 最佳实践更新 (2026)

#### 15.8.1 使用 Goal 系统管理长期任务

```markdown
✅ 推荐做法：
1. 开始复杂任务前先设置 /goal
2. 分步骤推进，Agent 会自动跟踪进度
3. 使用 /status 查看当前目标进度
4. 完成后使用 /goal clear 清除目标

❌ 避免做法：
1. 不使用 goal 直接开始复杂任务（容易丢失上下文）
2. 同时设置多个目标（会混淆 Agent）
3. 目标描述模糊（应具体明确）
```

#### 15.8.2 WebUI 与 CLI 的选择

```
使用 CLI 的场景：
· 快速测试和调试
· 服务器环境（无 GUI）
· 开发者日常使用
· 脚本集成

使用 WebUI 的场景：
· 非技术用户使用
· 需要可视化文件编辑
· 多会话管理
· 项目协作
· 查看 Thought 时间线
```

#### 15.8.3 生产部署检查清单

```bash
# 生产部署前检查清单

□ 1. 设置 restrict_to_workspace: true（沙箱隔离）
□ 2. 配置 Langfuse 可观测性
□ 3. 启用自动备份（自动保存 MEMORY.md）
□ 4. 设置合理的 max_tool_iterations（主Agent 30-40）
□ 5. 配置 Model Fallback（主模型 + 备用模型）
□ 6. 使用 Docker 部署（隔离和可重复性）
□ 7. 配置日志轮转和监控
□ 8. 定期审查 MEMORY.md 和 HISTORY.md
□ 9. 设置访问控制（Telegram allow-list 等）
□ 10. 测试紧急停止机制
```

### 15.9 Roadmap 与未来展望

根据官方 Roadmap（2026 年 5 月更新），Nanobot 的未来方向：

#### 15.9.1 短期目标 (2026 H2)

- [ ] **Plugin Marketplace**：插件市场，一键安装社区插件
- [ ] **Advanced Memory**：向量检索 + 知识图谱混合记忆
- [ ] **Multi-Agent Mode**：原生支持多 Agent 协作
- [ ] **Better Coding**：更强的代码理解和生成能力
- [ ] **Mobile Apps**：iOS/Android 客户端

#### 15.9.2 中期愿景

```
Nanobot 的终极愿景：开源个人 Agent 伴侣

成为一个：
· 轻量级（核心 < 10000 行）
· 本地优先（数据完全自主）
· 可扩展（Plugin SDK 生态）
· 跨平台（所有聊天应用 + Web + 终端）
· 有记忆（长期学习和适应）
· 可信赖（透明、可审计、安全）

的 AI Agent 伴侣，长期陪伴用户的工作和生活。
```

#### 15.9.3 与竞品对比的定位

```
Nanobot vs 其他框架的定位差异：

Nanobot:
  定位：个人 Agent 伴侣
  优势：轻量、透明、易定制、零依赖
  场景：个人助手、开发者工具、中小团队

LangChain:
  定位：企业级 Agent 平台
  优势：功能全面、生态丰富
  场景：企业应用、复杂工作流、生产级系统

AutoGen:
  定位：多 Agent 协作框架
  优势：多 Agent 对话、微软生态
  场景：团队协作、复杂任务分解

OpenAI Agents:
  定位：OpenAI 生态 SDK
  优势：官方支持、简单易用
  场景：OpenAI 模型应用开发

结论：Nanobot 在"轻量级个人助手"领域有独特优势，
      与其他框架是互补而非竞争关系。
```

### 15.10 学习资源更新

#### 15.10.1 官方资源

- **GitHub**: https://github.com/HKUDS/nanobot (44K+ Stars)
- **文档**: https://nanobot.readthedocs.io/
- **Roadmap 讨论**: https://github.com/HKUDS/nanobot/discussions/431
- **Release Notes**: https://github.com/HKUDS/nanobot/releases

#### 15.10.2 社区资源

- **Learn-Nanobot**: https://github.com/bcefghj/learn-nanobot (中文教程)
- **OpenClaw Index**: https://openclawindex.com/tools/hkudsnanobot
- **知乎专栏**: 搜索"Nanobot AI Agent"
- **掘金社区**: 多个 Nanobot 实战教程

#### 15.10.3 推荐学习路径

```
入门路径（1-2 周）:
1. 安装 Nanobot 并完成 onboard
2. 阅读 AGENTS.md 和 SOUL.md 理解配置驱动
3. 尝试 CLI 对话和工具调用
4. 添加自定义技能

进阶路径（2-4 周）:
5. 阅读源码（AgentLoop、AgentRunner）
6. 配置多平台接入（Telegram/Discord）
7. 集成 MCP Server
8. 使用 WebUI 工作台

生产路径（1-2 月）:
9. Docker 部署
10. 配置 Langfuse 监控
11. 开发自定义插件
12. 优化记忆系统和 Goal 管理
```

---

> 📝 **文档说明**：本文档基于 Nanobot v0.2.1 (2026-06-01) 更新，反映了最新的 WebUI Workbench、Goal 系统、Dream Memory 等特性。如需了解早期版本，请参考文档历史版本。
>
> 🔗 **项目地址**: https://github.com/HKUDS/nanobot
>
> ⭐ **如果这个项目对你有帮助，记得去 GitHub 点个 Star！**
