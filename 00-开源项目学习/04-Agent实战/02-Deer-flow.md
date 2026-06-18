# DeerFlow 2.0 - Super Agent Harness 学习指南

## 项目概述

**一句话总结**: DeerFlow 2.0 是字节跳动开源的 super agent harness，把 sub-agents、memory、sandbox 和 skills 组织在一起，让 agent 能完成几乎任何事情。

**核心亮点**:
- **Super Agent Harness**: 不是 framework，而是开箱即用的运行时基础设施，默认带 agent 真正会用的关键能力
- **Skills 系统**: 结构化能力模块（SKILL.md），按需渐进加载，覆盖研究、报告、演示文稿、网页生成等场景
- **Sub-Agents 并行**: lead agent 可动态拉起 sub-agents，各自独立上下文并行执行，最后汇总结果
- **Sandbox 执行环境**: 每个任务运行在隔离的 Docker 容器里，有完整文件系统，可执行 bash 命令和代码
- **长期记忆**: 跨 session 积累用户偏好、知识背景、工作习惯，越用越了解你
- **IM 渠道集成**: 支持 Telegram、Slack、飞书、企业微信、钉钉，无需公网 IP

**适合谁学**:
- 想构建生产级 deep research agent 的开发者
- 需要可扩展的 super agent harness 的工程团队
- 对 skills 系统、sandbox 执行、多 agent 并行感兴趣的学习者

---

## 核心架构解析

### 从 Deep Research 到 Super Agent Harness

```
DeerFlow 1.x: Deep Research 框架
  ↓ (社区推动，彻底重写)
DeerFlow 2.0: Super Agent Harness

核心理念转变:
  - 不再是"需要你自己拼装的 framework"
  - 而是"开箱即用又足够可扩展的运行时基础设施"
  - 基于 LangGraph + LangChain 构建
  - 默认带 filesystem、memory、skills、sandbox、sub-agents
```

### 架构全景

```
DeerFlow 2.0 = Lead Agent（主智能体）
              + Sub-Agents（子智能体，按需并行拉起）
              + Skills（技能模块，按需加载）
              + Tools（核心工具 + MCP Server 扩展）
              + Sandbox（隔离执行环境，Docker 容器）
              + Memory（长期记忆，跨 session 持久化）
              + Context Engineering（上下文管理，摘要压缩）
              + IM Channels（Telegram/Slack/飞书/企微/钉钉）
```

---

## 代码逻辑主线

### 1. Skills 与 Tools 系统

#### Skills：结构化能力模块

```markdown
# SKILL.md 示例
---
name: research
version: 1.0
author: DeerFlow Team
---

## 工作流程
1. 明确研究问题和范围
2. 使用 web_search 进行多角度搜索
3. 使用 web_scrape 抓取关键页面
4. 整理信息，形成结构化报告

## 最佳实践
- 使用多个搜索关键词交叉验证
- 优先引用权威来源（学术论文、官方文档）
- 对矛盾信息进行标注

## 参考资源
- Google Scholar: https://scholar.google.com
- arXiv: https://arxiv.org
```

**为什么这样设计**:
- **按需加载**: 不会一次性把所有 skills 塞进上下文，只有任务需要时才加载
- **渐进式**: 先加载 skill 概要，任务深入时再加载详细信息
- **可替换**: 可以替换内置 skills，或添加自定义 skills
- **可组合**: 多个 skills 可组合成复合工作流

**Skills 目录结构**:
```
/mnt/skills/
├── public/               # 内置 skills
│   ├── research/SKILL.md
│   ├── report-generation/SKILL.md
│   ├── slide-creation/SKILL.md
│   ├── web-page/SKILL.md
│   └── image-generation/SKILL.md
└── custom/               # 自定义 skills
    └── your-skill/SKILL.md
```

#### Tools：核心工具集

```python
# DeerFlow 核心工具
tools = [
    web_search,        # 网页搜索（Tavily API）
    web_scrape,        # 网页抓取
    file_read,         # 文件读取
    file_write,        # 文件写入
    bash_execute,      # Bash 命令执行
    # ... 可通过 MCP Server 扩展
]
```

**扩展方式**:
- **MCP Server**: 接入外部工具服务器
- **Python 函数**: 直接注册自定义工具函数
- **替换内置**: 可替换任何内置工具

---

### 2. Sub-Agents 并行执行

```python
# Sub-Agents 工作流程
def lead_agent_task(task):
    """
    Lead Agent 工作流程
    
    1. 理解任务，制定计划
    2. 拆解为子任务
    3. 动态拉起 sub-agents（并行执行）
    4. 等待所有 sub-agents 完成
    5. 汇总结果，生成最终输出
    """
    # 制定计划
    plan = create_plan(task)
    
    # 拉起 sub-agents（并行）
    sub_agents = []
    for subtask in plan.subtasks:
        sub_agent = spawn_sub_agent(
            task=subtask,
            context=isolated_context(),  # 独立上下文
            tools=available_tools(),
            termination_condition=subtask.condition
        )
        sub_agents.append(sub_agent)
    
    # 并行执行
    results = run_parallel(sub_agents)
    
    # 汇总结果
    final_output = synthesize_results(results)
    
    return final_output
```

**关键设计**:
- **独立上下文**: 每个 sub-agent 看不到主 agent 或其他 sub-agents 的上下文
- **并行执行**: 只要条件允许，sub-agents 同时运行
- **结构化结果**: 每个 sub-agent 返回结构化输出
- **聚焦任务**: sub-agent 只聚焦当前任务，不被无关信息干扰

**为什么重要**: 这是 DeerFlow 能处理从几分钟到几小时任务的原因。一个研究任务可拆成十几个 sub-agents，分别探索不同方向，最后合并成报告、网站或演示文稿。

---

### 3. Sandbox 与文件系统

```text
# Sandbox 容器内路径
/mnt/user-data/
├── uploads/          # 用户上传的文件
├── workspace/        # agents 的工作目录
└── outputs/          # 最终交付物

/mnt/skills/
├── public/           # 内置 skills
└── custom/           # 自定义 skills
```

**为什么这样设计**:
- **隔离执行**: 每个任务运行在独立的 Docker 容器中
- **完整文件系统**: agent 可读写、编辑文件，执行 bash 命令
- **可审计**: 所有操作都有日志，可追溯
- **不污染**: 不会在不同 session 之间互相污染

**Sandbox 模式**:
| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **Local** | 直接在宿主机执行 | 开发调试 |
| **Docker** | 在隔离容器内执行 | 生产环境 |
| **Kubernetes** | 通过 provisioner 在 K8s Pod 执行 | 大规模部署 |

---

### 4. Context Engineering

#### 隔离的 Sub-Agent Context

```python
# 每个 sub-agent 独立上下文
class SubAgentContext:
    def __init__(self, task, tools, memory=None):
        self.task = task  # 当前任务
        self.tools = tools  # 可用工具
        self.memory = memory or []  # 工作记忆
        self.parent_context = None  # 看不到父上下文
        self.sibling_contexts = []  # 看不到兄弟上下文
    
    def add_observation(self, observation):
        """添加观察结果到记忆"""
        self.memory.append(observation)
        
        # 如果记忆过长，进行压缩
        if len(self.memory) > threshold:
            self.compress()
    
    def compress(self):
        """摘要压缩"""
        # 保留最近 N 条
        recent = self.memory[-10:]
        
        # 压缩早期记忆
        summary = summarize(self.memory[:-10])
        
        self.memory = [summary] + recent
```

#### 主 Agent 上下文管理

```python
# 主 Agent 积极管理上下文
class LeadAgentContext:
    def manage_context(self):
        """
        上下文管理策略
        
        1. 总结已完成的子任务
        2. 将中间结果转存到文件系统
        3. 压缩暂时不重要的信息
        4. 保持聚焦，防止上下文爆炸
        """
        # 总结子任务
        summaries = []
        for completed_task in self.completed_tasks:
            summary = summarize_task(completed_task)
            summaries.append(summary)
        
        # 转存到文件系统
        save_to_filesystem("intermediate_results", summaries)
        
        # 压缩上下文
        self.context = compress_context(self.context)
```

**为什么重要**: 在长链路、多步骤任务中，积极的上下文管理能让 agent 保持聚焦，不会轻易把上下文窗口打爆。

---

### 5. 长期记忆

```python
# 长期记忆系统
class LongTermMemory:
    def __init__(self, user_id):
        self.user_id = user_id
        self.storage = LocalStorage(f"memory/{user_id}")
    
    def accumulate(self, session_data):
        """
        跨 session 积累记忆
        
        包括:
        - 个人偏好（写作风格、技术栈）
        - 知识背景（专业领域、熟悉程度）
        - 工作习惯（常用工具、工作流）
        """
        # 提取关键信息
        preferences = extract_preferences(session_data)
        knowledge = extract_knowledge(session_data)
        habits = extract_habits(session_data)
        
        # 更新记忆
        self.storage.update({
            "preferences": preferences,
            "knowledge": knowledge,
            "habits": habits
        })
    
    def retrieve(self, task):
        """检索相关记忆"""
        # 基于任务类型检索
        relevant = self.storage.search(task)
        
        # 返回记忆（让 agent 更了解用户）
        return relevant
```

**为什么重要**: 大多数 agents 在对话结束后会忘记一切，DeerFlow 会逐步积累关于你的持久记忆。你用得越多，它越了解你的写作风格、技术栈和重复出现的工作流。

---

## 快速上手实践

### 环境配置步骤

#### 方式一：Docker（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/bytedance/deer-flow.git
cd deer-flow

# 2. 生成配置文件
make config

# 3. 配置模型（编辑 config.yaml）
# 至少定义一个模型:
# models:
#   - name: gpt-4
#     use: langchain_openai:ChatOpenAI
#     model: gpt-4
#     api_key: $OPENAI_API_KEY

# 4. 配置 API Key（编辑 .env）
# OPENAI_API_KEY=your-api-key
# TAVILY_API_KEY=your-tavily-key

# 5. 启动服务
make docker-init    # 首次运行，拉取 sandbox 镜像
make docker-start   # 启动服务

# 6. 访问: http://localhost:2026
```

#### 方式二：本地开发

```bash
# 1. 检查依赖
make check  # 需要 Node.js 22+、pnpm、uv、nginx

# 2. 安装依赖
make install

# 3. 启动服务
make dev

# 4. 访问: http://localhost:2026
```

### 资源规划建议

| 部署场景 | 起步配置 | 推荐配置 |
|---------|---------|---------|
| 本地体验 | 4 vCPU, 8 GB | 8 vCPU, 16 GB |
| Docker 开发 | 4 vCPU, 8 GB | 8 vCPU, 16 GB |
| 长期运行 | 8 vCPU, 16 GB | 16 vCPU, 32 GB |

### 配置进阶功能

#### 配置 MCP Server

```yaml
# config.yaml
mcp_servers:
  - name: my-mcp-server
    url: http://localhost:8080/mcp
    # 支持 OAuth token 流程
```

#### 配置 IM 渠道

```yaml
# config.yaml
channels:
  telegram:
    enabled: true
    bot_token: $TELEGRAM_BOT_TOKEN
  
  feishu:
    enabled: true
    app_id: $FEISHU_APP_ID
    app_secret: $FEISHU_APP_SECRET
```

#### 配置 LangSmith 追踪

```bash
# .env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=lsv2_pt_xxx
LANGSMITH_PROJECT=deerflow
```

---

## 核心知识点总结

### 必须掌握的 8 个概念

1. **Super Agent Harness**: 开箱即用的运行时基础设施，不是 framework
2. **Skills 系统**: 结构化能力模块（SKILL.md），按需渐进加载
3. **Sub-Agents 并行**: lead agent 动态拉起 sub-agents，独立上下文并行执行
4. **Sandbox 执行**: 隔离的 Docker 容器，完整文件系统，可执行 bash
5. **Context Engineering**: 隔离子 agent 上下文，摘要压缩，防止上下文爆炸
6. **长期记忆**: 跨 session 积累用户偏好、知识、习惯
7. **IM 渠道集成**: Telegram/Slack/飞书/企微/钉钉，无需公网 IP
8. **模型路由**: 支持任何 OpenAI 兼容 API 的 LLM，无强绑定

---

## 常见疑问解答

### Q1: DeerFlow 和 LangGraph 有什么区别？

**关键区别**:
- **LangGraph**: 底层编排框架，提供状态机原语，需要你自己构建 agent
- **DeerFlow**: 基于 LangGraph 构建的**完整 agent 产品**，默认带 skills、memory、sandbox、sub-agents
- **类比**: LangGraph 是乐高积木，DeerFlow 是用乐高搭好的城堡

### Q2: 为什么用 Skills 而不是直接写工具？

**Skills 的优势**:
- **结构化**: SKILL.md 包含工作流、最佳实践、参考资源
- **按需加载**: 不会污染上下文，只在需要时加载
- **易维护**: 修改 skill 不需要改代码
- **可共享**: 可通过 Gateway 安装 `.skill` 压缩包

### Q3: Sandbox 安全吗？

**安全机制**:
- **隔离执行**: 每个任务在独立 Docker 容器中
- **文件系统隔离**: 不会访问宿主机的敏感文件
- **可审计**: 所有操作都有日志
- **注意**: 部署到公网需要额外安全措施（IP 白名单、身份验证、网络隔离）

### Q4: 可以用本地模型吗？

**可以**。只要模型实现了 OpenAI 兼容 API，都可以接入：

```yaml
# config.yaml - 使用 Ollama 本地模型
models:
  - name: ollama-llama3
    use: langchain_openai:ChatOpenAI
    model: llama3
    base_url: http://localhost:11434/v1
    api_key: ollama  # Ollama 不需要 API key，但字段必须有
```

---

## 延伸学习路径

1. **官方文档**: 
   - [配置指南](backend/docs/CONFIGURATION.md)
   - [架构概览](backend/CLAUDE.md)
   - [后端架构](backend/README.md)
   - [MCP Server 指南](backend/docs/MCP_SERVER.md)

2. **Claude Code 集成**:
   ```bash
   npx skills add https://github.com/bytedance/deer-flow --skill claude-to-deerflow
   ```

3. **进阶主题**:
   - Skills 开发：编写自己的 SKILL.md
   - MCP Server 开发：扩展自定义工具
   - 多 Agent 编排：理解 lead agent 调度逻辑
   - 生产部署：Docker Compose、Kubernetes

---

## 与教程目录的关联

本项目与以下教程内容强关联：

| 教程章节 | 关联点 |
|---------|-------|
| `06-单Agent开发与框架/01-Agent核心概念` | Super Agent Harness 理念 |
| `07-多Agent与Agent工程化/01-多Agent系统` | Sub-Agents 并行执行 |
| `07-多Agent与Agent工程化/03-MCP协议与Skills` | Skills 系统、MCP Server 集成 |
| `05-工具调用与RAG/01-RAG系统` | Deep Research 应用场景 |

---

## 总结

**DeerFlow 2.0 不是另一个 agent framework，而是 super agent harness**。它展示了：

1. ✅ **开箱即用**: 默认带 agent 真正会用的关键能力（filesystem、memory、skills、sandbox）
2. ✅ **足够可扩展**: 可替换任何组件，添加自定义 skills、tools、sub-agents
3. ✅ **生产就绪**: 隔离执行、上下文管理、长期记忆、IM 集成
4. ✅ **模型无关**: 支持任何 OpenAI 兼容 API 的 LLM
5. ✅ **社区驱动**: 从 Deep Research 到 Super Agent Harness，社区力量推动演进

**面试中可以讲的亮点**:
- 我深入学习了 DeerFlow 2.0，理解 super agent harness 的设计哲学
- 我能开发 Skills（SKILL.md），实现按需渐进加载的能力模块
- 我理解 sub-agents 并行执行、sandbox 隔离、上下文压缩等生产级机制
- 我能配置 IM 渠道（Telegram/飞书/钉钉），让 agent 接入即时通讯

---

**学习建议**: DeerFlow 是一个完整的产品，不要只看不跑！先用 Docker 快速体验，然后深入理解架构，最后尝试开发自己的 skills 和 tools。