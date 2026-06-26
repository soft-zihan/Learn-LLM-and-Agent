# AI Agents From Zero 学习指南

## 项目概述

**一句话总结**: AI Agents From Zero 是一套从零到企业级的 AI 智能体与大模型应用开发系统教程，包含 28 章完整课程、2 个实战项目（电商问数 + 深度研搜）和面试题库。

**核心亮点**:
- **系统完整**: 28 章系统教程，覆盖大模型基础、提示词工程、Coze/Dify 低代码平台、LangChain/LangGraph 框架、RAG、MCP、Agent、微调、企业级实战项目
- **可跑源码**: 每个案例都提供可运行的源码，环境说明和常见问题排查
- **实战项目**: 2 个企业级实战项目（电商问数 + 深度研搜），完整的主链路闭环
- **面试题库**: 按岗位能力域组织的面试题库，对齐培训班课表与 JD 高频考点
- **持续进化**: 伴随 AI 技术栈持续更新，面向 2026

**适合谁学**:
- 零基础想系统进入大模型应用开发的初学者
- 想从碎片化学习转向系统化学习的开发者
- 需要准备 AI 应用开发工程师面试的求职者
- 想掌握 Python + LangChain + LangGraph 技术栈的工程师

---

## 核心架构解析

### 教程知识体系

```
AI Agents From Zero 教程体系
├── 01 大模型基础能力构建
│   ├── 大模型（LLM）认识与环境准备
│   ├── 大模型架构原理（Transformer、MoE、自注意力）
│   ├── 大模型调度平台（Ollama、云端/本地部署）
│   └── 提示词工程（核心原则、链式思维、Few-shot、多轮对话）
│
├── 02 企业低代码平台开发与项目实战
│   ├── Coze（扣子）平台（工作流、Agent、知识库、插件）
│   ├── 项目：商户运营管家（行业调研 PPT、爆款视频复刻、营销海报）
│   ├── Dify AI 平台（工作流/Agent/知识库、多案例）
│   ├── 容器化技术（Docker 核心概念、安装与常用命令）
│   └── 企业级大模型部署（腾讯云/阿里云、AutoDL、Ollama、Xinference）
│
├── 03 大模型核心开发框架
│   ├── LangChain 框架（Model I/O、Prompt、Parser、LCEL、Memory、Tools、Retrieval、Agent）
│   ├── LangGraph 框架（图式思维、State/Node/Edge、持久化记忆、流式输出、多智能体与 A2A）
│   ├── MCP 从原理到实战（与 Function Calling 对比、通信机制、Server 部署）
│   └── 跨 Agent 通信：A2A 协议
│
├── 04 企业级 RAG / Agent 项目实战
│   ├── 掌柜智库（LangGraph RAG、MinerU/OCR、多路召回、RAGAS 评估）
│   ├── 电商小二（意图解析、多源知识库、流式回复、转人工机制）
│   ├── 电商问数（MySQL 数仓、元数据知识库、Qdrant、ES、LangGraph、SQL 生成）
│   └── 深度研搜（DeepAgents 多智能体、网络搜索、MySQL、RAGFlow、FastAPI、WebSocket）
│
├── 05 大模型微调实践
│   ├── 大模型微调核心（数据与格式、PEFT/LoRA/QLoRA、全参数微调、vLLM 部署）
│   ├── 企业级微调数据集构建
│   ├── 基于 Llama-Factory 的高效微调
│   └── 调优案例（学习率、LoRA 秩、loss/评测对比）
│
└── 06 大厂开发规范
    ├── 企业大模型研发流程（技术调研、方案设计、RAG 与 pipeline、评估）
    └── 大模型当下热点（Agent/RAG 主流技术、前沿跟踪）
```

### 技术栈概览

| 类别             | 技术/平台                                   | 说明                                                                             |
| ---------------- | ------------------------------------------- | -------------------------------------------------------------------------------- |
| **大模型与基础** | LLM、Transformer、MoE、自注意力             | LLaMA/Qwen/GPT、多模态、预训练/微调/推理                                         |
| **提示与编排**   | 提示词工程、Tool Calling、Skills            | 多轮对话、消息模板、结构化输出、工具调用、技能化能力沉淀                         |
| **低代码平台**   | Coze（扣子）、Dify                          | 工作流、Agent、知识库、插件、Python 调用与本地化部署                             |
| **开发框架**     | LangChain、LangGraph、DeepAgents            | Model I/O、Runnable / LCEL、Memory、Tools、Agents、图式工作流、多智能体          |
| **协议与通信**   | MCP（Model Context Protocol）、A2A          | Function Calling、服务解耦、外部工具接入、跨 Agent 协作                          |
| **RAG 与检索**   | 向量数据库、稀疏检索、混合检索、BGE-Rerank  | 多路召回、重排序、知识图谱、RAGAS 评估、高级 RAG 优化                            |
| **文档与多模态** | MinerU、OCR                                 | 图文混排 PDF 解析、设备手册与售后指南                                            |
| **部署与运维**   | Docker、Ollama、Xinference、vLLM            | 腾讯云/阿里云、AutoDL、Coze 本地部署                                             |
| **微调与训练**   | PEFT、LoRA、QLoRA、DeepSpeed、Llama-Factory | Alpaca/ShareGPT 数据格式、Safetensors/ONNX                                       |
| **编程与工具**   | Python；Codex、Cursor                       | 主语言为 Python；覆盖 AI 编程工具、Agent Skills、多模型 API、MCP 接入与调试      |
| **求职与面试**   | 面试题库                                    | 按岗位能力域组织 **问法 + 答法**；对齐同类**线上培训结业能力**与 **JD** 高频考点 |

---

## 代码逻辑主线

### 1. 从概念到实战的递进路线

```
教程设计哲学：
  概念篇（1-9 章） -> 框架篇（10-26 章） -> 微调篇（27-28 章） -> 实战项目篇
  
概念篇：
  - 大模型认知、提示词工程、RAG/微调/Agent 选型
  - 打好理论基础，理解"为什么这么设计"
  
框架篇：
  - LangChain 快速上手、Model I/O、Prompt、Parser、LCEL
  - Memory、Tools、Embedding、RAG、MCP、Agent
  - LangGraph 概述、API、节点、边、高级特性、多智能体
  
实战项目篇：
  - 电商问数：NL2SQL + LangGraph 完整链路
  - 深度研搜：DeepAgents 多智能体系统
```

### 2. 电商问数项目（NL2SQL + LangGraph）

```python
# 电商问数核心流程
def shopkeeper_query_natural_language(query):
    """
    自然语言问数完整链路
    
    1. 关键词抽取与多路召回
       - 从元数据知识库检索相关表/字段
       - Qdrant 向量检索
       - Elasticsearch 字段值检索
    
    2. 召回信息合并与上下文构建
       - 多路召回结果合并
       - 构建 SQL 生成上下文
    
    3. SQL 生成前的信息过滤与补全
       - 过滤不相关信息
       - 补全缺失的表/字段信息
    
    4. SQL 生成与执行闭环
       - LangGraph 工作流生成 SQL
       - SQL 校验与执行
       - 结果返回与格式化
    """
    pass
```

**核心技术点**:
- MySQL 数仓基础与元数据管理
- Qdrant 向量数据库与 Elasticsearch 检索
- LangGraph 工作流编排
- FastAPI SSE 接口与前后端联调

### 3. 深度研搜项目（DeepAgents 多智能体）

```python
# 深度研搜核心流程
def deep_search_research_task(task):
    """
    深度研究任务的多智能体调度
    
    1. 主智能体接收任务，分析需要哪些信息来源
    2. 调度子智能体并行执行：
       - 网络搜索子智能体（Tavily）
       - 数据库查询子智能体（MySQL）
       - RAGFlow 知识库子智能体
       - 上传文件分析工具
    3. 汇总多来源资料，判断信息是否足够
    4. 生成 Markdown 报告，可选转换 PDF
    5. 通过 WebSocket 实时推送执行进度到前端
    """
    pass
```

**核心技术点**:
- DeepAgents 框架：主智能体 + 子智能体分工
- 多来源工具接入（Tavily、MySQL、RAGFlow）
- 会话上下文隔离与文件系统管理
- FastAPI + WebSocket 实时通信
- React + Vite 前端闭环展示

---

## 快速上手实践

### 环境配置步骤

#### 1. 克隆项目并安装依赖

```bash
# 1. 克隆项目
git clone https://github.com/jackzhenguo/ai-agents-from-zero.git
cd ai-agents-from-zero

# 2. 准备环境（推荐 Python 3.10，支持 3.10-3.13）
python3.10 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows CMD

# 3. 安装依赖
pip install -r requirements.txt
```

#### 2. 配置 API Key

```bash
# 1. 复制环境配置文件
cp .env-example .env

# 2. 编辑 .env 文件，填入你的 API Key
# 例如：
# aliQwen-api=your-api-key
# QWEN_API_KEY=your-api-key
# deepseek-api=your-api-key
```

#### 3. 运行第一个案例

```bash
# 在项目根目录运行
python 案例与源码-2-LangChain框架/01-helloworld/StandardDesc.py
```

**注意**: 必须在**项目根目录**执行 `python`，否则会读不到 `.env`。

#### 4. 不想用云 API？使用本地 Ollama

```bash
# 1. 安装 Ollama
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# 2. 拉取模型
ollama pull llama3

# 3. 代码中使用本地模型（无需 API Key）
```

### 资源规划建议

| 使用场景 | 起步配置 | 推荐配置 |
|---------|---------|---------|
| 本地学习 | 4 vCPU, 8 GB | 8 vCPU, 16 GB |
| 运行案例 | 4 vCPU, 8 GB | 8 vCPU, 16 GB |
| 本地模型（Ollama） | 8 vCPU, 16 GB | 16 vCPU, 32 GB + GPU |
| 运行实战项目 | 8 vCPU, 16 GB | 16 vCPU, 32 GB |

---

## 核心知识点总结

### 必须掌握的 12 个概念

1. **提示词工程**: 核心原则、链式思维、Few-shot、多轮对话与记忆
2. **Coze/Dify 低代码平台**: 工作流、Agent、知识库、插件、Python 调用
3. **LangChain 框架**: Model I/O、Prompt、Parser、LCEL、Memory、Tools、RAG、Agent
4. **LangGraph 框架**: 图式思维、State/Node/Edge、持久化记忆、流式输出、多智能体
5. **MCP 协议**: 与 Function Calling 对比、通信机制、Server 部署与自定义开发
6. **RAG 系统**: 向量数据库、多路召回、重排序、RAGAS 评估、高级优化
7. **Agent 智能体**: Tool Calling、Skills、人机协作、中断恢复
8. **微调技术**: PEFT、LoRA、QLoRA、全参数微调、DeepSpeed、Llama-Factory
9. **部署运维**: Docker、Ollama、Xinference、vLLM、云端部署
10. **电商问数项目**: NL2SQL、元数据管理、Qdrant、ES、LangGraph 工作流
11. **深度研搜项目**: DeepAgents 多智能体、Tavily、MySQL、RAGFlow、FastAPI、WebSocket
12. **面试题库**: 按岗位能力域组织的问法与答法，对齐培训班课表与 JD

---

## 常见疑问解答

### Q1: 零基础能学吗？

**完全可以**。教程从大模型基础概念开始，逐步深入到提示词工程、框架使用、实战项目。每个案例都提供可运行源码和环境说明。

### Q2: 必须用 Python 吗？Java/前端可以学吗？

**主线是 Python**。教程聚焦 Python + LangChain + LangGraph 技术路线，不走 Spring AI / langchain4j 的 Java 路线。但概念部分（提示词、RAG、Agent）对所有语言都适用。

### Q3: 需要 GPU 吗？

**学习阶段不需要**。使用云 API（通义千问、DeepSeek 等）可以在 CPU 上完成所有学习。如果想跑本地模型（Ollama），建议有 GPU，但不是必须。

### Q4: 两个实战项目有什么区别？

| 项目 | 技术栈 | 核心能力 | 适合场景 |
|------|--------|---------|---------|
| **电商问数** | LangGraph + MySQL + Qdrant + ES | NL2SQL、元数据管理、多路召回 | 结构化数据问数 |
| **深度研搜** | DeepAgents + Tavily + MySQL + RAGFlow | 多智能体调度、多来源检索、文件交付 | 开放研究任务 |

### Q5: 面试题库有什么用？

面试题库按**岗位能力域**组织，包含问法与答法。对齐同类**线上培训结业能力**与 **JD 高频考点**，适合应届与转岗梳理口径。题库中有相当一部分题目整理自**大厂真实面试题**。

---

## 延伸学习路径

1. **在线阅读**: [完整教程在线文档](https://didilili.github.io/ai-agents-from-zero/#/)
2. **系统学习路线**:
   - 先学概念篇（1-9 章）：大模型基础、提示词、RAG/微调/Agent 选型
   - 再学框架篇（10-26 章）：LangChain、LangGraph、MCP、Agent
   - 然后学微调篇（27-28 章）：PEFT、LoRA、Llama-Factory
   - 最后做实战项目：电商问数 + 深度研搜
3. **进阶主题**:
   - RAG 系统优化：多路召回、重排序、RAGAS 评估
   - 多智能体编排：LangGraph 多智能体、A2A 协议
   - Skills 开发：编写自己的技能模块
   - 生产部署：Docker、Ollama、Xinference、vLLM

---

## 与教程目录的关联

本项目与以下教程内容强关联：

| 教程章节 | 关联点 |
|---------|-------|
| `06-单Agent开发与框架/01-Agent核心概念` | Agent 基础概念、Tool Calling、Skills |
| `06-单Agent开发与框架/03-Agent框架与编排` | LangChain、LangGraph 框架使用 |
| `07-多Agent与Agent工程化/01-多Agent系统` | 深度研搜多智能体架构 |
| `05-工具调用与RAG/01-RAG系统` | 电商问数 RAG 应用 |
| `04-推理与部署/01-推理引擎与KV-Cache` | Ollama、Xinference、vLLM 部署 |

---

## 总结

**AI Agents From Zero 不是碎片化教程，而是系统化、可运行、面试对齐的完整学习路线**。它展示了：

1. ✅ **系统完整**: 28 章教程，从基础到实战，覆盖大模型应用开发全链路
2. ✅ **可跑源码**: 每个案例都提供可运行源码，环境说明和常见问题排查
3. ✅ **实战项目**: 2 个企业级项目（电商问数 + 深度研搜），完整主链路闭环
4. ✅ **面试对齐**: 按岗位能力域组织的面试题库，对齐培训班课表与 JD
5. ✅ **持续进化**: 伴随 AI 技术栈持续更新，面向 2026

**面试中可以讲的亮点**:
- 我系统学习了 AI Agents From Zero 教程，掌握 LangChain、LangGraph、DeepAgents 等框架
- 我独立跑通了电商问数项目，理解 NL2SQL、元数据管理、多路召回、LangGraph 工作流
- 我完成了深度研搜项目，掌握多智能体调度、多来源工具接入、FastAPI + WebSocket 实时通信
- 我熟悉面试题库中的能力域问题，能讲清楚 RAG、Agent、MCP 的设计与取舍

---

**学习建议**: 这是一套长期维护的系统化教程，不要只看不跑！先按快速开始跑通案例，然后深入理解框架原理，最后完成实战项目。面试前刷一遍面试题库，梳理自己的项目表达口径。
