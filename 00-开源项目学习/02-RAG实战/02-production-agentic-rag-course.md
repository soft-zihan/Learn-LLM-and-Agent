# production-agentic-rag-course - 生产级 RAG 教程学习指南

## 项目概述

### 一句话总结
这是一个**7 周完整课程**，教你从零搭建**生产级 RAG（检索增强生成）系统**，最终打造出一个能自动抓取学术论文、理解内容并回答研究问题的 AI 助手。

### 核心亮点
- **渐进式学习路径**：从基础设施 → 数据管道 → 搜索 → RAG → 监控 → Agent，每周一个模块
- **真实生产栈**：Docker + FastAPI + PostgreSQL + OpenSearch + Airflow + Redis + Langfuse
- **先搜索后 AI**：不同于大多数教程直接上向量搜索，本项目先教你掌握 BM25 关键词搜索基础，再叠加语义理解
- **本地运行零成本**：使用 Ollama 本地 LLM，几乎不需要花钱
- **最终成果实用**：完成后可拥有一个支持 Telegram 的移动端 AI 研究助手

### 适合谁学
- **AI/ML 工程师**：想学习 RAG 的生产级架构，而非玩具级 demo
- **后端开发工程师**：想掌握如何把 LLM 集成到真实业务系统中
- **数据科学家**：想把模型能力变成端到端可用的产品

---

## 核心架构解析

### 生产级 RAG 系统整体架构

想象你在经营一家**图书馆**：

1. **数据采集（Week 2）**：像图书采购员，自动从 arXiv 抓取最新论文
2. **存储管理（Week 1）**：PostgreSQL 存放图书目录（元数据），OpenSearch 存放图书内容（全文索引）
3. **检索系统（Week 3-4）**：
   - BM25 关键词搜索 = 按书名和关键词找书
   - 向量语义搜索 = 理解你真正想找什么主题的书
   - 混合搜索 = 两者结合，找到最相关的书
4. **问答系统（Week 5）**：找到书后，让一个聪明的图书管理员（LLM）阅读并回答你的问题
5. **生产优化（Week 6）**：安装监控摄像头（Langfuse 追踪）和缓存柜（Redis），提升性能和可观测性
6. **智能助手（Week 7）**：升级为能自主决策的智能管家（Agent），还能通过 Telegram 随时交流

### 技术栈组成

| 组件 | 角色比喻 | 端口 |
|------|---------|------|
| **FastAPI** | 前台接待员，处理所有请求 | 8000 |
| **PostgreSQL 16** | 图书目录数据库，存论文元数据 | 5432 |
| **OpenSearch 2.19** | 全文搜索引擎，支持关键词+向量混合搜索 | 9200/5601 |
| **Apache Airflow 3.0** | 自动化流水线调度器 | 8080 |
| **Ollama** | 本地 AI 大脑，运行 LLM | 11434 |
| **Redis** | 高速缓存，记住常见问题的答案 | 6379 |
| **Langfuse** | 监控系统，追踪每次请求的全过程 | 3000 |

### 每周学习内容概览

```
Week 0 ── 项目总览：AI 项目的六大阶段
  ↓
Week 1 ── 基础设施：Docker 编排所有服务，FastAPI 搭建 API
  ↓
Week 2 ── 数据管道：从 arXiv 自动抓取论文，解析 PDF，存入数据库
  ↓
Week 3 ── 关键词搜索：BM25 算法，OpenSearch 索引管理
  ↓
Week 4 ── 混合搜索：文本分块 + 向量嵌入 + RRF 融合算法
  ↓
Week 5 ── 完整 RAG：接入 LLM，流式响应，Gradio 交互界面
  ↓
Week 6 ── 生产优化：Langfuse 全链路追踪 + Redis 缓存（提速 150-400 倍）
  ↓
Week 7 ── Agent 升级：LangGraph 智能决策 + Telegram 机器人
```

---

## 代码逻辑主线

### 从基础设施到生产的完整链路

项目的代码组织遵循**标准分层架构**，位于 `src/` 目录下：

```
src/
├── routers/          # API 层：定义 HTTP 端点
├── services/         # 业务逻辑层：各组件的具体实现
├── models/           # 数据模型层：数据库表结构
├── schemas/          # 数据验证层：请求/响应格式定义
└── config.py         # 配置中心：环境变量管理
```

### 核心代码流程

**1. 数据流入（Week 2）**

```
arXiv API → ArxivClient → MetadataFetcher → PDFParserService → PostgreSQL
```

- [ArxivClient](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/production-agentic-rag-course/src/services/arxiv/)：带速率限制和重试逻辑的论文抓取器
- [MetadataFetcher](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/production-agentic-rag-course/src/services/metadata_fetcher.py)：协调整个数据管道的主编排器
- [PDFParserService](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/production-agentic-rag-course/src/services/pdf_parser/)：使用 Docling 解析科学论文 PDF

**2. 索引构建（Week 3-4）**

```
论文内容 → TextChunker → Embeddings Service → OpenSearch
```

- [TextChunker](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/production-agentic-rag-course/src/services/indexing/text_chunker.py)：按章节智能分块，保留文档结构
- [Embeddings Service](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/production-agentic-rag-course/src/services/embeddings/)：使用 Jina AI 生成向量嵌入

**3. 检索与回答（Week 3-5）**

```
用户问题 → 搜索 API → OpenSearch(BM25/混合) → 检索相关段落 → LLM 生成回答
```

- [Search Router](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/production-agentic-rag-course/src/routers/search.py)：BM25 关键词搜索端点
- [Hybrid Search Router](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/production-agentic-rag-course/src/routers/hybrid_search.py)：统一搜索 API，支持关键词/向量/混合模式
- [Ask Router](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/production-agentic-rag-course/src/routers/ask.py)：完整 RAG 问答端点（普通 + 流式）

**4. 智能决策（Week 7）**

```
用户问题 → 边界检测 → 检索 → 相关性评分 → (不够好? 重写查询) → 生成回答
```

- [Agent Nodes](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/production-agentic-rag-course/src/services/agents/nodes/)：守卫、检索、评分、重写、生成等决策节点
- [Agentic RAG Workflow](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/production-agentic-rag-course/src/services/agents/agentic_rag.py)：LangGraph 工作流编排

### 关键组件交互

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│   用户请求   │ ───> │   FastAPI    │ ───> │  搜索服务     │
│ (Gradio/    │      │  (API 层)    │      │ (OpenSearch) │
│  Telegram)  │      └──────────────┘      └──────────────┘
└─────────────┘             │                     │
                            │                     │
                            ▼                     ▼
                     ┌──────────────┐      ┌──────────────┐
                     │   LLM 服务   │ <─── │  检索结果     │
                     │  (Ollama)    │      │  (文档块)     │
                     └──────────────┘      └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  缓存/监控    │
                     │ (Redis/      │
                     │  Langfuse)   │
                     └──────────────┘
```

---

## 快速上手实践

### 环境配置步骤

**第 1 步：安装必备工具**
```bash
# 确保已安装以下工具
- Docker Desktop（含 Docker Compose）
- Python 3.12+
- UV 包管理器：https://docs.astral.sh/uv/getting-started/installation/
```

**第 2 步：克隆并配置**
```bash
cd production-agentic-rag-course

# 复制环境配置文件
cp .env.example .env

# 安装 Python 依赖
uv sync
```

**第 3 步：启动所有服务**
```bash
# 一键启动（推荐使用 Makefile）
make start

# 或直接使用 Docker Compose
docker compose up --build -d
```

**第 4 步：验证安装**
```bash
# 检查服务健康状态
curl http://localhost:8000/api/v1/health

# 或查看所有服务状态
docker compose ps
```

### 运行第一个示例

**方式 1：使用 Jupyter Notebook（推荐学习）**
```bash
# 打开 Week 1 的交互式教程
uv run jupyter notebook notebooks/week1/week1_setup.ipynb
```

**方式 2：使用 Gradio 界面（体验最终效果）**
```bash
# 启动 Web 聊天界面
uv run python gradio_launcher.py

# 浏览器打开
http://localhost:7861
```

### 预期输出与验证方法

| 服务 | 访问地址 | 验证方法 |
|------|---------|---------|
| API 文档 | http://localhost:8000/docs | 能看到交互式 API 测试界面 |
| Gradio 聊天 | http://localhost:7861 | 能输入问题并获得回答 |
| Langfuse 监控 | http://localhost:3000 | 能看到请求追踪记录 |
| Airflow 调度 | http://localhost:8080 | 能看到数据管道任务 |
| OpenSearch | http://localhost:5601 | 能搜索已索引的文档 |

**快速测试命令：**
```bash
# 测试健康检查
curl http://localhost:8000/health

# 测试搜索 API（需要先有数据）
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention", "size": 5}'
```

---

## 核心知识点总结

### 1. RAG（检索增强生成）是什么？

**通俗解释**：RAG 就像**开卷考试**。与其让 AI 死记硬背所有知识（容易记错），不如给它一个图书馆，让它先查找资料再回答问题。

**为什么重要**：这是当前企业应用 LLM 最主流的方式，能解决幻觉问题、保证答案基于真实数据、支持知识实时更新。

### 2. BM25 关键词搜索

**通俗解释**：比简单的"包含关键词"更聪明的搜索算法。它不仅看你是否包含某个词，还考虑：
- 词在文档中出现的频率（TF）
- 这个词在整个数据集中有多稀有（IDF）
- 文档长度 normalization

**为什么重要**：大多数教程跳过这一步直接上向量搜索，但**生产环境中的 RAG 系统 90% 仍然依赖 BM25 作为基础**。它是精确匹配的利器，尤其适合专有名词、技术术语。

### 3. 向量语义搜索

**通俗解释**：把文本变成一串数字（向量），这串数字代表文本的"含义"。含义相似的文本，向量在空间中距离就近。搜索时不是匹配关键词，而是找"含义最接近"的文档。

**为什么重要**：能理解同义词、上下文，解决 BM25 无法处理语义的问题。比如搜索"机器学习"也能找到"深度学习"相关内容。

### 4. 混合搜索与 RRF 融合

**通俗解释**：把关键词搜索和语义搜索的结果合并。RRF（Reciprocal Rank Fusion）是一种聪明的合并算法：**两个算法都排在前面的文档，最终排名会更高**。

**为什么重要**：单一搜索方式总有盲区，混合搜索能结合两者的优势，是**生产级 RAG 的最佳实践**。

### 5. 文本分块（Chunking）策略

**通俗解释**：把长文档切成小块再索引。就像把一本厚书拆成章节，检索时定位到具体章节而不是整本书。

**本项目的做法**：按论文结构（标题、摘要、章节）智能分块，而不是机械地按字数切分。

**为什么重要**：分块策略直接影响检索质量。块太大包含太多噪声，块太小丢失上下文。这是 RAG 系统中**最容易被忽视但最关键**的环节之一。

### 6. LangGraph Agent 工作流

**通俗解释**：让 AI 不止是"检索→回答"，而是能**自主决策**：
- 这个问题我能回答吗？（边界检测）
- 检索的结果够好吗？（相关性评分）
- 不够好怎么办？重写问题再试一次（查询重写）
- 最终生成回答

**为什么重要**：这是从"被动检索"到"主动思考"的升级，**代表 RAG 技术的最新发展方向**。

### 7. 生产级可观测性

**通俗解释**：在生产环境中，你需要知道：
- 每次请求走了哪些步骤？（Langfuse 追踪）
- 哪些答案被缓存了？（Redis 缓存）
- 系统响应速度如何？成本多少？

**为什么重要**：没有监控的系统就像**盲飞**。Langfuse 能让你看到 RAG 管道的每一步，快速定位问题、优化性能。Redis 缓存能让相同问题的响应速度提升 **150-400 倍**。

### 8. Docker Compose 服务编排

**通俗解释**：用一份配置文件（compose.yml）同时启动 7-8 个服务，并管理它们之间的网络、依赖关系。

**为什么重要**：现代 AI 应用不再是一个脚本，而是**多个服务的协作**。掌握 Docker Compose 是搭建复杂系统的必备技能。

---

## 常见疑问解答

### Q1: 为什么要先学 BM25 而不是直接学向量搜索？

**这是本项目最大的教学亮点**。原因在于：

1. **生产现实**：真实企业中，BM25 仍然是搜索的基础设施。很多场景（如代码搜索、产品搜索）关键词匹配比语义更重要
2. **可解释性**：BM25 的评分逻辑是透明的，你能理解为什么某个文档排前面。向量搜索是黑盒
3. **成本与速度**：BM25 不需要生成向量，速度快、成本低
4. **混合搜索基础**：最好的 RAG 系统是 BM25 + 向量的混合，不懂 BM25 就做不好混合

> 💡 **比喻**：就像学数学要先学加减乘除，再学微积分。BM25 是搜索的"基本功"。

### Q2: 我的电脑配置不够怎么办？

**最低要求**：
- 8GB RAM（16GB 推荐）
- 20GB 磁盘空间

**如果配置较低**：
- 可以在 Week 5 之前不使用 Ollama（不需要本地 LLM）
- 减少 Docker 内存分配
- 按周逐步启动服务，不需要一次性全部运行

### Q3: 需要花钱吗？

**几乎免费**：
- 所有基础设施都是开源的（Docker、PostgreSQL、OpenSearch 等）
- LLM 使用 Ollama 本地运行，零成本
- 唯一可选的花费：Jina AI 的 Embedding API（有免费额度），或者你也可以用本地的 embedding 模型

### Q4: 这个项目和 LangChain 教程有什么区别？

**核心差异**：

| 方面 | 普通 LangChain 教程 | 本项目 |
|------|-------------------|--------|
| 搜索 | 直接用向量数据库 | 先学 BM25，再混合搜索 |
| 架构 | 单脚本 demo | 完整微服务架构 |
| 监控 | 通常不涉及 | Langfuse + Redis |
| 部署 | 本地跑通就行 | Docker 生产级编排 |
| 数据管道 | 手动准备数据 | Airflow 自动化流水线 |

本项目教你的是**如何在企业中真正部署和维护 RAG 系统**，而不仅仅是跑通一个 demo。

### Q5: 7 周内容太多，我应该按什么顺序学？

**推荐路径**：

**速成版（想快速看到效果）**：
1. Week 1（基础设施）→ Week 5（完整 RAG）
2. 先用起来，再回头理解细节

**扎实版（想真正掌握）**：
1. 按顺序每周一个模块
2. 每个 Notebook 都要动手跑一遍
3. 阅读对应的博客文章加深理解

**按需版（已有基础）**：
- 懂搜索？跳过 Week 3，直接看 Week 4 混合搜索
- 懂 RAG 基础？从 Week 6 生产优化开始
- 想学 Agent？直接看 Week 7

---

## 学习建议

### 如何高效使用这个项目

1. **先跑通再理解**：先按照 Notebook 跑起来，看到效果后再深入代码
2. **每周一个博客**：每个 Week 都有对应的 Substack 博客，配合代码阅读效果更好
3. **使用 Git Tag**：可以用 `git clone --branch week3.0` 只克隆某一周的代码，避免信息过载
4. **动手改代码**：尝试修改 prompt、调整分块策略、更换 LLM，观察效果变化

### 学完后你能做什么

- ✅ 搭建任意领域的 RAG 系统（企业知识库、客服、代码助手等）
- ✅ 设计生产级的搜索架构（BM25 + 向量混合）
- ✅ 实现 AI 系统的监控和缓存优化
- ✅ 构建基于 Agent 的智能问答系统
- ✅ 用 Docker 编排复杂的多服务 AI 应用

---

**项目作者**：[Shirin Khosravi Jam](https://www.linkedin.com/in/shirin-khosravi-jam/) & [Shantanu Ladhwe](https://www.linkedin.com/in/shantanuladhwe/)  
**学习博客**：https://jamwithai.substack.com/  
**GitHub**：https://github.com/jamwithai/production-agentic-rag-course
