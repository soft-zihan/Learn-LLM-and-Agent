# agentic-rag-for-dummies - LangGraph Agentic RAG 入门学习指南

## 项目概述

**一句话总结**：这是一个用 LangGraph 构建的**智能对话式 RAG 系统**，不仅能检索文档回答问题，还能理解多轮对话、主动澄清模糊问题、并并行处理复杂查询。

### 核心亮点

- 🗂️ **父子双层索引**：像"目录+正文"一样组织文档，搜索精准且回答内容丰富
- 🧠 **对话记忆**：能理解"它怎么更新？"中的"它"指代什么
- ❓ **主动澄清**：遇到模糊问题会暂停并追问用户，而不是盲目检索
- 🤖 **多Agent并行**：复杂问题自动拆分成子问题，多个Agent同时检索
- ✅ **自我纠错**：检索结果不足时自动扩大搜索范围
- 🗜️ **上下文压缩**：长对话中自动精简记忆，避免信息过载

### 适合谁学

- 已经了解基础 RAG（检索增强生成）概念，想学习 **Agent 驱动的高级 RAG**
- 想掌握 **LangGraph 状态图编排** 的实际应用
- 需要构建支持**多轮对话、复杂查询**的生产级问答系统

---

## 核心架构解析

### Agentic RAG 整体架构

把这个系统想象成一个**研究团队**：

```
用户提问 → 接待员（总结历史） → 翻译官（重写问题） → 判断是否需要澄清
    ↓
如果需要澄清 → 暂停等待用户补充
    ↓
如果问题清晰 → 分发任务给多个研究员（并行Agent）
    ↓
每个研究员：搜索 → 分析 → 自我纠错 → 提交答案
    ↓
总编辑（聚合答案） → 返回最终回答
```

整个流程由 **LangGraph 状态图** 驱动，数据像流水线一样在各个节点间传递。

### LangGraph 状态图设计

项目使用**双层图架构**（见 [project/rag_agent/graph.py](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/graph.py)）：

#### 1. 主图（Main Graph）- 协调整个流程

```
START → summarize_history（总结对话历史）
    ↓
rewrite_query（重写问题 + 判断清晰度）
    ↓
    ├─ 问题不清晰 → request_clarification（暂停等待用户）
    └─ 问题清晰 → 并行启动多个 agent 子图（Send API）
    ↓
aggregate_answers（聚合所有Agent答案）
    ↓
END
```

#### 2. Agent 子图（Agent Subgraph）- 处理单个问题

```
START → orchestrator（LLM决策：搜索还是回答）
    ↓
route_after_orchestrator_call（路由判断）
    ├─ 需要工具 → tools（执行检索） → should_compress_context（检查记忆容量）
    │       ├─ 记忆满了 → compress_context（压缩记忆） → orchestrator（继续）
    │       └─ 记忆正常 → orchestrator（继续）
    ├─ 预算耗尽 → fallback_response（基于已有信息尽力回答）
    └─ 完成检索 → collect_answer（收集答案）
    ↓
END
```

### 关键节点与边

| 节点 | 文件位置 | 职责 |
|------|---------|------|
| `summarize_history` | [nodes.py:10-28](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/nodes.py#L10-L28) | 提取最近6轮对话，生成30-50字摘要 |
| `rewrite_query` | [nodes.py:30-44](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/nodes.py#L30-L44) | 重写问题使其自包含，可拆分成最多3个子问题 |
| `route_after_rewrite` | [edges.py:6-13](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/edges.py#L6-L13) | **条件边**：决定是澄清还是启动Agent |
| `orchestrator` | [nodes.py:50-65](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/nodes.py#L50-L65) | Agent大脑，决定调用工具还是直接回答 |
| `should_compress_context` | [nodes.py:96-125](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/nodes.py#L96-L125) | **智能路由**：根据token数量决定是否压缩记忆 |
| `aggregate_answers` | [nodes.py:176-188](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/nodes.py#L176-L188) | 将多个Agent答案融合成自然流畅的单一回答 |

---

## 代码逻辑主线

### 核心代码流程

以一个完整对话为例：

**场景**：用户先问"如何安装SQL？"，然后问"怎么更新它？"

#### 第一轮对话

1. **总结历史**（[nodes.py:10](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/nodes.py#L10)）
   - 对话少于4条消息，返回空摘要
   
2. **重写问题**（[nodes.py:30](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/nodes.py#L30)）
   - 调用 LLM + [QueryAnalysis](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/schemas.py) 结构化输出
   - 判断问题清晰，返回 `["如何安装SQL？"]`

3. **路由**（[edges.py:6](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/edges.py#L6)）
   - `questionIsClear=True`，使用 `Send` API 启动Agent子图

4. **Agent执行**（[nodes.py:50](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/nodes.py#L50)）
   - `orchestrator` 调用 `search_child_chunks` 工具（见 [tools.py:11](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/tools.py#L11)）
   - 在Qdrant向量库中搜索相关chunk
   - 根据需要调用 `retrieve_parent_chunks` 获取完整上下文（见 [tools.py:37](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/tools.py#L37)）
   - 检查token预算（[nodes.py:96](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/nodes.py#L96)）
   - 生成答案，返回给主图

5. **聚合答案**（[nodes.py:176](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/nodes.py#L176)）
   - 只有一个Agent答案，直接格式化为最终回答

#### 第二轮对话

1. **总结历史**
   - 提取第一轮对话，生成摘要："用户询问了SQL安装步骤，来源：javascript.pdf"

2. **重写问题**
   - 输入：`current_query="怎么更新它？"` + `conversation_summary="用户询问SQL安装..."`
   - LLM理解"它"=SQL，重写为："如何更新SQL？"

3. **Agent执行**
   - 基于重写后的精准问题检索
   - 返回更新步骤

### 关键函数调用链

```
用户提问
  ↓
chat_interface.py: chat() 
  ↓
graph.invoke({"messages": [...]}, config)
  ↓
summarize_history(state, llm)
  ↓
rewrite_query(state, llm)
  → llm.with_structured_output(QueryAnalysis).invoke()
  ↓
route_after_rewrite(state)
  → Send("agent", {...})  # 并行启动
  ↓
[每个Agent子图]
  orchestrator(state, llm_with_tools)
    → llm_with_tools.invoke()  # 可能触发工具调用
  ↓
  tools (ToolNode)
    → search_child_chunks(query, limit)
    → retrieve_parent_chunks(parent_id)
  ↓
  should_compress_context(state)
    → estimate_context_tokens(messages)  # 计算token
    → Command(goto="compress_context" or "orchestrator")
  ↓
  collect_answer(state)
  ↓
aggregate_answers(state, llm)
  → llm.invoke([SystemMessage(content=get_aggregation_prompt()), ...])
  ↓
返回最终答案
```

### RAG 决策逻辑

项目的智能体现在**多层决策**：

1. **清晰度判断**（[nodes.py:39](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/nodes.py#L39)）
   ```python
   if response.questions and response.is_clear:
       # 问题清晰，启动Agent
   else:
       # 请求澄清
   ```

2. **工具调用 vs 直接回答**（[edges.py:15](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/edges.py#L15)）
   ```python
   if not tool_calls:
       return "collect_answer"  # 没有工具调用，说明LLM认为可以直接回答
   ```

3. **预算控制**（[edges.py:19](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/edges.py#L19)）
   ```python
   if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
       return "fallback_response"  # 防止无限循环
   ```

4. **上下文压缩触发**（[nodes.py:124](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/nodes.py#L124)）
   ```python
   max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)
   goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
   ```

---

## 快速上手实践

### 环境配置步骤

#### 方式一：Jupyter Notebook（推荐快速体验）

```bash
# 1. 进入项目目录
cd agentic-rag-for-dummies

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 Ollama（本地LLM）
# 访问 https://ollama.com 下载并安装

# 4. 拉取模型
ollama pull qwen3:4b-instruct-2507-q4_K_M

# 5. 准备PDF文档
# 将PDF文件放入 docs/ 目录

# 6. 打开 notebook
jupyter notebook notebooks/agentic_rag.ipynb
```

#### 方式二：完整Python项目（推荐深入学习）

```bash
# 1. 安装依赖
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置环境变量（可选，用于云服务）
cp project/.env.example project/.env
# 编辑 .env 填入 API Key

# 3. 运行应用
python project/app.py

# 4. 浏览器访问 http://127.0.0.1:7860
```

### 运行第一个示例

**测试场景1：基础问答**

```
用户：JavaScript中如何声明变量？
预期：Agent检索javascript.pdf，返回var/let/const的说明
```

**测试场景2：多轮对话记忆**

```
用户：Python的列表推导式是什么？
Agent：[解释列表推导式语法]

用户：它能用于字典吗？
预期：Agent理解"它"指列表推导式，回答字典推导式
```

**测试场景3：模糊问题澄清**

```
用户：告诉我关于那个东西的信息
Agent：我需要更多信息。您想了解哪个具体主题？

用户：PostgreSQL的安装流程
Agent：[检索并返回安装步骤]
```

### 预期输出与验证方法

1. **验证向量索引是否成功**
   - 运行后检查 `qdrant_db/` 目录是否生成
   - 检查 `parent_store/` 目录是否有JSON文件

2. **验证对话记忆**
   - 连续提问3-4个问题
   - 第二个问题使用代词（"它"、"这个"）
   - 观察Agent是否正确理解上下文

3. **验证并行处理**
   - 提问："什么是JavaScript？什么是Python？"
   - 应该看到两个Agent同时工作（在notebook中可观察）

4. **验证自我纠错**
   - 提问一个文档中信息较少的问题
   - Agent应该自动扩大搜索范围或承认信息不足

---

## 核心知识点总结

### 1. Agentic RAG vs 传统 RAG

**传统 RAG**：用户问题 → 检索 → 生成回答（一次性流程）

**Agentic RAG**：问题 → 分析 → 检索 → 评估结果 → 不满意则重新检索 → 生成回答（循环决策）

**为什么重要**：传统RAG遇到复杂问题就失败，Agent能通过"思考-行动-观察"循环自主解决问题。

### 2. LangGraph 状态图

状态图 = **节点（处理函数）** + **边（数据流向）** + **状态（共享数据）**

类比：就像工厂流水线，每个工位（节点）加工零件，传送带（边）传递，仓库（状态）存放半成品。

**为什么重要**：LangGraph 让复杂的多步决策流程可视化、可调试、可扩展。

### 3. 父子双层索引（Parent-Child Chunking）

```
父Chunk（2000-4000字）：完整的章节，保留上下文
  ↓ 切分
子Chunk（500字）：精准片段，用于检索
```

搜索时查子Chunk（精准），回答时用父Chunk（完整上下文）。

**为什么重要**：解决传统RAG"检索精准但回答缺少上下文"的痛点。

### 4. Send API 与并行Agent

```python
Send("agent", {"question": query, "question_index": idx})
```

`Send` 可以同时启动多个Agent子图，每个独立处理一个子问题。

**为什么重要**：复杂问题分解后并行处理，大幅提升效率和准确率。

### 5. 上下文压缩（Context Compression）

当对话历史超过token阈值（默认2000），调用LLM将冗长对话压缩成结构化摘要，同时记录已检索的ID避免重复。

**为什么重要**：防止长对话中上下文窗口爆炸，同时保持关键信息不丢失。

### 6. Human-in-the-Loop（人在环路）

```python
agent_graph.compile(interrupt_before=["request_clarification"])
```

`interrupt_before` 让图执行到特定节点暂停，等待用户输入后再继续。

**为什么重要**：让AI知道何时该"承认不懂"并求助人类，避免瞎编答案。

### 7. 条件边（Conditional Edges）

```python
add_conditional_edges("rewrite_query", route_after_rewrite)
```

根据状态数据动态决定下一步走向哪个节点，是图的核心分支逻辑。

**为什么重要**：实现智能决策流，让系统能处理不同场景。

### 8. 状态累加器（Annotated Reducers）

```python
agent_answers: Annotated[List[dict], accumulate_or_reset] = []
retrieval_keys: Annotated[Set[str], set_union] = set()
```

自定义状态更新规则：列表可以追加或清空，集合可以合并去重。

**为什么重要**：优雅处理并行Agent的答案收集和去重。

---

## 常见疑问解答

### Q1：为什么需要两层图（主图+Agent子图）？不能写在一个图里吗？

**答**：可以，但不推荐。双层图的核心价值是**并行化**：

- 主图负责**流程编排**（总结→重写→分发→聚合）
- Agent子图负责**单个问题的深度检索**
- 当用户问"JavaScript和Python有什么区别？"时，主图用 `Send` 同时启动2个Agent子图，一个查JavaScript，一个查Python，**并行执行**而非串行

如果写在一个图里，就只能串行处理，效率低下且代码难以维护。

### Q2：上下文压缩会不会丢失重要信息？

**答**：这是精心设计的平衡：

1. **压缩不是丢弃**：LLM会提取关键事实、数据、技术术语，去除的是重复内容和客套话
2. **渐进式压缩**：每次压缩基于已有摘要+新内容，不是从零开始
3. **防重复机制**：`retrieval_keys` 记录已检索的ID，附加在摘要末尾，确保Agent不会重复检索相同内容
4. **可调阈值**：`BASE_TOKEN_THRESHOLD=2000` 可以调整，延迟压缩时机以保留更多信息

实际测试表明，良好的压缩提示词（见 [prompts.py:118](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/prompts.py#L118)）能保留90%以上的关键信息，同时减少70%的token消耗。

### Q3：如果LLM不调用工具直接回答怎么办？

**答**：项目有多层防护：

1. **强制指令**：在orchestrator节点首次调用时，额外发送消息 `YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP...`（见 [nodes.py:59](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/nodes.py#L59)）

2. **系统提示词**：orchestrator提示词明确要求 `You MUST call 'search_child_chunks' before answering`（见 [prompts.py:64](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/rag_agent/prompts.py#L64)）

3. **模型选择**：README 明确建议 **7B+ 模型**，小模型（如4B）可能忽略指令

4. **温度设置**：`temperature=0` 确保输出一致性，避免随机性导致不遵循指令

如果仍然发生，可以：
- 换用更强的模型（如 GPT-4o、Claude Sonnet）
- 在系统提示词中进一步强化检索要求
- 降低检索分数阈值 `RETRIEVAL_SCORE_THRESHOLD`

### Q4：如何调试和观察Agent的执行过程？

**答**：三种方式：

1. **Notebook 模式**（推荐学习）
   - 打开 `notebooks/agentic_rag.ipynb`
   - 每个单元格可单独执行，观察中间状态
   - 支持流式输出，实时查看Agent思考过程

2. **Langfuse 可观测性**（推荐生产）
   - 配置 `LANGFUSE_ENABLED=true`
   - 访问 Langfuse Dashboard 查看：
     - LLM调用追踪
     - 工具使用统计
     - 图执行路径

3. **手动打印状态**
   - 在节点函数中添加 `print(state)` 查看数据流转
   - 使用 `agent_graph.get_state(config)` 获取当前状态

### Q5：如何适配自己的文档和模型？

**答**：只需修改 [config.py](file:///Users/han/Documents/projects/Learn-LLM-and-Agent/99-开源项目学习/02-RAG实战/agentic-rag-for-dummies/project/config.py)：

```python
# 换模型
LLM_MODEL = "gpt-4o-mini"  # 或 "claude-sonnet-4-5-20250929"
DENSE_MODEL = "text-embedding-3-small"

# 调整分块策略
CHILD_CHUNK_SIZE = 800  # 更大的chunk
MIN_PARENT_SIZE = 3000  # 更大的parent

# 调整检索
RETRIEVAL_SCORE_THRESHOLD = 0.3  # 降低阈值提高召回
DEFAULT_RETRIEVAL_K = 10  # 检索更多结果
```

文档准备：
- PDF放入 `docs/` 目录
- 系统自动转换为Markdown并建立索引
- 也可以直接使用已有的Markdown文件放入 `markdown_docs/`

---

## 下一步学习建议

1. **阅读 Notebook**：`notebooks/agentic_rag.ipynb` 包含详细的逐步解释
2. **实验修改**：尝试调整 config.py 参数，观察效果变化
3. **评估系统**：运行 `notebooks/evaluation.ipynb` 学习如何用 RAGAS 评估RAG质量
4. **可观测性**：运行 `notebooks/observability.ipynb` 学习追踪LLM调用
5. **生产部署**：参考 `project/README.md` 了解Docker部署和高级配置
