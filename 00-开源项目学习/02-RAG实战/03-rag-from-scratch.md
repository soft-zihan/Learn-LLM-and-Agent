# rag-from-scratch - LangChain RAG 从零构建学习指南

## 项目概述

**一句话总结**: 这是 LangChain 官方出品的 RAG 教学项目，通过 5 个 Notebook、18 个渐进式案例，带你从零理解并构建完整的 RAG（检索增强生成）系统。

**核心亮点**:
- 🎯 **由浅入深**: 从最基础的"索引-检索-生成"三步曲，逐步演进到多查询、路由、重排序等高级技术
- 📹 **配套视频**: 每个 Notebook 对应 B 站/YouTube 视频教程，边看边练
- 🔧 **纯实战**: 所有代码都是可运行的 Jupyter Notebook，无需额外项目结构
- 🧠 **全景图谱**: 学完后你会拥有一张完整的 RAG 技术地图

**适合谁学**:
- 想系统掌握 RAG 技术的初学者
- 已经会用 LangChain 但想深入理解底层原理的开发者
- 准备在企业项目中落地 RAG 的工程师

---

## 核心架构解析

### RAG 系统整体架构

RAG 的本质是**给大模型外挂一个知识库**。想象你在考试时可以翻书——RAG 就是让 AI 在回答问题前，先从你的文档中"翻书"找到相关资料，再结合这些资料作答。

整个系统分为三大阶段：

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  索引阶段    │────▶│  检索阶段     │────▶│  生成阶段     │
│  (Indexing) │     │  (Retrieval) │     │ (Generation) │
└─────────────┘     └──────────────┘     └──────────────┘
   文档切分             向量搜索            大模型回答
   向量化               重排序              上下文整合
   存储                 查询转换
```

### 关键模块拆解

#### 1️⃣ 索引模块 (Indexing)
**文件**: `rag_from_scratch_1_to_4.ipynb` - Part 2

索引阶段负责把文档变成可搜索的格式：

- **文档加载**: 使用 `WebBaseLoader` 从网页抓取内容
  ```python
  from langchain_community.document_loaders import WebBaseLoader
  loader = WebBaseLoader(web_paths)
  docs = loader.load()
  ```

- **文本切分**: 用 `RecursiveCharacterTextSplitter` 把长文档切成小块
  ```python
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  splits = text_splitter.split_documents(docs)
  ```
  **为什么要切分？** 因为大模型的上下文窗口有限，而且检索小片段比检索整本书更精准。

- **向量化**: 用 `OpenAIEmbeddings` 把文本转成向量
  ```python
  from langchain_openai import OpenAIEmbeddings
  embeddings = OpenAIEmbeddings()
  ```
  向量是文本的"数学指纹"，语义相似的文本向量在空间中距离更近。

- **存储**: 用 `Chroma` 向量数据库保存向量
  ```python
  from langchain_community.vectorstores import Chroma
  vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
  ```

#### 2️⃣ 检索模块 (Retrieval)
**文件**: `rag_from_scratch_1_to_4.ipynb` - Part 3

检索阶段负责找到与问题最相关的文档片段：

```python
retriever = vectorstore.as_retriever()
docs = retriever.get_relevant_documents(question)
```

底层使用**余弦相似度**计算问题向量与文档向量的距离，返回最接近的 K 个文档。

**高级检索技术** (在后续 Notebook 中展开):
- **多查询检索** (Part 5): 把一个问题变成多个变体，分别检索后合并结果
- **RAG-Fusion** (Part 6): 用多个查询的倒数排名融合结果
- **问题分解** (Part 7): 把复杂问题拆成子问题，逐个解答
- **Step-Back** (Part 8): 先问一个更抽象的"后退问题"获取背景知识
- **HyDE** (Part 9): 先生成假设性答案，再用答案去检索真实文档
- **重排序** (Part 15): 检索后用更强的模型对结果重新排序

#### 3️⃣ 生成模块 (Generation)
**文件**: `rag_from_scratch_1_to_4.ipynb` - Part 4

生成阶段把检索到的文档和问题一起交给大模型：

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

# 从 LangChain Hub 拉取预定义的 RAG prompt
prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 构建 RAG 链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 执行
result = rag_chain.invoke("What is RAG?")
```

**关键理解**: `|` 符号是 LangChain Expression Language (LCEL) 的管道操作符，类似于 Unix 的管道 `|`，数据从左流向右。

### 数据流向

以一个完整问答为例：

```
用户提问: "LangChain 是什么？"
    ↓
[检索] 向量数据库找到 3 个相关文档片段
    ↓
[格式化] 把文档拼成字符串: "文档1...\n\n文档2...\n\n文档3..."
    ↓
[组装 Prompt] 
  "你是 AI 助手。根据以下资料回答问题:
   资料: 文档1...

文档2...

文档3...
   问题: LangChain 是什么？"
    ↓
[生成] GPT-3.5 根据资料生成答案
    ↓
返回: "LangChain 是一个用于开发大语言模型应用的框架..."
```

---

## 代码逻辑主线

### Notebook 执行流程

项目包含 5 个 Notebook，按顺序学习：

| Notebook | 涵盖部分 | 核心主题 | 难度 |
|---------|---------|---------|------|
| `rag_from_scratch_1_to_4.ipynb` | Part 1-4 | RAG 基础: 索引、检索、生成 | ⭐ |
| `rag_from_scratch_5_to_9.ipynb` | Part 5-9 | 查询转换: 多查询、分解、HyDE | ⭐⭐ |
| `rag_from_scratch_10_and_11.ipynb` | Part 10-11 | 路由与查询结构化 | ⭐⭐⭐ |
| `rag_from_scratch_12_to_14.ipynb` | Part 12-14 | 高级索引: 多表示、RAPTOR、ColBERT | ⭐⭐⭐⭐ |
| `rag_from_scratch_15_to_18.ipynb` | Part 15-18 | 重排序、CRAG、长上下文 | ⭐⭐⭐⭐ |

### 关键函数调用链

以最基础的 RAG 流程为例：

```
WebBaseLoader.load()
    ↓ 返回 Document 列表
RecursiveCharacterTextSplitter.split_documents()
    ↓ 返回切分后的 Document 列表
OpenAIEmbeddings.embed_documents()
    ↓ 返回向量列表
Chroma.from_documents()
    ↓ 创建向量数据库
Chroma.as_retriever()
    ↓ 创建检索器
Retriever.get_relevant_documents()
    ↓ 返回相关文档
ChatPromptTemplate + ChatOpenAI.invoke()
    ↓ 生成最终答案
StrOutputParser.parse()
    ↓ 返回字符串
```

### 核心算法实现思路

#### 向量相似度检索
```python
# 伪代码展示核心逻辑
def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sqrt(sum(a ** 2 for a in vec1))
    norm2 = sqrt(sum(b ** 2 for b in vec2))
    return dot_product / (norm1 * norm2)

# 检索过程
question_vec = embedding_model.encode(question)
similarities = [cosine_similarity(question_vec, doc_vec) for doc_vec in document_vectors]
top_k_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
```

#### 多查询检索 (Part 5)
```python
# 核心思路: 一个问题 → 多个问题 → 合并结果
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 用 LLM 生成多个查询变体
multi_query_prompt = """你是一位 AI 助手。请为以下问题生成 3 个不同表述的变体:
原始问题: {question}
请输出 3 个变体，每行一个:"""

# 对每个变体分别检索，然后去重合并
```

---

## 快速上手实践

### 环境配置步骤

**第 1 步: 安装依赖**
```bash
pip install langchain langchain-openai langchain-community chromadb tiktoken bs4
```

**第 2 步: 配置 API Key**
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."  # 你的 OpenAI API Key
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # 可选: 启用 LangSmith 追踪
os.environ["LANGCHAIN_API_KEY"] = "lsv2_..."  # 可选: LangSmith API Key
```

**第 3 步: 启动 Jupyter**
```bash
cd 99-开源项目学习/02-RAG实战/rag-from-scratch
jupyter notebook
```

### 运行第一个示例

打开 `rag_from_scratch_1_to_4.ipynb`，按顺序执行单元格：

1. **Part 1: Overview** - 运行完整 RAG 示例
2. **Part 2: Indexing** - 观察文档加载、切分、向量化过程
3. **Part 3: Retrieval** - 测试检索效果
4. **Part 4: Generation** - 看到最终问答结果

### 预期输出与验证方法

执行完 Part 1 后，你应该能得到类似输出：

```python
# 输入
question = "What is RAG?"
result = rag_chain.invoke(question)

# 输出示例
print(result)
# "RAG stands for Retrieval-Augmented Generation. It's a technique that 
#  combines retrieval of relevant documents with generative AI models 
#  to improve the accuracy and relevance of generated responses..."
```

**验证方法**:
- ✅ 检查 `len(splits)` 应该大于 1（文档被成功切分）
- ✅ 检查 `len(query_result)` 应该是 1536（OpenAI embedding 维度）
- ✅ 检索返回的 `docs` 应该是列表，包含 3-4 个 Document 对象
- ✅ 最终答案应该与问题相关，且不是模型的"编造"内容

---

## 核心知识点总结

### 1. 为什么需要 RAG？
**问题**: 大模型的知识截止于训练数据，无法回答私有或最新信息。  
**解决**: RAG 通过检索外部文档，让模型"带着参考资料"回答问题。  
**类比**: 就像开卷考试，模型不再死记硬背，而是学会查资料。

### 2. 文本嵌入 (Embedding) 是什么？
**概念**: 把文本转成固定长度的向量（数字列表），语义相似的文本向量在空间中距离更近。  
**为什么重要**: 这是实现语义搜索的基础，让计算机能"理解"文本含义而非仅仅匹配关键词。  
**示例**: "苹果"和"水果"的向量距离 < "苹果"和"手机"的向量距离。

### 3. 为什么要切分文档？
**原因**:
- 大模型上下文窗口有限（如 4K/8K/128K tokens）
- 检索小片段比检索整本书更精准
- 减少无关信息干扰模型判断  
**切分策略**: `chunk_size` 控制每块大小，`chunk_overlap` 控制块之间的重叠（避免关键信息被切断）。

### 4. LCEL (LangChain Expression Language) 怎么用？
**概念**: 用 `|` 操作符串联组件，形成数据流管道。  
**示例**:
```python
chain = retriever | format_docs | prompt | llm | output_parser
```
**优势**: 代码简洁、易于调试、支持异步和流式输出。

### 5. 查询转换的价值是什么？
**核心思想**: 用户的问题往往不够好（太复杂、太模糊），直接检索效果差。  
**方法**:
- **多查询**: 一个问题变多个，提高召回率
- **问题分解**: 复杂问题拆成子问题
- **HyDE**: 先猜答案，再用答案检索（逆向思维）  
**类比**: 就像搜索引擎会自动纠正你的错别字、补充省略的关键词。

### 6. 路由 (Routing) 解决什么问题？
**场景**: 你有多个知识库（如技术文档、产品说明、FAQ），如何决定去哪个库检索？  
**方案**:
- **逻辑路由**: 用规则判断（如问题包含"价格"→产品库）
- **语义路由**: 用 LLM 判断问题属于哪个主题  
**类比**: 图书馆的分馆系统，根据你的需求引导你去正确的分馆。

### 7. 重排序 (Re-ranking) 为什么必要？
**问题**: 向量检索速度快但精度有限，可能把相关但不精准的文档排前面。  
**解决**: 检索后用更强的交叉编码器 (Cross-Encoder) 重新打分排序。  
**类比**: 初筛（简历筛选）→ 精筛（面试），先用快速方法缩小范围，再用精准方法排序。

---

## 常见疑问解答

### Q1: 必须用 OpenAI 的 Embedding 和 GPT 吗？
**答**: 不是。项目使用 OpenAI 是为了演示简洁。你可以替换为：
- Embedding: `HuggingFaceEmbeddings`、`CohereEmbeddings`、本地 `SentenceTransformer`
- LLM: `ChatOllama`（本地部署）、`ChatAnthropic`、`ChatZhipuAI`  
只需替换对应的类，接口保持一致。

### Q2: 向量数据库 Chroma 可以换成其他吗？
**答**: 完全可以。LangChain 支持多种向量数据库：
- `FAISS`: Facebook 出品，速度快，适合本地
- `Pinecone`: 云服务，适合生产环境
- `Milvus`: 开源，支持大规模数据
- `Weaviate`: 支持混合搜索（向量 + 关键词）  
替换方式：`Chroma.from_documents()` → `FAISS.from_documents()` 等。

### Q3: `chunk_size` 和 `chunk_overlap` 怎么设置？
**答**: 没有标准答案，取决于你的文档类型：
- **短文档**（如 FAQ）: `chunk_size=500`, `chunk_overlap=50`
- **长文档**（如技术手册）: `chunk_size=1000-2000`, `chunk_overlap=200`
- **代码文档**: `chunk_size=1500`, `chunk_overlap=100`  
**调优方法**: 先用默认值，根据检索效果调整。如果检索结果总是"差一点"，尝试增大 `chunk_overlap`。

### Q4: 学完这个项目能做什么？
**答**: 你将具备以下能力：
- ✅ 从零构建企业级 RAG 系统（如智能客服、知识库问答）
- ✅ 根据场景选择合适的检索策略（简单检索 vs 多查询 vs 重排序）
- ✅ 调试和优化 RAG 效果（分析检索失败原因、调整切分策略）
- ✅ 阅读和理解更复杂的 RAG 项目源码

### Q5: 这个项目和我在 05-工具调用与RAG/ 中学的 RAG 有什么区别？
**答**: 
- **05 目录的文档**: 偏理论和架构讲解，适合建立整体认知
- **这个项目**: 偏实战和代码演练，适合动手实践  
**建议学习路径**: 先读 `05-工具调用与RAG/01-RAG基础架构与实现.md` 建立概念，再用本项目动手验证。

---

## 学习建议

1. **按顺序执行**: 不要跳过前面的 Notebook，后面的技术建立在前面的基础上
2. **边看视频边练**: 配合 [官方视频列表](https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZytA0x) 学习
3. **修改参数实验**: 尝试修改 `chunk_size`、替换模型、调整 prompt，观察效果变化
4. **用 LangSmith 调试**: 项目集成了 LangSmith 追踪，可以看到每一步的输入输出
5. **延伸思考**: 学完每个 Part 后，思考"这个技术在我的业务场景中怎么用？"

**学习路线图**:
```
基础 RAG (Part 1-4)
    ↓
查询优化 (Part 5-9)  ← 掌握后能应对 80% 的场景
    ↓
高级路由 (Part 10-11)
    ↓
索引优化 (Part 12-14) ← 适合有性能/精度要求的场景
    ↓
前沿技术 (Part 15-18) ← 了解最新研究进展
```

祝学习愉快！🚀
