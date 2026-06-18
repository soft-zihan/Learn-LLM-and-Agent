# Smolagents - HuggingFace 代码智能体框架学习指南

## 项目概述

**一句话总结**: Smolagents 是 HuggingFace 出品的轻量级 agent 框架，主打"Agents that think in code"，让 agent 用 Python 代码编写行动，核心逻辑仅 ~1000 行代码。

**核心亮点**:
- **极简设计**: 核心 agent 逻辑仅 ~1000 行代码（agents.py），保持最小抽象
- **Code Agent 优先**: [`CodeAgent`](https://huggingface.co/docs/smolagents/reference/agents#smolagents.CodeAgent) 将行动写为 Python 代码片段，比 JSON 工具调用性能更高（少 30% 步骤）
- **Hub 集成**: 可直接从 HuggingFace Hub 分享/拉取 tools 或 agents， instant sharing
- **模型无关**: 支持任何 LLM（transformers、ollama、OpenAI、Anthropic、LiteLLM 等 100+ 提供商）
- **多模态**: 支持文本、视觉、视频、音频输入
- **工具无关**: 支持 MCP Server、LangChain tools、Hub Spaces 等任意工具源
- **沙箱执行**: 支持 E2B、Blaxel、Modal、Docker 等沙箱环境

**适合谁学**:
- 想理解 Code Agent 原理的开发者（agent 用代码编写行动）
- 喜欢轻量级框架，不喜欢重型抽象的 Python 开发者
- 想快速集成 HuggingFace Hub 生态的 AI 应用开发者

---

## 核心架构解析

### Code Agent vs ToolCalling Agent

```python
# CodeAgent：行动是 Python 代码片段
code_agent_action = """
# 搜索多个关键词
requests_to_search = ["gulf of mexico america", "greenland denmark", "tariffs"]
for request in requests_to_search:
    print(f"Here are the search results for {request}:", web_search(request))
"""

# ToolCallingAgent：行动是 JSON 字典
tool_calling_action = {
    "tool": "web_search",
    "parameters": {"query": "gulf of mexico america"}
}
```

**为什么 Code Agent 更好**:
- **更高效**: 少 30% 步骤（ thus 少 30% LLM 调用）
- **更强大**: 在困难基准测试上表现更好
- **更灵活**: 可在一个行动中执行复杂逻辑（循环、条件、多工具调用）

---

## 代码逻辑主线

### 1. CodeAgent 核心循环

```python
# CodeAgent ReAct 循环
class CodeAgent:
    def run(self, task):
        """
        CodeAgent 工作流程
        
        1. 将任务加入 memory
        2. ReAct 循环:
           - 从 memory 构建上下文
           - 调用模型生成代码行动
           - 解析并执行代码
           - 将执行结果加入 memory
           - 重复直到调用 'final_answer' 工具
        3. 返回最终答案
        """
        # 添加任务到 memory
        self.memory.add_task(task)
        
        while True:
            # 从 memory 构建上下文
            context = self.build_context()
            
            # 调用模型生成代码
            code_action = self.model.generate(context)
            
            # 执行代码
            result = self.execute_code(code_action)
            
            # 检查是否完成
            if "final_answer" in code_action:
                return extract_final_answer(result)
            
            # 将执行结果加入 memory
            self.memory.add_execution_log(code_action, result)
```

**关键设计**:
- **代码执行**: 行动是 Python 代码，工具调用是函数调用
- **格式一致**: 系统提示词、解析器、执行器保持代码格式一致
- **安全沙箱**: 代码在沙箱中执行（E2B、Blaxel、Modal、Docker）

---

### 2. 工具系统

#### 从 Hub 拉取工具

```python
from smolagents import Tool

# 从 Hub 拉取工具
tool = Tool.from_hub("m-ric/my-tool")

# 使用工具
result = tool(param1="value1")
```

#### 从 MCP Server 拉取工具

```python
from smolagents import ToolCollection

# 从 MCP Server 拉取工具集合
tools = ToolCollection.from_mcp("http://localhost:8080/mcp")

# 使用工具
agent = CodeAgent(tools=tools, model=model)
```

#### 从 LangChain 转换工具

```python
from smolagents import Tool

# 从 LangChain 工具转换
tool = Tool.from_langchain(langchain_tool)

# 使用工具
result = tool(query="search term")
```

#### 从 Hub Space 创建工具

```python
from smolagents import Tool

# 从 Hub Space 创建工具（如图像生成 Space）
tool = Tool.from_space("huggingface/diffusers-webui")

# 使用工具生成图像
image = tool(prompt="A cat wearing a hat")
```

**为什么这样设计**: 工具来源无关性，让你能用任何工具源的 tools，不被框架锁定。

---

### 3. 模型集成

#### InferenceClientModel（HuggingFace 推理）

```python
from smolagents import InferenceClientModel

# 使用 HuggingFace 推理提供商
model = InferenceClientModel(
    model_id="deepseek-ai/DeepSeek-R1",
    provider="together",  # 或其他提供商
)

agent = CodeAgent(tools=[WebSearchTool()], model=model)
agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
```

#### LiteLLM（100+ LLMs）

```python
from smolagents import LiteLLMModel

# 使用 LiteLLM 接入 100+ LLMs
model = LiteLLMModel(
    model_id="anthropic/claude-4-sonnet-latest",
    temperature=0.2,
    api_key=os.environ["ANTHROPIC_API_KEY"]
)
```

#### OpenAI 兼容服务器

```python
from smolagents import OpenAIModel

# Together AI
model = OpenAIModel(
    model_id="deepseek-ai/DeepSeek-R1",
    api_base="https://api.together.xyz/v1/",
    api_key=os.environ["TOGETHER_API_KEY"],
)

# OpenRouter
model = OpenAIModel(
    model_id="openai/gpt-4o",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

#### 本地 transformers 模型

```python
from smolagents import TransformersModel

# 使用本地模型
model = TransformersModel(
    model_id="Qwen/Qwen3-Next-80B-A3B-Thinking",
    max_new_tokens=4096,
    device_map="auto"
)
```

---

### 4. CLI 命令

```bash
# 运行 CodeAgent（直接 prompt + 选项）
smolagent "Plan a trip to Tokyo, Kyoto and Osaka between Mar 28 and Apr 7." \
  --model-type "InferenceClientModel" \
  --model-id "Qwen/Qwen3-Next-80B-A3B-Thinking" \
  --imports pandas numpy \
  --tools web_search

# 运行交互式模式（启动设置向导）
smolagent

# 运行 Web Agent（基于 helium 的浏览器自动化）
webagent "go to xyz.com/men, get to sale section, click the first clothing item you see. Get the product details, and the price, return them. note that I'm shopping from France" \
  --model-type "LiteLLMModel" \
  --model-id "gpt-5"
```

**交互式模式引导**:
- Agent 类型选择（CodeAgent vs ToolCallingAgent）
- 工具选择（从可用工具箱中选择）
- 模型配置（类型、ID、API 设置）
- 高级选项（额外导入）
- 任务 prompt 输入

---

## 快速上手实践

### 环境配置步骤

```bash
# 1. 安装 smolagents（带工具包）
pip install "smolagents[toolkit]"

# 2. 配置 API Key（根据你使用的模型）
export HF_TOKEN="your-huggingface-token"
# 或
export OPENAI_API_KEY="your-openai-key"
# 或
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### 运行第一个示例

```python
from smolagents import CodeAgent, WebSearchTool, InferenceClientModel

# 创建模型
model = InferenceClientModel()

# 创建 agent（带搜索工具）
agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    stream_outputs=True  # 流式输出
)

# 运行 agent
result = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")

print(result)
```

### 分享到 Hub

```python
# 分享 agent 到 Hub
agent.push_to_hub("m-ric/my_agent")

# 从 Hub 加载 agent
agent = CodeAgent.from_hub("m-ric/my_agent")
```

### 预期输出

```
# Agent 执行的代码示例:
distance = 100  # Pont des Arts 长度约 100 米
leopard_speed = 60  # 猎豹最高速度约 60 km/h = 16.67 m/s
time = distance / leopard_speed
final_answer(time)

# 输出:
6.0  # 约 6 秒
```

---

## 核心知识点总结

### 必须掌握的 8 个概念

1. **CodeAgent**: agent 将行动写为 Python 代码片段，比 JSON 工具调用更高效
2. **ToolCallingAgent**: 传统方式，行动写为 JSON/text blobs
3. **工具来源无关性**: 支持 MCP Server、LangChain、Hub Spaces、自定义函数
4. **模型无关性**: 支持任何 LLM（transformers、ollama、OpenAI、Anthropic、LiteLLM）
5. **沙箱执行**: 代码在沙箱中执行（E2B、Blaxel、Modal、Docker），不是 LocalPythonExecutor
6. **Hub 集成**: 可从 Hub 分享/拉取 tools 或 agents
7. **多模态支持**: 支持文本、视觉、视频、音频输入
8. **CLI 命令**: `smolagent`（通用 agent）和 `webagent`（浏览器自动化）

---

## 常见疑问解答

### Q1: CodeAgent 为什么比 ToolCallingAgent 更好？

**性能优势**:
- **少 30% 步骤**: 代码可在一行中调用多个工具，JSON 每次只能调一个
- **更高性能**: 在困难基准测试上表现更好
- **更灵活**: 可在代码中使用循环、条件、变量等 Python 特性

**示例对比**:
```python
# CodeAgent: 一个行动中搜索多个关键词
for keyword in ["key1", "key2", "key3"]:
    print(web_search(keyword))

# ToolCallingAgent: 需要三个独立行动
{"tool": "web_search", "parameters": {"query": "key1"}}
{"tool": "web_search", "parameters": {"query": "key2"}}
{"tool": "web_search", "parameters": {"query": "key3"}}
```

### Q2: LocalPythonExecutor 安全吗？

**不安全**。LocalPythonExecutor 仅提供最佳努力的限制措施，**不是安全边界**。

**正确做法**:
- **生产环境**: 使用沙箱（E2B、Blaxel、Modal、Docker）
- **开发调试**: 可使用 LocalPythonExecutor，但不要执行不受信任的代码

### Q3: 如何选择沙箱提供商？

| 提供商 | 特点 | 适用场景 |
|--------|------|---------|
| **E2B** | 托管云沙箱，设置最简单 | 快速原型、个人项目 |
| **Blaxel** | 托管云沙箱 | 中等规模项目 |
| **Modal** | 托管云沙箱，支持 GPU | 需要 GPU 的任务 |
| **Docker** | 自托管容器隔离 | 生产环境、数据隐私要求高 |

### Q4: 开源模型能用于 Code Agent 吗？

**可以**。DeepSeek-R1 等开源模型在 Code Agent 基准上击败闭源模型。

**推荐模型**:
- **DeepSeek-R1**: 开源推理模型，Code Agent 表现优秀
- **Qwen 系列**: 代码能力强，支持本地部署
- **Llama 3**: 通用能力强，生态好

---

## 延伸学习路径

1. **官方文档**: https://huggingface.co/docs/smolagents/index
2. **概念指南**: https://huggingface.co/docs/smolagents/conceptual_guides/intro_agents
3. **示例代码**: `examples/` 目录
4. **基准测试**: `examples/smolagents_benchmark/run.py`
5. **进阶主题**:
   - 开发自定义工具
   - 从 Hub 分享 agents
   - 多模态 agent（视觉、视频、音频）
   - 沙箱环境配置

---

## 与教程目录的关联

本项目与以下教程内容强关联：

| 教程章节 | 关联点 |
|---------|-------|
| `06-单Agent开发与框架/01-Agent核心概念` | Code Agent 理念、ReAct 循环 |
| `05-工具调用与RAG/01-工具调用基础` | 工具调用、MCP Server 集成 |
| `07-多Agent与Agent工程化/03-MCP协议与Skills` | MCP 工具集成 |

---

## 总结

**Smolagents 不是重型框架，而是极简的 Code Agent 实现**。它展示了：

1. ✅ **极简设计**: 核心逻辑仅 ~1000 行代码，最小抽象
2. ✅ **Code Agent 优先**: 行动写为 Python 代码，性能优于 JSON 工具调用
3. ✅ **生态集成**: HuggingFace Hub、MCP Server、LangChain、多模态
4. ✅ **模型无关**: 支持任何 LLM，100+ 提供商
5. ✅ **沙箱执行**: 安全的代码执行环境

**面试中可以讲的亮点**:
- 我深入学习了 Smolagents，理解 Code Agent 的设计哲学和性能优势
- 我能开发自定义工具，从 MCP Server、LangChain、Hub Spaces 集成工具
- 我理解沙箱执行的重要性，能配置 E2B/Docker 等安全环境
- 我能使用 HuggingFace Hub 分享和拉取 agents，实现 instant sharing

---

**学习建议**: 先用 WebSearchTool 快速体验 Code Agent，然后尝试自定义工具和多模态输入，最后深入理解 agents.py 源码（仅 ~1000 行）。