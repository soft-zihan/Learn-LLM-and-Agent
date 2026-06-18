# LLM & Agent 学习笔记

从预训练到 Agent 工程化的完整学习路线，面向 AI 应用开发者的系统化教程。

## 教程目录

### 01 预训练与架构基础

**01-Transformer与架构演进**
<details>
<summary>- 01-Transformer架构演进</summary>
  - Transformer 架构演进（2025-2026）
  - 大规模数据准备与清洗
  - 分布式训练策略
  - 训练优化技术
  - 实战示例：从头训练 7B 模型
  - 最佳实践
  - 参考资源
  - 词元化技术详解
  - 混合精度训练
  - 训练稳定性技巧
</details>
<details>
<summary>- 02-混合架构与SSM</summary>
  - Transformer 的局限性
  - SSM 状态空间模型
  - Mamba 架构
  - Mamba-2 改进
  - RWKV 架构
  - Hybrid 混合架构
  - 性能对比与基准测试
  - 适用场景分析
  - 2025 最新进展
  - 未来展望
  - 参考资料
  - 附录：关键公式
</details>

**02-分布式训练**
<details>
<summary>- 01-训练基础设施与集群</summary>
  - GPU 硬件选择
  - GPU 集群管理与调度
  - 训练框架对比
  - 成本优化与效率提升
  - 训练监控与调试
  - 最佳实践
  - 参考资源
</details>

**03-预训练数据与流程**
<details>
<summary>- 01-数据工程与预训练实践</summary>
  - 预训练数据收集
  - 数据清洗与去重
  - 数据质量控制
  - 预训练流程
  - 数据配比策略
  - 合成数据技术
  - 实战案例与最佳实践
  - 参考资料
</details>

### 02 微调与训练工程

**01-微调技术**
<details>
<summary>- 01-全量微调与SFT</summary>
  - 微调技术概览
  - 全量微调（Full Fine-tuning）
  - 参数高效微调（PEFT）
  - LoRA 技术详解
  - QLoRA 技术详解
  - 指令微调（Instruction Tuning）
  - 多模态微调技术
  - 实战：完整微调工作流
  - 最佳实践
  - 参考资源
  - Prompt Tuning 系列
</details>

**02-垂域与专项训练**
<details>
<summary>- 01-垂域模型训练实战</summary>
  - 垂域模型训练概述
  - 医疗领域模型训练（概述）
  - 法律领域模型训练（概述）
  - 金融领域模型训练（概述）
  - 代码领域模型训练（概述）
  - 垂域训练通用方法论
  - 最佳实践与常见陷阱
  - 参考资料
</details>

**03-训练框架与工具**
<details>
<summary>- 01-开源训练框架实战</summary>
  - 训练框架概览与选型
  - Axolotl 实战
  - LLaMA-Factory 实战
  - OpenRLHF 实战
  - veRL (verl) 实战
  - Unsloth 实战
  - 框架对比与选型决策
  - 参考资料
</details>
<details>
<summary>- 02-训练数据集资源大全</summary>
  - 数据集概览与分类
  - 预训练数据集
  - 指令微调数据集
  - 偏好对齐数据集
  - 领域专用数据集
  - 合成数据集
  - 推理数据集（2025-2026 新增）
  - 多模态数据集
  - 数据集质量评估
  - 数据集选择指南
  - 数据处理工具与框架
  - 参考资料
</details>

### 03 对齐与安全

**01-对齐技术**
<details>
<summary>- 01-RLHF与PPO实战</summary>
  - 为什么需要对齐？
  - RLHF（Reinforcement Learning from Human Feedback）
  - DPO（Direct Preference Optimization）
  - ORPO（Odds Ratio Preference Optimization）
  - SimPO（Simple Preference Optimization）
  - 其他对齐方法
  - 安全对齐与红队测试
  - 对齐方法选择指南
  - 实战：完整对齐流程
  - 最佳实践
  - 参考资源
</details>

**02-后训练工程**
<details>
<summary>- 01-RLVR与GRPO工程实践</summary>
  - 后训练技术概览
  - RLHF 工程实践
  - RLVR（可验证奖励强化学习）
  - GRPO 与 RE-PO
  - Agentic RL
  - 自动化红队测试
  - 大规模对齐训练
  - 对齐失败案例分析
  - 最佳实践
  - 参考资料
</details>

**03-安全工程**
<details>
<summary>- 01-安全基础与红队测试</summary>
  - LLM 安全基础概念
  - 红队测试(Red Teaming)
  - Constitutional AI 技术
  - 企业安全实践
  - 总结
</details>
<details>
<summary>- 02-安全防护与合规评估</summary>
  - 输入安全防护
  - 输出安全防护
  - 偏见与公平性
  - 隐私保护
  - 安全评估体系
  - 生产安全架构
  - 前沿技术
  - 实战项目
  - 参考文献与资源
  - 总结
</details>

### 04 推理与部署

**01-推理引擎与优化**
<details>
<summary>- 01-推理引擎与KV-Cache</summary>
  - 前言
  - 推理基础概念
  - KV Cache 技术
  - vLLM 高吞吐推理
  - TensorRT-LLM 优化
  - Ollama 与 llama.cpp
  - SGLang 结构化生成
  - 实战场景与最佳实践
  - LLM推理优化检查清单
  - 2025年最新进展
  - 总结与展望
  - 参考资料
</details>

**02-生产部署**
<details>
<summary>- 01-模型服务化部署</summary>
  - API 设计最佳实践
  - 流式输出实现
  - 负载均衡策略
  - 自动扩缩容
  - 缓存策略
  - 成本优化
  - 监控与告警
  - 参考资料
</details>
<details>
<summary>- 02-FastAPI与Docker部署实战</summary>
  - 引言：为什么选择 FastAPI + Docker 部署 LLM 应用
  - FastAPI 基础
  - LLM 应用架构设计
  - Docker 容器化
  - 生产级最佳实践
  - 完整实战案例
  - 部署到云服务器（可选）
  - 参考资料
  - 总结
</details>

**03-Java生态**
<details>
<summary>- 01-Spring-AI-2.0开发实战</summary>
  - Spring AI 概述
  - 快速入门
  - 核心概念
  - 工具调用与 Function Calling
  - RAG 实现
  - Agent 开发
  - 生产级应用开发
  - 最佳实践
  - 参考资料
</details>

### 06 单 Agent 开发与框架

**01-Agent基础与概念**
<details>
<summary>- 01-AI-Agent基础与架构</summary>
  - Agent 基础概念
  - 单 Agent 工作流
  - LangChain Agent 开发
  - LangGraph 单 Agent 状态图
  - 规划与推理
  - 规划与推理
</details>
<details>
<summary>- 02-Agent设计模式</summary>
  - 记忆系统设计
  - 工具使用深度实践
  - 规划与推理
</details>
<details>
<summary>- 02-上下文工程与记忆系统</summary>
  - 什么是上下文工程
  - 为什么需要上下文工程
  - 上下文的解剖学
  - GSSC 流程
  - JIT 上下文检索
  - 长期任务的上下文管理
  - 上下文工程实战
  - 最佳实践与常见陷阱
  - 总结
  - 为什么 Agent 需要记忆
  - 从人类记忆到 Agent 记忆
  - 记忆系统架构设计
  - 2026年主流记忆方案对比
  - 实战：构建记忆系统
  - 记忆与 MCP 集成
  - 记忆系统最佳实践与常见陷阱
  - 总结
</details>

**02-Agent工具与协议**
<details>
<summary>- 01-工具调用与集成实战</summary>
  - 工具调用基础概念
  - OpenAI Function Calling
  - Anthropic Tool Use
  - Google Gemini Function Calling
  - 开源模型工具调用
  - LangChain 工具系统
  - 高级工具调用技术
  - 生产级工具调用
  - MCP（Model Context Protocol）
  - 工具调用评测与基准
  - 参考资料
  - 工具调用基础
  - 工具实现模式
  - 工具调用流程
  - 错误处理与重试
  - 工具编排与组合
  - 工具测试与调试
  - 性能优化
  - 参考资料
</details>
<details>
<summary>- 02-MCP协议详解与Server开发</summary>
  - MCP 协议基础概念
  - MCP 2026-07-28 规范重大更新
  - MCP 协议详解
  - MCP Server 开发（Python）
  - MCP Server 开发（TypeScript）
  - MCP Server 开发（Go）
  - MCP Client 集成
  - MCP 2026-07-28 新特性速查
  - 更多实战场景
  - 生态工具
  - 前沿技术
  - 参考资料
  - MCP 2026-07-28 安全与生产部署更新
</details>
<details>
<summary>- 03-Skills范式与实战</summary>
  - Skills 开发最佳实践
  - Inputs
  - Decision Tree
  - Main Workflow
  - Output Files
  - Non-Negotiables
  - 安全与权限控制
  - 生产部署
</details>

**02-Agent核心组件**
<details>
<summary>- 01-工具调用基础与主流API</summary>
  - 工具调用基础概念
  - OpenAI Function Calling
  - Anthropic Tool Use
  - Google Gemini Function Calling
  - 开源模型工具调用
  - LangChain 工具系统
  - 高级工具调用技术
  - 生产级工具调用
  - MCP（Model Context Protocol）
  - 工具调用评测与基准
  - 参考资料
</details>
<details>
<summary>- 02-上下文工程</summary>
  - 什么是上下文工程
  - 为什么需要上下文工程
  - 上下文的解剖学
  - GSSC 流程
  - JIT 上下文检索
  - 长期任务的上下文管理
  - 上下文工程实战
  - 最佳实践与常见陷阱
  - 总结
</details>
<details>
<summary>- 05-Memory</summary>
  - 为什么 Agent 需要记忆
  - 从人类记忆到 Agent 记忆
  - 记忆系统架构设计
  - 2026年主流记忆方案对比
  - 实战：构建记忆系统
  - 记忆与 MCP 集成
  - 最佳实践与常见陷阱
  - 总结
</details>

**03-Agent框架与编排**
<details>
<summary>- 01-LangChain与LangGraph实践</summary>
  - LangChain v1.0 核心变化与概念
  - LangGraph v1.0 状态图与工作流编排
  - 多 Agent 协作模式
  - 最新特性与最佳实践（2025-2026）
  - 实战案例：智能代码审查 Agent
  - 参考资源
</details>
<details>
<summary>- 02-RAG进阶与Agentic-RAG</summary>
  - RAG 技术演进
  - 文档处理与分割
  - 向量数据库选型
  - 高级检索策略
  - 查询优化
  - Agentic RAG
  - GraphRAG
  - RAG 评估
  - 实战案例
  - 最佳实践与陷阱
  - 参考资料
</details>

**03-Agent运行模式**
<details>
<summary>- 01-Agent-Loop运行模式</summary>
  - 从 Prompt 到 Loop 的演进
  - Agent Loop 的核心组件
  - 记忆系统设计
  - 项目约定
  - 构建步骤
  - 历史教训
  - 当前状态
  - 子 Agent 架构
  - Loop 实战案例
  - Token 成本优化
  - Loop 的风险与应对
  - 总结：构建 Loop，但保持你是工程师
  - 参考链接
</details>
<details>
<summary>- 02-工作流编排与LangGraph</summary>
  - 评估与测试
  - 总览
  - 按分类
  - 生产级最佳实践
  - 实战项目
  - 从单 Agent 到多 Agent
  - 单 Agent 的局限性
  - 何时需要使用多 Agent 架构
  - 单 Agent 到多 Agent 迁移 Checklist
  - 总结
</details>

**04-Agent运行模式与工程化**
<details>
<summary>- 01-Harness-Engineering</summary>
  - Harness Engineering 基础
  - 核心组件
  - AGENTS.md 与项目配置实战
  - Minimal Harness 实战
  - Superpowers 框架深度解析
  - 高级 Harness 模式
  - 验证与测试策略
  - 可观测性与调试
  - 生产级 Harness 设计
  - 实战项目
  - 项目概述
  - 技术栈
</details>
<details>
<summary>- 02-Loop-Engineering</summary>
  - 从 Harness 到 Loop 的演进
  - Loop Engineering 五大核心组件
  - 生成器-评估器-规划器架构
  - Loop 实战模式
  - Loop 的风险与治理
  - Token 成本优化实战
  - 最佳实践
  - 总结：构建 Loop，但保持你是工程师
  - 参考链接
</details>

**05-Agent工程化**
<details>
<summary>- 01-Harness-Engineering基础</summary>
  - Harness Engineering 基础
  - 核心组件
  - AGENTS.md 与项目配置实战
  - Minimal Harness 实战
  - Superpowers 框架深度解析
  - 高级 Harness 模式
</details>
<details>
<summary>- 02-Harness高级与生产实战</summary>
  - 验证与测试策略
  - 可观测性与调试
  - 生产级 Harness 设计
  - 实战项目
  - 项目概述
  - 技术栈
</details>
<details>
<summary>- 03-Loop-Engineering进阶</summary>
  - 从 Harness 到 Loop 的演进
  - Loop Engineering 五大核心组件
  - 生成器-评估器-规划器架构
  - Loop 实战模式
  - Loop 的风险与治理
  - Token 成本优化实战
  - 最佳实践
  - 参考链接
</details>

**05-生产部署与可观测性**
<details>
<summary>- 01-单Agent生产级最佳实践</summary>
  - 生产环境架构设计
  - 错误处理与恢复
  - 性能优化
  - 安全最佳实践
  - 部署与运维
  - 监控与告警
  - 测试策略
  - 参考资料
</details>
<details>
<summary>- 02-Agent测试与可观测性工程</summary>
  - Agent 测试策略
  - 可观测性工程
  - 性能测试
  - 安全测试
  - 监控仪表板
  - 告警规则
  - 参考资料
</details>

**06-工具集成与实战**
<details>
<summary>- 01-Agent工具集成实战</summary>
  - 工具调用基础
  - 工具实现模式
  - 工具调用流程
  - 错误处理与重试
  - 工具编排与组合
  - 工具测试与调试
  - 性能优化
  - 参考资料
</details>

### 07 多 Agent 与 Agent 工程化

**01-多Agent系统**
<details>
<summary>- 01-多Agent架构与协作模式</summary>
  - 多 Agent 系统概述
  - CrewAI 实战
  - AutoGen/AG2 实战
  - OpenAI Agents SDK 实战
  - Google ADK 实战
  - 多 Agent 通信与协调
  - Agent 编排模式
  - 最佳实践与陷阱
  - 参考资料
  - 工具调用深度实践
  - Agent 评估与基准测试
  - 解决方案
  - 修改的文件
  - 测试
</details>

**02-Agent编排与通信**
<details>
<summary>- 01-高级编排与Agent通信协议</summary>
  - Agent 编排模式
  - Agent 通信协议
  - 高级通信模式
  - 容错与恢复
  - 监控与可观测性
  - 性能优化
  - 参考资料
</details>

### 08 多模态与前沿技术

**01-多模态大模型**
<details>
<summary>- 01-VLM架构与多模态预训练</summary>
  - 多模态基础概念
  - VLM 架构详解
  - 多模态预训练
  - 多模态能力
  - 实践与部署
  - 前沿研究
  - 参考资料
  - 附录
</details>
<details>
<summary>- 02-多模态微调与部署评估</summary>
  - 多模态推理优化
  - 多模态评估体系
  - 生产部署实践
  - 实战项目案例
  - 附录与
  - 总结
</details>

**02-MoE架构**
<details>
<summary>- 01-MoE原理与模型解析</summary>
  - MoE 基础概念
  - MoE 架构原理
</details>
<details>
<summary>- 02-MoE训练微调与推理优化</summary>
  - Mixtral 8x7B 深度解析
  - DeepSeek-V3 架构详解
  - MoE 训练实战
  - MoE 推理优化
  - MoE 微调技术
  - MoE 评估与监控
  - 前沿技术与发展方向
  - 实战应用方向
  - 参考文献
  - 附录
</details>

**03-推理模型与思维链**
<details>
<summary>- 01-推理模型基础与CoT技术</summary>
  - 推理模型基础概念
  - 思维链 CoT 技术
  - 已知条件
  - 推理过程
  - 验证
  - 答案
  - 已知条件
  - 推理过程
  - OpenAI o1 系列深度解析
  - 分析
  - 解法1
  - 解法2（如果有）
  - 验证
  - 答案
</details>
<details>
<summary>- 02-RL-for-Reasoning与训练实战</summary>
  - 主流推理模型对比
  - 思维链工程实践
  - 前沿技术趋势
  - 参考资料
  - 附录:快速参考卡片
</details>

**04-学习路线与前沿展望**
<details>
<summary>- 01-Agent学习路线图入门到进阶</summary>
  - 学习路径总览
  - 阶段 1：基础入门
  - 阶段 2：单 Agent 开发
  - 阶段 3：高级 Agent 技术
  - 阶段 4：多 Agent 协作
</details>
<details>
<summary>- 02-Agent学习路线图高级与实战项目</summary>
  - 阶段 5：生产部署
  - 阶段 6：前沿技术
  - 各阶段核心知识点清单
  - 推荐资源与教程
  - 实战项目建议
  - 能力评估标准
  - 学习社区与交流
  - 持续学习建议
  - 结语
  - 附录
</details>
<details>
<summary>- 03-前沿技术总结与展望</summary>
  - 导读
  - 2025-2026 技术趋势
  - 世界模型：AI 的物理直觉
  - 具身智能：AI 走进物理世界
  - AGI 路径探讨：我们离通用智能还有多远
  - 开放挑战：技术、安全与社会
  - 学习建议：如何在 AI 时代成长
  - 结语：站在 AI 革命的起点
  - 参考文献与延伸阅读
  - 附录：关键术语表
</details>

## 开源项目学习

本仓库包含 22 个精选开源项目的学习文档。

### 01 大模型训练基础

| 项目 | 说明 |
|------|------|
| [01-llm-action](https://github.com/liguodongiot/llm-action) | 从6B到65B模型全链路实战教程 |
| [02-minimind](https://github.com/jingyaogong/minimind) | 2小时从零训练64M小模型 |
| [03-self-llm](https://github.com/datawhalechina/self-llm) | Datawhale开源大模型食用指南，中文友好 |

### 02 RAG 实战

| 项目 | 说明 |
|------|------|
| [01-agentic-rag-for-dummies](https://github.com/GiovanniPasq/agentic-rag-for-dummies) | 基于LangGraph的Agentic RAG极简入门 |
| [02-production-agentic-rag-course](https://github.com/jamwithai/production-agentic-rag-course) | 7周生产级RAG课程 |
| [03-rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch) | LangChain官方RAG入门教程 |
| [04-rag-tutorial](https://github.com/vivy-yi/rag-tutorial) | 完整RAG教程，4模块20章17个Notebook+6个企业案例 |

### 03 MCP 与 Skills

| 项目 | 说明 |
|------|------|
| [01-LLMInternSkill](https://github.com/couragec/llm-intern-skill) | Codex Skill实战项目 |
| [02-clawkeeper](https://github.com/SafeAI-Lab-X/ClawKeeper) | OpenClaw安全加固项目，1000+Stars |
| [03-mcp-python-sdk](https://github.com/modelcontextprotocol/python-sdk) | MCP协议Python官方实现 |
| [04-mcp-servers](https://github.com/modelcontextprotocol/servers) | MCP官方Server参考实现集合 |
| [05-mcp-typescript-sdk](https://github.com/modelcontextprotocol/typescript-sdk) | MCP协议TypeScript官方实现 |

### 04 Agent 实战

| 项目 | 说明 |
|------|------|
| [01-Agent-Learning-Hub](https://github.com/Agent-Learning-Hub/Agent-Learning-Hub) | Agent学习资源聚合项目，包含完整学习路线图、交互式网页、实战项目推荐 |
| [02-Deer-flow](https://github.com/bytedance/deer-flow) | 字节跳动super agent harness |
| [03-GenAI_Agents](https://github.com/chinobling/GenAI_Agents) | LLM与Agent论文资源库 |
| [04-Hello-agents](https://github.com/datawhalechina/Hello-Agents) | Datawhale社区系统性教程，16章 |
| [05-Ragent](https://github.com/nageoffer/ragent) | 企业级Agentic RAG平台，Spring AI 2.0 |
| [06-Smolagents](https://github.com/huggingface/smolagents) | HuggingFace轻量Code Agent框架 |
| [07-crewai](https://github.com/crewAIInc/crewAI) | 角色驱动的多Agent编排框架 |
| [08-learn-claude-code](https://github.com/jason19990103/learn-claude-code) | Claude Code源码深度解析 |
| [09-nanobot](https://github.com/HKUDS/nanobot) | 4000行轻量级Agent框架，简洁优雅的架构设计 |
| [10-openai-agents-sdk](https://github.com/openai/openai-agents-python) | OpenAI官方Agent SDK |

## 自动更新目录

```bash
python3 update-readme
```

**统计**：48 个教程文档


