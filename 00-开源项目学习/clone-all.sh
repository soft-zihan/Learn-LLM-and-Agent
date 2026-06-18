#!/bin/bash

# ===========================================
# 99-开源项目学习 clone 脚本
# 面向初学者的完整教学仓库
# ===========================================

# 大模型训练基础 (3 个) - 从零理解训练与微调
mkdir -p 01-大模型训练基础 && cd 01-大模型训练基础
[ ! -d "self-llm" ] && git clone --depth 1 https://github.com/datawhalechina/self-llm.git self-llm || echo "self-llm 已存在"
[ ! -d "llm-action" ] && git clone --depth 1 https://github.com/liguodongiot/llm-action.git llm-action || echo "llm-action 已存在"
[ ! -d "minimind" ] && git clone --depth 1 https://github.com/jingyaogong/minimind.git minimind || echo "minimind 已存在"
cd ..

# RAG 实战 (4 个) - 从基础到生产
mkdir -p 02-RAG实战 && cd 02-RAG实战
[ ! -d "rag-from-scratch" ] && git clone --depth 1 https://github.com/langchain-ai/rag-from-scratch.git rag-from-scratch || echo "rag-from-scratch 已存在"
[ ! -d "rag-tutorial" ] && git clone --depth 1 https://github.com/vivy-yi/rag-tutorial.git rag-tutorial || echo "rag-tutorial 已存在"
[ ! -d "agentic-rag-for-dummies" ] && git clone --depth 1 https://github.com/GiovanniPasq/agentic-rag-for-dummies.git agentic-rag-for-dummies || echo "agentic-rag-for-dummies 已存在"
[ ! -d "production-agentic-rag-course" ] && git clone --depth 1 https://github.com/jamwithai/production-agentic-rag-course.git production-agentic-rag-course || echo "production-agentic-rag-course 已存在"
cd ..

# MCP 与 Skills (5 个) - 协议 + 工具集成 + Skills/Plugins 编写
mkdir -p 03-MCP与Skills && cd 03-MCP与Skills
[ ! -d "mcp-python-sdk" ] && git clone --depth 1 https://github.com/modelcontextprotocol/python-sdk.git mcp-python-sdk || echo "mcp-python-sdk 已存在"
[ ! -d "mcp-typescript-sdk" ] && git clone --depth 1 https://github.com/modelcontextprotocol/typescript-sdk.git mcp-typescript-sdk || echo "mcp-typescript-sdk 已存在"
[ ! -d "mcp-servers" ] && git clone --depth 1 https://github.com/modelcontextprotocol/servers.git mcp-servers || echo "mcp-servers 已存在"
[ ! -d "LLMInternSkill" ] && git clone --depth 1 https://github.com/couragec/llm-intern-skill.git LLMInternSkill || echo "LLMInternSkill 已存在"
[ ! -d "clawkeeper" ] && git clone --depth 1 https://github.com/anthropics/claude-code.git clawkeeper || echo "clawkeeper 已存在"
cd ..

# Agent 实战 (8 个) - 从理论到企业级实战
mkdir -p 04-Agent实战 && cd 04-Agent实战
[ ! -d "GenAI_Agents" ] && git clone --depth 1 https://github.com/chinobling/GenAI_Agents.git GenAI_Agents || echo "GenAI_Agents 已存在"
[ ! -d "Hello-agents" ] && git clone --depth 1 https://github.com/datawhalechina/Hello-Agents.git Hello-agents || echo "Hello-agents 已存在"
[ ! -d "Deer-flow" ] && git clone --depth 1 https://github.com/bytedance/deer-flow.git Deer-flow || echo "Deer-flow 已存在"
[ ! -d "Smolagents" ] && git clone --depth 1 https://github.com/huggingface/smolagents.git Smolagents || echo "Smolagents 已存在"
[ ! -d "Ragent" ] && git clone --depth 1 https://github.com/nageoffer/ragent.git Ragent || echo "Ragent 已存在"
[ ! -d "openai-agents-sdk" ] && git clone --depth 1 https://github.com/openai/openai-agents-python.git openai-agents-sdk || echo "openai-agents-sdk 已存在"
[ ! -d "crewai" ] && git clone --depth 1 https://github.com/crewAIInc/crewAI.git crewai || echo "crewai 已存在"
[ ! -d "learn-claude-code" ] && git clone --depth 1 https://github.com/jason19990103/learn-claude-code.git learn-claude-code || echo "learn-claude-code 已存在"
cd ..

echo "✅ 所有项目 clone 完成！共 20 个项目"
