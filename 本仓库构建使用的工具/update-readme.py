#!/usr/bin/env python3
"""update-readme: 自动更新 README.md"""
import os, re

def extract_all_headings(filepath):
    headings = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = re.findall(r'^##\s+(.+)$', content, re.MULTILINE)
            for match in matches:
                heading = match.strip()
                if '目录' not in heading and 'TOC' not in heading and heading != '':
                    clean = re.sub(r'^\d+\.\s*', '', heading)
                    headings.append(clean)
    except Exception as e:
        print(f"警告: {filepath}: {e}")
    return headings

def scan_tutorials(base_dir):
    structure = []
    total_docs = 0
    tutorial_dirs = [
        ('01-预训练与架构基础', '01 预训练与架构基础'),
        ('02-微调与训练工程', '02 微调与训练工程'),
        ('03-对齐与安全', '03 对齐与安全'),
        ('04-推理与部署', '04 推理与部署'),
        ('05-工具调用与RAG', '05 工具调用与 RAG'),
        ('06-单Agent开发与框架', '06 单 Agent 开发与框架'),
        ('07-多Agent与Agent工程化', '07 多 Agent 与 Agent 工程化'),
        ('08-多模态与前沿技术', '08 多模态与前沿技术'),
    ]
    for dir_name, display_name in tutorial_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path): continue
        subdirs = sorted([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))])
        if not subdirs: continue
        structure.append(f"### {display_name}\n")
        for subdir in subdirs:
            subdir_path = os.path.join(dir_path, subdir)
            md_files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.md') and not f.startswith('.')])
            if not md_files: continue
            structure.append(f"**{subdir}**")
            for md_file in md_files:
                doc_name = md_file.replace('.md', '')
                filepath = os.path.join(subdir_path, md_file)
                headings = extract_all_headings(filepath)
                if headings:
                    # 只有文档级别使用折叠
                    structure.append(f"<details>")
                    structure.append(f"<summary>- {doc_name}</summary>")
                    # 使用无序列表
                    for heading in headings:
                        structure.append(f"  - {heading}")
                    structure.append(f"</details>")
                    total_docs += 1
                else:
                    structure.append(f"- {doc_name}")
                    total_docs += 1
            structure.append("")
    return '\n'.join(structure), total_docs

def scan_open_source_projects(base_dir):
    projects_dir = os.path.join(base_dir, '00-开源项目学习')
    if not os.path.exists(projects_dir): return ""
    sections = []
    project_dirs = [
        ('01-大模型训练基础', '01 大模型训练基础'),
        ('02-RAG实战', '02 RAG 实战'),
        ('03-MCP与Skills', '03 MCP 与 Skills'),
        ('04-Agent实战', '04 Agent 实战'),
    ]
    github_links = {
        'Agent-Learning-Hub': 'https://github.com/Agent-Learning-Hub/Agent-Learning-Hub',
        'nanobot': 'https://github.com/HKUDS/nanobot',
        'self-llm': 'https://github.com/datawhalechina/self-llm',
        'llm-action': 'https://github.com/liguodongiot/llm-action',
        'minimind': 'https://github.com/jingyaogong/minimind',
        'rag-from-scratch': 'https://github.com/langchain-ai/rag-from-scratch',
        'rag-tutorial': 'https://github.com/vivy-yi/rag-tutorial',
        'agentic-rag-for-dummies': 'https://github.com/GiovanniPasq/agentic-rag-for-dummies',
        'production-agentic-rag-course': 'https://github.com/jamwithai/production-agentic-rag-course',
        'mcp-python-sdk': 'https://github.com/modelcontextprotocol/python-sdk',
        'mcp-typescript-sdk': 'https://github.com/modelcontextprotocol/typescript-sdk',
        'mcp-servers': 'https://github.com/modelcontextprotocol/servers',
        'LLMInternSkill': 'https://github.com/couragec/llm-intern-skill',
        'clawkeeper': 'https://github.com/SafeAI-Lab-X/ClawKeeper',
        'GenAI_Agents': 'https://github.com/chinobling/GenAI_Agents',
        'Hello-agents': 'https://github.com/datawhalechina/Hello-Agents',
        'Deer-flow': 'https://github.com/bytedance/deer-flow',
        'Smolagents': 'https://github.com/huggingface/smolagents',
        'Ragent': 'https://github.com/nageoffer/ragent',
        'openai-agents-sdk': 'https://github.com/openai/openai-agents-python',
        'crewai': 'https://github.com/crewAIInc/crewAI',
        'learn-claude-code': 'https://github.com/jason19990103/learn-claude-code',
        'ai-agents-from-zero': 'https://github.com/jackzhenguo/ai-agents-from-zero',
    }
    descriptions = {
        'Agent-Learning-Hub': 'Agent学习资源聚合项目，包含完整学习路线图、交互式网页、实战项目推荐',
        'nanobot': '4000行轻量级Agent框架，简洁优雅的架构设计',
        'self-llm': 'Datawhale开源大模型食用指南，中文友好',
        'llm-action': '从6B到65B模型全链路实战教程',
        'minimind': '2小时从零训练64M小模型',
        'rag-from-scratch': 'LangChain官方RAG入门教程',
        'rag-tutorial': '完整RAG教程，4模块20章17个Notebook+6个企业案例',
        'agentic-rag-for-dummies': '基于LangGraph的Agentic RAG极简入门',
        'production-agentic-rag-course': '7周生产级RAG课程',
        'mcp-python-sdk': 'MCP协议Python官方实现',
        'mcp-typescript-sdk': 'MCP协议TypeScript官方实现',
        'mcp-servers': 'MCP官方Server参考实现集合',
        'LLMInternSkill': 'Codex Skill实战项目',
        'clawkeeper': 'OpenClaw安全加固项目，1000+Stars',
        'GenAI_Agents': 'LLM与Agent论文资源库',
        'Hello-agents': 'Datawhale社区系统性教程，16章',
        'Deer-flow': '字节跳动super agent harness',
        'Smolagents': 'HuggingFace轻量Code Agent框架',
        'Ragent': '企业级Agentic RAG平台，Spring AI 2.0',
        'openai-agents-sdk': 'OpenAI官方Agent SDK',
        'crewai': '角色驱动的多Agent编排框架',
        'learn-claude-code': 'Claude Code源码深度解析',
        'ai-agents-from-zero': 'AI智能体与大模型应用开发从零开始，28章系统教程+2个实战项目',
    }
    for dir_name, display_name in project_dirs:
        dir_path = os.path.join(projects_dir, dir_name)
        if not os.path.exists(dir_path): continue
        md_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.md') and not f.startswith('.')])
        if not md_files: continue
        sections.append(f"### {display_name}\n")
        sections.append("| 项目 | 说明 |")
        sections.append("|------|------|")
        for md_file in md_files:
            project_name = md_file.replace('.md', '')
            clean_name = re.sub(r'^\d+-', '', project_name)
            github_url = github_links.get(clean_name, github_links.get(project_name, ''))
            description = descriptions.get(clean_name, descriptions.get(project_name, ''))
            if github_url:
                sections.append(f"| [{project_name}]({github_url}) | {description} |")
            else:
                sections.append(f"| {project_name} | {description} |")
        sections.append("")
    return '\n'.join(sections)

def update_readme(base_dir):
    readme_path = os.path.join(base_dir, 'README.md')
    if not os.path.exists(readme_path):
        print("错误: README.md 不存在")
        return False
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    tutorial_toc, total_docs = scan_tutorials(base_dir)
    projects_toc = scan_open_source_projects(base_dir)
    header_match = re.search(r'^(# LLM & Agent 学习笔记.*?)\n\n## 教程目录', content, re.DOTALL)
    if not header_match:
        print("错误: 无法找到标题")
        return False
    header = header_match.group(1)
    license_match = re.search(r'(## 许可证.*?)$', content, re.DOTALL)
    license_section = license_match.group(1) if license_match else ""
    new_readme = f"""{header}

## 教程目录

{tutorial_toc}
## 开源项目学习

本仓库包含 22 个精选开源项目的学习文档。

{projects_toc}
## 自动更新目录

```bash
python3 update-readme
```

**统计**：{total_docs} 个教程文档

{license_section}
"""
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_readme)
    print(f"✅ README.md 已更新")
    print(f"📊 教程文档: {total_docs} 个")
    return True

if __name__ == '__main__':
    # 脚本在子目录中，需要回到项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # 向上一级到项目根目录
    
    print("="*60)
    print("更新 README.md")
    print(f"脚本位置: {script_dir}")
    print(f"项目根目录: {base_dir}")
    print("="*60)
    update_readme(base_dir)
