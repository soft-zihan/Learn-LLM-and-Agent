# Skills 系统集成

> 📅 **更新时间**: 2026-06-29

---

## 目录

- [1. Skills 概念回顾](#1-skills-概念回顾)
- [2. LangGraph 加载 Skills](#2-langgraph-加载-skills)
- [3. Spring AI 加载 Skills](#3-spring-ai-加载-skills)
- [4. 实战案例](#4-实战案例)

---

## 1. Skills 概念回顾

### 1.1 什么是 Skills

**Agent Skill** 是**一个包含 `SKILL.md` 文件的文件夹**，AI Agent 在执行任务时动态加载，用于提升特定领域的表现。

> **Skill 不是可执行代码，而是教 AI 如何做事的结构化知识文档。**

### 1.2 Skill 标准结构

```
my-skill/
├── SKILL.md              # 必须：主指令文件
├── references/           # 可选：参考文档
├── scripts/              # 可选：实用脚本
└── templates/            # 可选：输出模板
```

### 1.3 SKILL.md 核心字段

```markdown
---
name: code-reviewer
description: Review code for quality, security, and maintainability. Use when reviewing pull requests, examining code changes, or when the user asks for a code review.
---

# Code Reviewer

## 工作流程
1. 读取待审查代码
2. 检查逻辑正确性
3. 检查安全漏洞
4. 给出改进建议
```

---

## 2. LangGraph 加载 Skills

### 2.1 从文件系统读取 Skill

```python
import os
from pathlib import Path
from typing import TypedDict

class SkillDefinition(TypedDict):
    """Skill 定义"""
    name: str
    description: str
    content: str  # SKILL.md 完整内容
    path: str     # Skill 根目录


def load_skill(skill_path: str) -> SkillDefinition:
    """从文件系统加载 Skill"""
    skill_dir = Path(skill_path)
    skill_file = skill_dir / "SKILL.md"
    
    if not skill_file.exists():
        raise FileNotFoundError(f"SKILL.md not found: {skill_file}")
    
    # 读取文件
    content = skill_file.read_text(encoding="utf-8")
    
    # 解析 YAML frontmatter（name, description）
    import re
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
    
    if not match:
        raise ValueError(f"Invalid SKILL.md format: {skill_file}")
    
    frontmatter = match.group(1)
    body = match.group(2)
    
    # 提取字段
    name_match = re.search(r"^name:\s*(.+)$", frontmatter, re.MULTILINE)
    desc_match = re.search(r"^description:\s*(.+)$", frontmatter, re.MULTILINE)
    
    name = name_match.group(1).strip() if name_match else "unknown"
    description = desc_match.group(1).strip() if desc_match else ""
    
    return {
        "name": name,
        "description": description,
        "content": content,
        "path": skill_path,
    }
```

### 2.2 Skill 注册与发现

```python
class SkillRegistry:
    """Skill 注册表"""
    
    def __init__(self):
        self.skills: dict[str, SkillDefinition] = {}
    
    def register(self, skill: SkillDefinition):
        """注册单个 Skill"""
        self.skills[skill["name"]] = skill
        print(f"✅ 注册 Skill: {skill['name']}")
    
    def register_directory(self, skills_dir: str):
        """批量注册目录下所有 Skills"""
        skills_path = Path(skills_dir)
        
        for item in skills_path.iterdir():
            if item.is_dir() and (item / "SKILL.md").exists():
                skill = load_skill(str(item))
                self.register(skill)
    
    def find_relevant(self, query: str) -> list[SkillDefinition]:
        """根据查询找到相关 Skills"""
        # 简单关键词匹配（实际可用向量检索）
        relevant = []
        query_lower = query.lower()
        
        for skill in self.skills.values():
            if (query_lower in skill["name"].lower() or 
                query_lower in skill["description"].lower()):
                relevant.append(skill)
        
        return relevant
    
    def get_all_descriptions(self) -> str:
        """获取所有 Skill 的描述（用于 System Prompt）"""
        descriptions = []
        for skill in self.skills.values():
            descriptions.append(
                f"- **{skill['name']}**: {skill['description']}"
            )
        return "\n".join(descriptions)
```

### 2.3 在 LangGraph Agent 中使用 Skills

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Annotated


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    active_skills: list[str]  # 当前激活的 Skills


class SkillEnabledAgent:
    """支持 Skills 的 Agent"""
    
    def __init__(self, skills_dir: str):
        # 1. 加载 LLM
        self.llm = ChatOpenAI(
            model="qwen3.6-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="your-api-key",
        )
        
        # 2. 加载 Skills
        self.registry = SkillRegistry()
        self.registry.register_directory(skills_dir)
        
        print(f"📚 已加载 {len(self.registry.skills)} 个 Skills")
    
    def build_system_prompt(self, query: str) -> str:
        """动态构建 System Prompt（包含相关 Skills）"""
        # 找到相关 Skills
        relevant_skills = self.registry.find_relevant(query)
        
        # 基础 System Prompt
        prompt = "你是一个智能助手。"
        
        if relevant_skills:
            prompt += f"\n\n你当前激活了 {len(relevant_skills)} 个 Skills：\n"
            for skill in relevant_skills:
                prompt += f"\n### {skill['name']}\n"
                # 只注入 SKILL.md 的核心指令（前 200 行）
                prompt += skill["content"].split("\n")[:200]
        
        prompt += "\n\n请根据 Skills 指导完成任务。"
        return prompt
    
    def create_graph(self):
        """创建 LangGraph 图"""
        
        def agent_node(state: AgentState) -> AgentState:
            # 1. 获取用户查询
            query = state["messages"][-1].content
            
            # 2. 动态构建 System Prompt
            system_prompt = self.build_system_prompt(query)
            
            # 3. 调用 LLM
            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            response = self.llm.invoke(messages)
            
            return {"messages": [response]}
        
        # 构建图
        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)
        
        return graph.compile()
    
    def run(self, query: str) -> str:
        """运行 Agent"""
        app = self.create_graph()
        result = app.invoke({
            "messages": [HumanMessage(content=query)],
            "active_skills": [],
        })
        return result["messages"][-1].content


# 使用示例
if __name__ == "__main__":
    agent = SkillEnabledAgent("path/to/skills")
    response = agent.run("请审查这段代码：\n```python\ndef foo():\n    pass\n```")
    print(response)
```

---

## 3. Spring AI 加载 Skills

### 3.1 Skills 配置类

```java
package com.example.agent;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import java.util.List;

@Data
@ConfigurationProperties(prefix = "agent.skills")
public class SkillsConfig {
    /** 是否启用 Skills */
    private boolean enabled = true;
    /** Skills 根目录 */
    private String directory = "./skills";
    /** 自动加载的 Skill 列表 */
    private List<String> autoLoad;
}
```

### 3.2 Skill 加载器

```java
package com.example.agent;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Data
@Slf4j
@Component
public class SkillLoader {

    private final Map<String, SkillDefinition> skills = new HashMap<>();

    @Data
    public static class SkillDefinition {
        private String name;
        private String description;
        private String content;
        private Path path;
    }

    /**
     * 从目录加载所有 Skills
     */
    public List<SkillDefinition> loadSkills(Path skillsDir) {
        List<SkillDefinition> loaded = new ArrayList<>();

        if (!Files.isDirectory(skillsDir)) {
            log.warn("Skills 目录不存在: {}", skillsDir);
            return loaded;
        }

        try (var paths = Files.list(skillsDir)) {
            paths.filter(Files::isDirectory)
                 .filter(dir -> Files.exists(dir.resolve("SKILL.md")))
                 .forEach(dir -> {
                     try {
                         SkillDefinition skill = loadSkill(dir);
                         skills.put(skill.getName(), skill);
                         loaded.add(skill);
                         log.info("✅ 加载 Skill: {}", skill.getName());
                     } catch (IOException e) {
                         log.error("加载 Skill 失败: {}", dir, e);
                     }
                 });
        } catch (IOException e) {
            log.error("扫描 Skills 目录失败", e);
        }

        return loaded;
    }

    /**
     * 加载单个 Skill
     */
    private SkillDefinition loadSkill(Path skillDir) throws IOException {
        Path skillFile = skillDir.resolve("SKILL.md");
        String content = Files.readString(skillFile);

        // 解析 YAML frontmatter
        Pattern pattern = Pattern.compile("^---\\n(.*?)\\n---\\n(.*)$", Pattern.DOTALL);
        Matcher matcher = pattern.matcher(content);

        SkillDefinition skill = new SkillDefinition();
        skill.setPath(skillDir);
        skill.setContent(content);

        if (matcher.find()) {
            String frontmatter = matcher.group(1);
            String body = matcher.group(2);

            // 提取 name
            Pattern namePattern = Pattern.compile("^name:\\s*(.+)$", Pattern.MULTILINE);
            Matcher nameMatcher = namePattern.matcher(frontmatter);
            if (nameMatcher.find()) {
                skill.setName(nameMatcher.group(1).trim());
            }

            // 提取 description
            Pattern descPattern = Pattern.compile("^description:\\s*(.+)$", Pattern.MULTILINE);
            Matcher descMatcher = descPattern.matcher(frontmatter);
            if (descMatcher.find()) {
                skill.setDescription(descMatcher.group(1).trim());
            }
        }

        return skill;
    }

    /**
     * 根据查询找到相关 Skills
     */
    public List<SkillDefinition> findRelevant(String query) {
        String queryLower = query.toLowerCase();

        return skills.values().stream()
                .filter(skill ->
                        skill.getName().toLowerCase().contains(queryLower) ||
                        skill.getDescription().toLowerCase().contains(queryLower)
                )
                .toList();
    }

    /**
     * 获取所有 Skill 描述（用于 System Prompt）
     */
    public String getAllDescriptions() {
        StringBuilder sb = new StringBuilder();
        for (SkillDefinition skill : skills.values()) {
            sb.append(String.format("- **%s**: %s%n", skill.getName(), skill.getDescription()));
        }
        return sb.toString();
    }
}
```

### 3.3 Skills 配置（application.yml）

```yaml
agent:
  skills:
    enabled: true
    directory: ./skills
    auto-load:
      - code-reviewer
      - doc-writer
```

### 3.4 Spring AI Agent 服务

```java
package com.example.agent;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class SkillAgentService {

    private final ChatClient.Builder chatClientBuilder;
    private final SkillLoader skillLoader;

    /**
     * 构建 System Prompt（包含 Skills）
     */
    private String buildSystemPrompt(String query) {
        StringBuilder prompt = new StringBuilder();
        prompt.append("你是一个智能助手。");

        // 找到相关 Skills
        List<SkillLoader.SkillDefinition> relevant = skillLoader.findRelevant(query);

        if (!relevant.isEmpty()) {
            prompt.append("\n\n你当前激活了 ").append(relevant.size()).append(" 个 Skills：\n");

            for (SkillLoader.SkillDefinition skill : relevant) {
                prompt.append("\n### ").append(skill.getName()).append("\n");
                // 只注入核心指令（前 200 行）
                String[] lines = skill.getContent().split("\n");
                int limit = Math.min(lines.length, 200);
                for (int i = 0; i < limit; i++) {
                    prompt.append(lines[i]).append("\n");
                }
            }
        }

        prompt.append("\n\n请根据 Skills 指导完成任务。");
        return prompt.toString();
    }

    /**
     * 同步调用
     */
    public String chat(String userMessage) {
        String systemPrompt = buildSystemPrompt(userMessage);

        return ChatClient.builder(chatClientBuilder)
                .defaultSystem(systemPrompt)
                .build()
                .prompt()
                .user(userMessage)
                .call()
                .content();
    }

    /**
     * 流式调用
     */
    public Flux<String> chatStream(String userMessage) {
        String systemPrompt = buildSystemPrompt(userMessage);

        return ChatClient.builder(chatClientBuilder)
                .defaultSystem(systemPrompt)
                .build()
                .prompt()
                .user(userMessage)
                .stream()
                .content();
    }
}
```

---

## 4. 实战案例

### 4.1 LangGraph 加载 skill-creator-example

完整示例见：`examples/13-skills-langgraph.py`

**关键步骤**：
1. 使用 `SkillRegistry` 加载目录下所有 Skills
2. 根据用户查询动态找到相关 Skills
3. 将 SKILL.md 注入到 System Prompt
4. LangGraph Agent 执行任务

### 4.2 Spring AI 加载 Skills

完整示例见：`examples/spring-ai-demo/` 扩展

**关键步骤**：
1. 配置 `agent.skills.directory`
2. `SkillLoader` 自动扫描并加载
3. `SkillAgentService` 动态构建 System Prompt
4. ChatClient 执行任务

### 4.3 对比总结

| 维度 | LangGraph | Spring AI |
|------|-----------|-----------|
| Skill 加载 | 手动 `SkillRegistry` | 自动扫描目录 |
| 动态注入 | 手动构建 System Prompt | `buildSystemPrompt()` 方法 |
| 相关性匹配 | 简单关键词匹配 | 可扩展向量检索 |
| 灵活性 | 高（完全控制） | 中（约定优于配置） |
| 适合场景 | Python Agent 开发 | Java 微服务 |

### 4.4 Spring AI Skills REST 控制器

完整的 Web API 实现：

```java
package com.example.agent;

import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/skills")
@RequiredArgsConstructor
public class SkillAgentController {

    private final SkillAgentService agentService;
    private final SkillLoader skillLoader;

    /**
     * 同步对话（带 Skills）
     * POST /api/skills/chat
     * Body: {"message": "请审查这段代码"}
     */
    @PostMapping("/chat")
    public Map<String, String> chat(@RequestBody Map<String, String> request) {
        String response = agentService.chat(request.get("message"));
        return Map.of("content", response);
    }

    /**
     * 流式对话（SSE）
     * POST /api/skills/chat/stream
     */
    @PostMapping(value = "/chat/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> chatStream(@RequestBody Map<String, String> request) {
        return agentService.chatStream(request.get("message"));
    }

    /**
     * 列出所有已加载的 Skills
     * GET /api/skills/list
     */
    @GetMapping("/list")
    public List<Map<String, String>> listSkills() {
        return skillLoader.getSkills().values().stream()
            .map(skill -> Map.of(
                "name", skill.getName(),
                "description", skill.getDescription()
            ))
            .toList();
    }

    /**
     * 查询相关 Skills
     * GET /api/skills/search?query=code
     */
    @GetMapping("/search")
    public List<Map<String, String>> searchSkills(@RequestParam String query) {
        return skillLoader.findRelevant(query).stream()
            .map(skill -> Map.of(
                "name", skill.getName(),
                "description", skill.getDescription(),
                "content", skill.getContent().substring(0, Math.min(200, skill.getContent().length()))
            ))
            .toList();
    }
}
```

---

## 5. 最佳实践

### 5.1 Skill 缓存

```python
# LangGraph：缓存已加载的 Skills
import functools

@functools.lru_cache(maxsize=32)
def load_skill_cached(skill_path: str) -> SkillDefinition:
    return load_skill(skill_path)
```

```java
// Spring AI：使用 Spring Cache
@Cacheable(value = "skills", key = "#skillDir.toString()")
public SkillDefinition loadSkill(Path skillDir) {
    // ...
}
```

### 5.2 Skill 版本控制

```python
# 在 SKILL.md frontmatter 中添加版本
---
name: code-reviewer
version: 1.2.0
description: ...
---
```

### 5.3 热重载

```python
# 监听文件系统变化，自动重载 Skills
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class SkillReloader(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("SKILL.md"):
            print(f"🔄 热重载 Skill: {event.src_path}")
            self.registry.register(event.src_path)
```
