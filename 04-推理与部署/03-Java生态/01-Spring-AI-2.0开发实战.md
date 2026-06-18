# Spring AI 2.0 Java 生态 LLM 开发

> 📅 **更新时间**: 2026-06-17

---

## 目录

- [1. Spring AI 概述](#1-spring-ai-概述)
- [2. 快速入门](#2-快速入门)
- [3. 核心概念](#3-核心概念)
- [4. 工具调用与 Function Calling](#4-工具调用与-function-calling)
- [5. RAG 实现](#5-rag-实现)
- [6. Agent 开发](#6-agent-开发)
- [7. 生产级应用开发](#7-生产级应用开发)
- [8. 最佳实践](#8-最佳实践)
- [9. 参考资料](#9-参考资料)

---

## 1. Spring AI 概述

### 1.1 为什么需要 Spring AI？

Java 开发者在 AI 时代面临的挑战：

- **Python 主导**：LLM 生态以 Python 为主（LangChain、LlamaIndex）
- **Java 生态碎片化**：多个库并存，缺乏统一标准
- **企业需求**：大量企业系统基于 Java/Spring 构建
- **集成复杂度**：将 AI 能力集成到现有 Spring 应用困难

Spring AI 的解决方案：

| 特性 | 描述 | 优势 |
|------|------|------|
| **统一 API** | 抽象多模型提供商 | 切换模型无需改代码 |
| **Spring 原生** | 遵循 Spring 设计哲学 | Java 开发者易上手 |
| **生产就绪** | Spring Boot 集成 | 企业级部署 |
| **生态完整** | Prompt、RAG、Agent | 全栈 AI 开发 |
| **类型安全** | Java 强类型 | 编译期检查错误 |

### 1.2 Spring AI vs LangChain4j 对比

```
Spring AI vs LangChain4j
├── Spring AI
│   ├── 优势
│   │   ├── Spring 官方支持
│   │   ├── 与 Spring Boot 深度集成
│   │   ├── 统一抽象层设计
│   │   └── 活跃的社区
│   └── 适用场景
│       ├── 现有 Spring 应用添加 AI
│       ├── 企业级生产部署
│       └── 需要多模型支持
│
└── LangChain4j
    ├── 优势
    │   ├── 功能更丰富（早期）
    │   ├── LangChain 生态移植
    │   └── 文档完善
    └── 适用场景
        ├── LangChain Python 用户转 Java
        ├── 需要丰富组件库
        └── 原型开发
```

**2026 年选型建议**：
- **新项目**：优先 Spring AI（官方支持、长期维护）
- **现有 Spring 应用**：Spring AI（无缝集成）
- **需要 LangChain 组件**：LangChain4j

### 1.3 Spring AI 2.0 GA 新特性

Spring AI 2.0.0-GA（2026 年 6 月正式发布）核心更新：

- **Spring Boot 4.0 GA 支持**：全面兼容 Spring Boot 4.0 稳定版
- **Jackson 3 适配**：迁移至 Jackson 3.x，提升 JSON 处理性能
- **Null-safety 增强**：全面引入 `@Nullable` 和 `@NonNull` 注解，提升代码安全性
- **增强 Agent 框架**：Tool Calling Agent、ReAct Agent
- **改进的 RAG**：GraphRAG、混合检索
- **多模态支持**：图像、音频处理
- **性能优化**：流式输出、批量处理
- **可观测性**：Micrometer 集成、分布式追踪

### 1.4 与 Spring Boot 4 集成

```xml
<!-- pom.xml -->
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>4.0.0</version>
</parent>

<dependencyManagement>
    <dependencies>
        <!-- Spring AI BOM（推荐） -->
        <dependency>
            <groupId>org.springframework.ai</groupId>
            <artifactId>spring-ai-bom</artifactId>
            <version>2.0.0</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>

<dependencies>
    <!-- Spring Boot Web -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    
    <!-- Spring AI OpenAI Starter（使用 BOM 管理版本） -->
    <dependency>
        <groupId>org.springframework.ai</groupId>
        <artifactId>spring-ai-starter-model-openai</artifactId>
    </dependency>
    
    <!-- Spring AI Vector Store -->
    <dependency>
        <groupId>org.springframework.ai</groupId>
        <artifactId>spring-ai-starter-vector-store-chroma</artifactId>
    </dependency>
</dependencies>
```

**⚠️ 2.0 重要变化**：
- 使用 `spring-ai-bom` 统一管理版本，无需为每个依赖指定版本号
- Starter 命名变更：`spring-ai-openai-spring-boot-starter` → `spring-ai-starter-model-openai`
- GA 版本直接在 Maven Central，无需额外配置仓库

## 2. 快速入门

### 2.1 项目依赖配置

#### Maven 配置

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>4.0.0</version>
    </parent>
    
    <groupId>com.example</groupId>
    <artifactId>spring-ai-demo</artifactId>
    <version>1.0.0</version>
    
    <properties>
        <java.version>21</java.version>
        <spring-ai.version>2.0.0</spring-ai.version>
    </properties>
    
    <dependencyManagement>
        <dependencies>
            <!-- Spring AI BOM -->
            <dependency>
                <groupId>org.springframework.ai</groupId>
                <artifactId>spring-ai-bom</artifactId>
                <version>${spring-ai.version}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>
    
    <dependencies>
        <!-- Spring AI OpenAI（使用 BOM 管理版本，无需 version） -->
        <dependency>
            <groupId>org.springframework.ai</groupId>
            <artifactId>spring-ai-starter-model-openai</artifactId>
        </dependency>
        
        <!-- Spring AI Ollama（本地模型） -->
        <dependency>
            <groupId>org.springframework.ai</groupId>
            <artifactId>spring-ai-starter-model-ollama</artifactId>
        </dependency>
        
        <!-- Spring AI Chroma Vector Store -->
        <dependency>
            <groupId>org.springframework.ai</groupId>
            <artifactId>spring-ai-starter-vector-store-chroma</artifactId>
        </dependency>
        
        <!-- Spring Boot Actuator（监控） -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-actuator</artifactId>
        </dependency>
    </dependencies>
    
    <repositories>
        <!-- GA 版本无需 milestone 仓库，使用 Maven Central -->
    </repositories>
</project>
```

### 2.2 第一个 LLM 调用

```java
package com.example.demo;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@SpringBootApplication
@RestController
public class SpringAiDemoApplication {
    
    private final ChatClient chatClient;
    
    public SpringAiDemoApplication(ChatClient.Builder chatClientBuilder) {
        // 构建 ChatClient
        this.chatClient = chatClientBuilder.build();
    }
    
    @GetMapping("/chat")
    public String chat(@RequestParam String message) {
        // 简单对话
        return chatClient.prompt()
                .user(message)
                .call()
                .content();
    }
    
    public static void main(String[] args) {
        SpringApplication.run(SpringAiDemoApplication.class, args);
    }
}
```

#### 配置文件

```yaml
# application.yml
spring:
  ai:
    openai:
      api-key: ${OPENAI_API_KEY}
      chat:
        # ⚠️ 2.0 变化：移除 .options 前缀
        model: gpt-4o
        temperature: 0.7
        max-tokens: 2048
    
    # 全局工具调用配置
    chat:
      client:
        tool-calling:
          enabled: true  # 默认启用 ToolCallingAdvisor
```

**⚠️ 2.0 配置变化**：
- 移除 `.options` 前缀：`spring.ai.openai.chat.options.model` → `spring.ai.openai.chat.model`
- 新增工具调用全局配置：`spring.ai.chat.client.tool-calling.enabled`

### 2.3 多模型提供商支持

```java
@Configuration
public class AiModelConfig {
    
    // OpenAI
    @Bean
    @Primary
    public ChatClient openaiChatClient(ChatClient.Builder builder) {
        return builder.build();
    }
    
    // Ollama（本地）
    @Bean("ollamaChatClient")
    public ChatClient ollamaChatClient(
            @Qualifier("ollamaChatClientBuilder") ChatClient.Builder builder) {
        return builder.build();
    }
    
    // Azure OpenAI
    @Bean("azureChatClient")
    public ChatClient azureChatClient(
            @Qualifier("azureChatClientBuilder") ChatClient.Builder builder) {
        return builder.build();
    }
}
```

```yaml
# application.yml - 多模型配置
spring:
  ai:
    openai:
      api-key: ${OPENAI_API_KEY}
      chat:
        # 2.0: 移除 .options 前缀
        model: gpt-4o
    
    ollama:
      base-url: http://localhost:11434
      chat:
        model: llama3.1:8b
        temperature: 0.7
    
    azure:
      openai:
        api-key: ${AZURE_OPENAI_API_KEY}
        endpoint: ${AZURE_OPENAI_ENDPOINT}
        chat:
          model: gpt-4
```

### 2.4 配置管理

```java
@ConfigurationProperties(prefix = "app.ai")
public record AiProperties(
    String defaultModel,
    Double defaultTemperature,
    Integer maxTokens,
    Boolean enableStreaming,
    Integer maxRetries
) {}

// 使用配置
@Service
public class ChatService {
    
    private final ChatClient chatClient;
    private final AiProperties aiProperties;
    
    public ChatService(ChatClient chatClient, AiProperties aiProperties) {
        this.chatClient = chatClient;
        this.aiProperties = aiProperties;
    }
    
    public String chat(String message) {
        return chatClient.prompt()
                .options(ChatOptions.builder()
                        .model(aiProperties.defaultModel())
                        .temperature(aiProperties.defaultTemperature())
                        .maxTokens(aiProperties.maxTokens())
                        .build())
                .user(message)
                .call()
                .content();
    }
}
```

## 3. 核心概念

### 3.1 ChatClient API

**⚠️ 2.0 重要变化**：
- **Options 不可变性**：`ChatOptions` 现在严格不可变，使用 `mutate()` 替代 `copy()`
- **Advisor 链架构**：工具调用、记忆管理等都通过 Advisor 实现
- **combineWith()**：新增 builder 方法组合多个配置

```java
@Service
public class ChatClientExamples {
    
    private final ChatClient chatClient;
    
    public ChatClientExamples(ChatClient chatClient) {
        this.chatClient = chatClient;
    }
    
    // 1. 简单对话
    public String simpleChat(String message) {
        return chatClient.prompt()
                .user(message)
                .call()
                .content();
    }
    
    // 2. 带系统提示词
    public String chatWithSystem(String userMessage) {
        return chatClient.prompt()
                .system("你是一个专业的编程助手")
                .user(userMessage)
                .call()
                .content();
    }
    
    // 3. 流式输出
    public Flux<String> streamingChat(String message) {
        return chatClient.prompt()
                .user(message)
                .stream()
                .content();
    }
    
    // 4. 多轮对话
    public String multiTurnChat(List<Message> history, String newMessage) {
        return chatClient.prompt()
                .messages(history)
                .user(newMessage)
                .call()
                .content();
    }
    
    // 5. 结构化输出
    public PersonResponse structuredChat(String description) {
        return chatClient.prompt()
                .user("根据描述创建人物：" + description)
                .call()
                .entity(PersonResponse.class);
    }
    
    // 6. 使用 Options（2.0 不可变方式）
    public String chatWithOptions(String message) {
        // 2.0: 使用 mutate() 而非 copy()
        ChatOptions options = chatClient.getOptions().mutate()
                .model("gpt-4o")
                .temperature(0.7)
                .maxTokens(2048)
                .build();
        
        return chatClient.prompt()
                .options(options)
                .user(message)
                .call()
                .content();
    }
}
```

### 3.2 Prompt 模板系统

```java
@Service
public class PromptTemplateService {
    
    private final ChatClient chatClient;
    
    // 1. 简单模板
    public String simpleTemplate(String topic) {
        PromptTemplate template = new PromptTemplate(
                "解释 {topic} 的基本概念"
        );
        
        String prompt = template.render(Map.of("topic", topic));
        
        return chatClient.prompt()
                .user(prompt)
                .call()
                .content();
    }
    
    // 2. 复杂模板（从文件加载）
    public String complexTemplate(Map<String, Object> variables) {
        ClassPathResource resource = new ClassPathResource(
                "prompts/code-review.st"
        );
        
        PromptTemplate template = new PromptTemplate(resource);
        String prompt = template.render(variables);
        
        return chatClient.prompt()
                .user(prompt)
                .call()
                .content();
    }
    
    // 3. 多消息模板
    public String multiMessageTemplate(String code, String language) {
        return chatClient.prompt()
                .system("""
                    你是一位资深代码审查专家。
                    请审查以下 {language} 代码。
                    """)
                .user("""
                    请审查以下代码：
                    
                    ```{language}
                    {code}
                    ```
                    
                    请从以下方面审查：
                    1. 代码质量
                    2. 潜在 bug
                    3. 性能优化
                    4. 最佳实践
                    """, Map.of(
                        "language", language,
                        "code", code
                    ))
                .call()
                .content();
    }
}
```

#### Prompt 模板文件

```
// resources/prompts/code-review.st
你是一位资深代码审查专家。

请审查以下 {language} 代码：

```{language}
{code}
```

审查要求：
1. 识别潜在 bug 和边界情况
2. 评估代码可读性和可维护性
3. 检查性能问题
4. 提供改进建议

请以 JSON 格式返回审查结果。
```

### 3.3 输出解析器

```java
@Service
public class OutputParserService {
    
    private final ChatClient chatClient;
    
    // 1. Bean Output 解析
    public record CodeReview(
        String quality,
        List<String> issues,
        List<String> suggestions,
        Double score
    ) {}
    
    public CodeReview parseBeanOutput(String code) {
        return chatClient.prompt()
                .user("""
                    审查以下代码，以 JSON 格式返回：
                    
                    ```
                    {code}
                    ```
                    """.formatted(code))
                .call()
                .entity(CodeReview.class);
    }
    
    // 2. 列表输出
    public List<String> parseListOutput(String topic) {
        return chatClient.prompt()
                .user("列出 5 个关于 {topic} 的关键词".formatted(topic))
                .call()
                .entity(new ParameterizedTypeReference<List<String>>() {});
    }
    
    // 3. Map 输出
    public Map<String, Object> parseMapOutput(String data) {
        return chatClient.prompt()
                .user("解析以下数据：{data}".formatted(data))
                .call()
                .entity(new ParameterizedTypeReference<Map<String, Object>>() {});
    }
}
```

### 3.4 消息历史管理

```java
@Service
public class ChatHistoryService {
    
    // 内存存储（开发环境）
    private final Map<String, List<Message>> histories = new ConcurrentHashMap<>();
    
    // 添加消息
    public void addMessage(String sessionId, String role, String content) {
        histories.computeIfAbsent(sessionId, k -> new ArrayList<>())
                .add(new Message(role, content));
    }
    
    // 获取历史
    public List<Message> getHistory(String sessionId) {
        return histories.getOrDefault(sessionId, List.of());
    }
    
    // 带历史的对话
    public String chatWithHistory(String sessionId, String newMessage) {
        List<Message> history = getHistory(sessionId);
        
        String response = chatClient.prompt()
                .messages(history)
                .user(newMessage)
                .call()
                .content();
        
        // 保存新消息
        addMessage(sessionId, "user", newMessage);
        addMessage(sessionId, "assistant", response);
        
        return response;
    }
    
    // 清除历史
    public void clearHistory(String sessionId) {
        histories.remove(sessionId);
    }
}
```

## 4. 工具调用与 Function Calling

### 4.1 工具定义与注册

**⚠️ 2.0 重大变化**：Spring AI 2.0 引入了 **Advisor 链架构**，Tool Calling 现在通过 `ToolCallingAdvisor` 自动处理，不再需要手动管理工具执行循环。

```java
@Service
public class ToolDefinitionService {
    
    // 1. 使用 @Tool 注解（推荐）
    @Tool(description = "查询当前天气")
    public String getWeather(
            @ToolParam("city") String city,
            @ToolParam(value = "unit", required = false) String unit) {
        // 调用天气 API
        return weatherService.fetchWeather(city);
    }
    
    // 2. 编程式定义（使用 ToolCallback）
    @Bean
    public MethodToolCallback weatherToolCallback() {
        return MethodToolCallback.builder()
                .toolMethod(ToolDefinitionService.class, "getWeather")
                .build();
    }
}
```

**⚠️ 2.0 API 变化**：
- `@Param` → `@ToolParam`（专用注解）
- `FunctionCallback` → `MethodToolCallback`（新抽象）
- `internalToolExecutionEnabled` 选项已移除，工具执行现在由 Advisor 自动管理

### 4.2 自动工具调用

**⚠️ 2.0 核心变化**：`ToolCallingAdvisor` 现在自动注册到 `ChatClient`，工具调用在 Advisor 链中作为一等公民处理。

```java
@Service
public class ToolCallingService {
    
    private final ChatClient chatClient;
    
    // 2.0 方式：工具自动通过 Advisor 执行
    public ToolCallingService(ChatClient.Builder builder,
                              MethodToolCallback weatherToolCallback) {
        this.chatClient = builder
                .defaultTools(weatherToolCallback)  // 注册工具
                .build();
        // ToolCallingAdvisor 自动注册，无需手动配置
    }
    
    // 自动调用工具（Advisor 自动处理工具循环）
    public String chatWithTools(String message) {
        return chatClient.prompt()
                .system("你可以使用工具来帮助用户")
                .user(message)
                .call()
                .content();
    }
    
    // 禁用自动工具执行（可选）
    public String chatWithoutToolExecution(String message) {
        return chatClient.prompt()
                .system("你可以使用工具来帮助用户")
                .user(message)
                .advisors(AdvisorParams.toolCallingAdvisorAutoRegister(false))
                .call()
                .content();
    }
}

// 使用示例
@RestController
public class ToolController {
    
    private final ToolCallingService toolService;
    
    @GetMapping("/chat/tools")
    public String chat(@RequestParam String message) {
        // 用户问："北京今天天气怎么样？"
        // ToolCallingAdvisor 自动：
        // 1. 检测模型返回的工具调用请求
        // 2. 执行工具获取结果
        // 3. 将结果返回给模型
        // 4. 获取最终回复
        return toolService.chatWithTools(message);
    }
}
```

**⚠️ 2.0 移除的功能**：
- `internalToolExecutionEnabled` 配置项已移除
- `streamToolCallResponses` 选项已移除（设计缺陷）
- 使用 `AdvisorParams.toolCallingAdvisorAutoRegister(false)` 手动控制工具执行

### 4.3 多工具协调

```java
@Configuration
public class MultiToolConfig {
    
    @Bean
    public ChatClient multiToolChatClient(
            ChatClient.Builder builder,
            ToolCallback weatherTool,
            ToolCallback calendarTool,
            ToolCallback emailTool) {
        
        return builder
                .defaultTools(
                        weatherTool,
                        calendarTool,
                        emailTool
                )
                .build();
    }
}

@Service
public class MultiToolService {
    
    private final ChatClient chatClient;
    
    // 多工具自动协调（ToolCallingAdvisor 自动管理工具循环）
    public String handleComplexRequest(String request) {
        return chatClient.prompt()
                .system("""
                    你是个人助理。
                    可以使用以下工具：
                    - 查询天气
                    - 查看日程
                    - 发送邮件
                    """)
                .user(request)
                .call()
                .content();
    }
}

// 示例：复杂请求
// 用户："明天上海天气如何？如果有空，帮我发邮件给张三"
// ToolCallingAdvisor 自动执行：
// 1. 调用 getWeather("上海")
// 2. 调用 checkSchedule("明天")
// 3. 如果有空，调用 sendEmail("张三", ...)
```

### 4.4 Tool Search Advisor（2.0 新特性）

当工具数量很多时，可以使用 `ToolSearchToolCallingAdvisor` 动态选择最相关的工具：

```java
// 配置 Tool Search Advisor
spring:
  ai:
    chat:
      client:
        tool-search-advisor:
          enabled: true
          # 索引类型：regex（默认）, lucene, vector
          tool-index-type: regex

// 添加依赖
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-starter-tool-search-advisor</artifactId>
</dependency>
```

```java
@Service
public class ToolSearchService {
    
    private final ChatClient chatClient;
    
    // Tool Search Advisor 会根据问题动态选择最相关的工具
    // 而不是将所有工具定义都发送给模型
    public String chatWithToolSearch(String message) {
        return chatClient.prompt()
                .user(message)
                .call()
                .content();
    }
}
```

**优势**：
- 减少 token 消耗（只发送相关工具定义）
- 提高模型响应质量（减少工具混淆）
- 支持数百个工具的动态检索

### 4.5 实战案例：天气查询助手

```java
@RestController
@RequestMapping("/api/weather")
public class WeatherAssistantController {
    
    private final ChatClient chatClient;
    
    public WeatherAssistantController(ChatClient.Builder builder) {
        // 2.0: 使用 defaultTools 和 ToolCallback
        this.chatClient = builder
                .defaultTools(weatherToolCallbacks())
                .build();
    }
    
    @PostMapping("/chat")
    public ResponseEntity<ChatResponse> chat(
            @RequestBody ChatRequest request) {
        
        String response = chatClient.prompt()
                .system("""
                    你是天气助手。
                    可以查询全球主要城市的天气信息。
                    请用友好的语气回复。
                    """)
                .user(request.message())
                .call()
                .content();
        
        return ResponseEntity.ok(new ChatResponse(response));
    }
    
    @Bean
    public ToolCallback[] weatherToolCallbacks() {
        return new ToolCallback[] {
            MethodToolCallback.builder()
                    .toolMethod(this, "getCurrentWeather")
                    .build()
        };
    }
    
    @Tool(description = "获取指定城市的当前天气")
    public String getCurrentWeather(
            @ToolParam("city") String city,
            @ToolParam(value = "unit", required = false) String unit) {
        // 调用天气 API
        WeatherData data = weatherApiClient.fetchWeather(city);
        
        return """
            城市：%s
            天气：%s
            温度：%.1f°%s
            湿度：%d%%
            风速：%.1f m/s
            """.formatted(
                city,
                data.condition(),
                data.temperature(),
                unit != null && unit.equals("fahrenheit") ? "F" : "C",
                data.humidity(),
                data.windSpeed()
            );
    }
}

// 请求示例
record ChatRequest(String message) {}
record ChatResponse(String reply) {}

// POST /api/weather/chat
// {
//   "message": "上海今天天气怎么样？"
// }
// 
// 响应：
// {
//   "reply": "上海今天晴天，温度 22°C，湿度 60%，适合外出！"
// }
```

## 5. RAG 实现

### 5.1 文档读取器

```java
@Service
public class DocumentReaderService {
    
    // 1. PDF 读取
    public List<Document> readPdf(Path pdfPath) {
        PdfDocumentReader reader = new PdfDocumentReader(
                new org.springframework.core.io.FileSystemResource(pdfPath)
        );
        
        return reader.get();
    }
    
    // 2. TXT 读取
    public List<Document> readText(Path txtPath) {
        TextDocumentReader reader = new TextDocumentReader(
                new org.springframework.core.io.FileSystemResource(txtPath)
        );
        
        return reader.get();
    }
    
    // 3. Web 页面读取
    public List<Document> readWebPage(String url) {
        // 使用 Jsoup 抓取网页
        String html = webClient.get()
                .uri(url)
                .retrieve()
                .bodyToMono(String.class)
                .block();
        
        // 解析 HTML
        TextDocumentReader reader = new TextDocumentReader(
                html,
                new Metadata()
        );
        
        return reader.get();
    }
}
```

### 5.2 文档分割策略

```java
@Service
public class DocumentSplittingService {
    
    // 1. 字符分割
    public List<Document> splitByCharacter(List<Document> documents) {
        CharacterTextSplitter splitter = new CharacterTextSplitter(
                1000,  // chunk size
                200    // chunk overlap
        );
        
        return splitter.apply(documents);
    }
    
    // 2. 递归字符分割
    public List<Document> splitRecursive(List<Document> documents) {
        TokenTextSplitter splitter = new TokenTextSplitter(
                500,   // chunk size
                100,   // chunk overlap
                10000, // min chunk size
                10     // max chunks
        );
        
        return splitter.apply(documents);
    }
    
    // 3. 语义分割（使用 LLM）
    public List<Document> splitSemantic(List<Document> documents,
                                        ChatClient chatClient) {
        List<Document> result = new ArrayList<>();
        
        for (Document doc : documents) {
            // 使用 LLM 识别语义边界
            String splitPrompt = """
                将以下文本按主题分割成多个段落：
                
                {text}
                
                请用 <SPLIT> 标记分隔点。
                """.formatted(doc.getText());
            
            String splitResult = chatClient.prompt()
                    .user(splitPrompt)
                    .call()
                    .content();
            
            // 解析分割结果
            String[] chunks = splitResult.split("<SPLIT>");
            for (String chunk : chunks) {
                result.add(new Document(chunk.trim(), doc.getMetadata()));
            }
        }
        
        return result;
    }
}
```

### 5.3 向量存储

```java
@Configuration
public class VectorStoreConfig {
    
    // Chroma（轻量级）
    @Bean
    public VectorStore chromaVectorStore(
            ChromaApi chromaApi,
            EmbeddingModel embeddingModel) {
        return new ChromaVectorStore(chromaApi, embeddingModel, "my-collection");
    }
    
    // Milvus（分布式）
    @Bean
    public VectorStore milvusVectorStore(
            MilvusServiceClient milvusClient,
            EmbeddingModel embeddingModel) {
        return new MilvusVectorStore(milvusClient, embeddingModel, "documents");
    }
}

@Service
public class VectorStoreService {
    
    private final VectorStore vectorStore;
    
    public VectorStoreService(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }
    
    // 添加文档
    public void addDocuments(List<Document> documents) {
        vectorStore.add(documents);
    }
    
    // 相似性搜索
    public List<Document> search(String query, int topK) {
        return vectorStore.similaritySearch(
                SearchRequest.query(query)
                        .withTopK(topK)
                        .withSimilarityThreshold(0.7)
        );
    }
    
    // 带过滤的搜索
    public List<Document> searchWithFilter(String query,
                                           String category) {
        return vectorStore.similaritySearch(
                SearchRequest.query(query)
                        .withTopK(5)
                        .withFilterExpression("category == '{category}'")
        );
    }
}
```

### 5.4 检索与增强生成

```java
@Service
public class RagService {
    
    private final ChatClient chatClient;
    private final VectorStore vectorStore;
    
    public RagService(ChatClient chatClient, VectorStore vectorStore) {
        this.chatClient = chatClient;
        this.vectorStore = vectorStore;
    }
    
    // 基础 RAG
    public String ragQuery(String question) {
        // 1. 检索相关文档
        List<Document> documents = vectorStore.similaritySearch(
                SearchRequest.query(question).withTopK(5)
        );
        
        // 2. 构建上下文
        String context = documents.stream()
                .map(Document::getText)
                .collect(Collectors.joining("\n\n"));
        
        // 3. 增强生成
        return chatClient.prompt()
                .system("""
                    你是知识助手。
                    请基于提供的参考资料回答问题。
                    如果资料不足，请说明。
                    """)
                .user("""
                    参考资料：
                    {context}
                    
                    问题：{question}
                    
                    回答：
                    """, Map.of(
                        "context", context,
                        "question", question
                    ))
                .call()
                .content();
    }
    
    // 流式 RAG
    public Flux<String> streamingRagQuery(String question) {
        List<Document> documents = vectorStore.similaritySearch(
                SearchRequest.query(question).withTopK(5)
        );
        
        String context = documents.stream()
                .map(Document::getText)
                .collect(Collectors.joining("\n\n"));
        
        return chatClient.prompt()
                .system("基于参考资料回答问题")
                .user("""
                    参考资料：{context}
                    问题：{question}
                    """, Map.of("context", context, "question", question))
                .stream()
                .content();
    }
}
```

### 5.5 实战案例：知识库问答

```java
@RestController
@RequestMapping("/api/knowledge")
public class KnowledgeBaseController {
    
    private final RagService ragService;
    private final DocumentIngestionService ingestionService;
    
    @PostMapping("/query")
    public ResponseEntity<KnowledgeResponse> query(
            @RequestBody KnowledgeQuery query) {
        
        String answer = ragService.ragQuery(query.question());
        
        return ResponseEntity.ok(new KnowledgeResponse(answer));
    }
    
    @PostMapping("/ingest")
    public ResponseEntity<String> ingest(
            @RequestParam MultipartFile file) {
        
        try {
            ingestionService.ingestDocument(file);
            return ResponseEntity.ok("文档导入成功");
        } catch (Exception e) {
            return ResponseEntity.status(500)
                    .body("导入失败：" + e.getMessage());
        }
    }
}

record KnowledgeQuery(String question) {}
record KnowledgeResponse(String answer) {}

@Service
public class DocumentIngestionService {
    
    private final DocumentReaderService readerService;
    private final DocumentSplittingService splittingService;
    private final VectorStore vectorStore;
    
    public void ingestDocument(MultipartFile file) throws IOException {
        // 1. 保存临时文件
        Path tempFile = Files.createTempFile("upload-", ".pdf");
        file.transferTo(tempFile);
        
        // 2. 读取文档
        List<Document> documents = readerService.readPdf(tempFile);
        
        // 3. 分割文档
        List<Document> chunks = splittingService.splitRecursive(documents);
        
        // 4. 添加到向量存储
        vectorStore.add(chunks);
        
        // 5. 清理临时文件
        Files.delete(tempFile);
    }
}
```

## 6. Agent 开发

### 6.1 Tool Calling Agent

**⚠️ 2.0 核心变化**：Tool Calling 现在是 Advisor 链的一部分，自动处理工具循环。

```java
@Service
public class ToolCallingAgent {
    
    private final ChatClient chatClient;
    
    // 2.0: 工具通过 Advisor 自动执行
    public ToolCallingAgent(ChatClient.Builder builder,
                            List<ToolCallback> tools) {
        this.chatClient = builder
                .defaultTools(tools.toArray(new ToolCallback[0]))
                .build();
        // ToolCallingAdvisor 自动注册并管理工具循环
    }
    
    public String execute(String task) {
        return chatClient.prompt()
                .system("""
                    你是任务执行 Agent。
                    可以使用各种工具来完成任务。
                    请逐步思考并执行。
                    """)
                .user(task)
                .call()
                .content();
    }
    
    // 自定义 ToolCallingAdvisor（可选）
    public String executeWithCustomAdvisor(String task) {
        // 自定义工具执行检查器
        ToolCallingAdvisor customAdvisor = ToolCallingAdvisor.builder()
                .toolExecutionEligibilityChecker(response -> 
                    response != null && response.hasToolCalls())
                .build();
        
        return chatClient.prompt()
                .advisors(customAdvisor)
                .user(task)
                .call()
                .content();
    }
}

// 使用示例
@RestController
@RequestMapping("/api/agent")
public class AgentController {
    
    private final ToolCallingAgent agent;
    
    @PostMapping("/execute")
    public ResponseEntity<AgentResponse> execute(
            @RequestBody AgentRequest request) {
        
        String result = agent.execute(request.task());
        
        return ResponseEntity.ok(new AgentResponse(result));
    }
}
```

### 6.2 ReAct Agent

```java
@Service
public class ReActAgent {
    
    private final ChatClient chatClient;
    private final ToolRegistry toolRegistry;
    
    public String execute(String task) {
        StringBuilder thoughtProcess = new StringBuilder();
        
        for (int i = 0; i < 10; i++) {  // 最多 10 步
            // 1. 思考
            String thought = chatClient.prompt()
                    .system("""
                        使用 ReAct 模式：
                        Thought: 下一步做什么？
                        Action: 使用什么工具？
                        Action Input: 工具输入是什么？
                        """)
                    .user("""
                        任务：{task}
                        历史：{history}
                        
                        请给出下一步的 Thought/Action/Action Input。
                        如果任务完成，输出 "FINISHED"。
                        """, Map.of(
                            "task", task,
                            "history", thoughtProcess.toString()
                        ))
                    .call()
                    .content();
            
            // 2. 检查是否完成
            if (thought.contains("FINISHED")) {
                break;
            }
            
            // 3. 执行动作
            String action = extractAction(thought);
            String actionInput = extractActionInput(thought);
            
            String observation = toolRegistry.execute(action, actionInput);
            
            // 4. 记录
            thoughtProcess.append(thought)
                    .append("\nObservation: ")
                    .append(observation)
                    .append("\n");
        }
        
        // 5. 生成最终答案
        return chatClient.prompt()
                .user("""
                    任务：{task}
                    思考过程：{thoughts}
                    
                    请给出最终答案。
                    """, Map.of(
                        "task", task,
                        "thoughts", thoughtProcess.toString()
                    ))
                .call()
                .content();
    }
}
```

### 6.3 多 Agent 协作

```java
@Service
public class MultiAgentSystem {
    
    private final ResearchAgent researchAgent;
    private final WritingAgent writingAgent;
    private final ReviewAgent reviewAgent;
    
    public String executeWorkflow(String topic) {
        // 1. 研究
        String research = researchAgent.research(topic);
        
        // 2. 写作
        String draft = writingAgent.write(research);
        
        // 3. 审核
        String finalResult = reviewAgent.review(draft);
        
        return finalResult;
    }
}

@Component
class ResearchAgent {
    private final ChatClient chatClient;
    
    public String research(String topic) {
        return chatClient.prompt()
                .system("你是研究员。搜集关于主题的信息。")
                .user("研究主题：" + topic)
                .call()
                .content();
    }
}

@Component
class WritingAgent {
    private final ChatClient chatClient;
    
    public String write(String research) {
        return chatClient.prompt()
                .system("你是作家。基于研究结果撰写文章。")
                .user("基于以下研究写文章：\n" + research)
                .call()
                .content();
    }
}

@Component
class ReviewAgent {
    private final ChatClient chatClient;
    
    public String review(String draft) {
        return chatClient.prompt()
                .system("你是审核员。审查并改进文章。")
                .user("审查以下文章：\n" + draft)
                .call()
                .content();
    }
}
```

### 6.4 实战案例：数据分析 Agent

```java
@RestController
@RequestMapping("/api/agent/data-analysis")
public class DataAnalysisAgentController {
    
    private final ChatClient chatClient;
    
    @PostMapping("/analyze")
    public ResponseEntity<AnalysisResult> analyze(
            @RequestBody AnalysisRequest request) {
        
        // 1. 分析需求
        String analysis = chatClient.prompt()
                .system("""
                    你是数据分析专家。
                    分析提供的数据集，提供洞察。
                    """)
                .user("""
                    数据集：{data}
                    
                    请分析：
                    1. 数据概览
                    2. 关键趋势
                    3. 异常检测
                    4. 建议
                    """, Map.of("data", request.data()))
                .call()
                .content();
        
        return ResponseEntity.ok(new AnalysisResult(analysis));
    }
}

record AnalysisRequest(String data) {}
record AnalysisResult(String insights) {}
```

## 7. 生产级应用开发

### 7.1 错误处理与重试

```java
@Service
public class ResilientChatService {
    
    private final ChatClient chatClient;
    
    @Retryable(
            value = {ApiException.class},
            maxAttempts = 3,
            backoff = @Backoff(delay = 1000)
    )
    public String chatWithRetry(String message) {
        return chatClient.prompt()
                .user(message)
                .call()
                .content();
    }
    
    @Recover
    public String recoverFromRetry(ApiException e, String message) {
        log.error("AI 服务调用失败：{}", e.getMessage());
        return "抱歉，AI 服务暂时不可用，请稍后重试。";
    }
}
```

### 7.2 限流与降级

```java
@Service
public class RateLimitedChatService {
    
    private final ChatClient chatClient;
    private final RateLimiter rateLimiter;
    
    public String chat(String message) {
        // 限流检查
        if (!rateLimiter.tryAcquire()) {
            return "请求过于频繁，请稍后重试";
        }
        
        return chatClient.prompt()
                .user(message)
                .call()
                .content();
    }
}

// 降级策略
@Service
public class FallbackChatService {
    
    private final ChatClient primaryClient;
    private final ChatClient fallbackClient;
    
    public String chat(String message) {
        try {
            return primaryClient.prompt()
                    .user(message)
                    .call()
                    .content();
        } catch (Exception e) {
            log.warn("主服务失败，使用备用服务：{}", e.getMessage());
            
            return fallbackClient.prompt()
                    .user(message)
                    .call()
                    .content();
        }
    }
}
```

### 7.3 监控与日志

```java
@Configuration
public class ObservabilityConfig {
    
    @Bean
    public MicrometerObservationRegistry observationRegistry(MeterRegistry meterRegistry) {
        return new MicrometerObservationRegistry(meterRegistry);
    }
}

@Service
public class MonitoredChatService {
    
    private final ChatClient chatClient;
    private final MeterRegistry meterRegistry;
    
    public String chat(String message) {
        Timer.Sample sample = Timer.start(meterRegistry);
        
        try {
            String response = chatClient.prompt()
                    .user(message)
                    .call()
                    .content();
            
            // 记录成功指标
            meterRegistry.counter("ai.chat.success").increment();
            
            return response;
        } catch (Exception e) {
            // 记录失败指标
            meterRegistry.counter("ai.chat.failure").increment();
            throw e;
        } finally {
            // 记录耗时
            sample.stop(Timer.builder("ai.chat.duration")
                    .register(meterRegistry));
        }
    }
}
```

### 7.4 缓存策略

```java
@Service
public class CachedChatService {
    
    private final ChatClient chatClient;
    
    @Cacheable(value = "chatResponses", key = "#message")
    public String chatWithCache(String message) {
        return chatClient.prompt()
                .user(message)
                .call()
                .content();
    }
}

@Configuration
@EnableCaching
public class CacheConfig {
    
    @Bean
    public CacheManager cacheManager() {
        CaffeineCacheManager cacheManager = new CaffeineCacheManager("chatResponses");
        cacheManager.setCaffeine(Caffeine.newBuilder()
                .maximumSize(1000)
                .expireAfterWrite(1, TimeUnit.HOURS));
        return cacheManager;
    }
}
```

### 7.5 安全与权限

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {
    
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(authorize -> authorize
                .requestMatchers("/api/ai/**").authenticated()
                .anyRequest().permitAll()
            )
            .oauth2ResourceServer(oauth2 -> oauth2
                .jwt(Customizer.withDefaults())
            );
        
        return http.build();
    }
}

@RestController
@RequestMapping("/api/ai")
public class SecureChatController {
    
    private final ChatService chatService;
    
    @PostMapping("/chat")
    public ResponseEntity<ChatResponse> chat(
            @RequestBody ChatRequest request,
            Authentication authentication) {
        
        // 记录用户 ID
        String userId = authentication.getName();
        
        String response = chatService.chat(request.message(), userId);
        
        return ResponseEntity.ok(new ChatResponse(response));
    }
}
```

## 8. 最佳实践

### 8.1 Spring AI 2.0 迁移指南

**从 1.x 迁移到 2.0 的关键变化**：

| 变化项 | 1.x API | 2.0 API | 说明 |
|--------|---------|---------|------|
| **Starter 命名** | `spring-ai-openai-spring-boot-starter` | `spring-ai-starter-model-openai` | 统一命名规范 |
| **配置属性** | `spring.ai.openai.chat.options.model` | `spring.ai.openai.chat.model` | 移除 `.options` 前缀 |
| **工具回调** | `FunctionCallback` | `MethodToolCallback` | 新抽象层 |
| **工具参数** | `@Param` | `@ToolParam` | 专用注解 |
| **注册工具** | `.defaultFunctions()` | `.defaultTools()` | API 重命名 |
| **Options 复制** | `options.copy()` | `options.mutate()` | 严格不可变 |
| **工具执行控制** | `internalToolExecutionEnabled` | `AdvisorParams.toolCallingAdvisorAutoRegister(false)` | Advisor 架构 |
| **版本管理** | 每个依赖指定版本 | 使用 `spring-ai-bom` | 推荐 BOM |

### 8.2 性能优化

```java
// 1. 使用流式输出
@GetMapping(value = "/chat/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
public Flux<String> streamingChat(@RequestParam String message) {
    return chatClient.prompt()
            .user(message)
            .stream()
            .content();
}

// 2. 批量处理
public List<String> batchChat(List<String> messages) {
    return messages.parallelStream()
            .map(msg -> chatClient.prompt()
                    .user(msg)
                    .call()
                    .content())
            .toList();
}

// 3. 连接池配置
spring:
  ai:
    openai:
      base-url: https://api.openai.com
      # 2.0 配置属性（移除 .options 前缀）
      connection-timeout: 5000
      read-timeout: 30000
```

### 8.3 Spring AI 2.0 新特性利用

**1. 使用 Tool Search Advisor 优化大规模工具场景**：

```java
// 当有 50+ 工具时，启用 Tool Search Advisor
// 自动根据问题选择最相关的 5-10 个工具
spring:
  ai:
    chat:
      client:
        tool-search-advisor:
          enabled: true
          tool-index-type: lucene  // 或 vector（需要 VectorStore）
```

**2. 利用增强的 Null-safety**：

```java
// 2.0 全面引入 @Nullable 和 @NonNull
@Service
public class SafeChatService {
    
    public @NonNull String chat(@NonNull String message) {
        // 编译期 null-safety 检查
        return chatClient.prompt()
                .user(message)
                .call()
                .content();
    }
}
```

**3. 使用 Jackson 3 提升 JSON 性能**：

```java
// 2.0 迁移至 Jackson 3.x
// 新的 JSON 工具类
JsonHelper jsonHelper = new JsonHelper();

// 序列化
String json = jsonHelper.toJson(myObject);

// 反序列化
MyObject obj = jsonHelper.fromJson(json, MyObject.class);
```

### 8.4 MCP（Model Context Protocol）支持

Spring AI 2.0 原生支持 MCP 协议：

```xml
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-mcp</artifactId>
</dependency>
```

```java
// 使用 @McpTool 注解定义 MCP 工具
@McpTool(description = "查询数据库")
public String queryDatabase(
        @McpToolParam("sql") String sql) {
    return databaseService.execute(sql);
}

// MCP Client 连接外部 MCP Server
@Bean
public McpSyncClient mcpClient() {
    return McpClient.builder()
            .transport(new StdioClientTransport(
                    Command.of("npx", "-y", "@modelcontextprotocol/server-fetch")))
            .build();
}
```

### 8.5 成本控制

```java
@Service
public class CostTrackingService {
    
    private final AtomicDouble totalCost = new AtomicDouble(0.0);
    
    public String chatWithCostTracking(String message) {
        ChatResponse response = chatClient.prompt()
                .user(message)
                .call()
                .chatResponse();
        
        // 计算成本
        double cost = calculateCost(
                response.getMetadata().getUsage().getPromptTokens(),
                response.getMetadata().getUsage().getCompletionTokens()
        );
        
        totalCost.addAndGet(cost);
        
        log.info("本次成本：${}, 累计成本：${}", cost, totalCost.get());
        
        return response.getResult().getOutput().getContent();
    }
    
    private double calculateCost(int promptTokens, int completionTokens) {
        // GPT-4o 定价
        double promptCost = promptTokens / 1_000_000.0 * 5.0;
        double completionCost = completionTokens / 1_000_000.0 * 15.0;
        return promptCost + completionCost;
    }
}
```

### 8.6 测试策略

```java
@SpringBootTest
class ChatServiceTest {
    
    @Autowired
    private ChatService chatService;
    
    @Test
    void testSimpleChat() {
        String response = chatService.chat("你好");
        
        assertNotNull(response);
        assertFalse(response.isEmpty());
    }
    
    @Test
    void testToolCalling() {
        String response = chatService.chat("北京天气怎么样？");
        
        assertNotNull(response);
        assertTrue(response.contains("北京"));
    }
}
```

### 8.7 部署指南

```dockerfile
# Dockerfile
FROM eclipse-temurin:21-jre

WORKDIR /app

COPY target/*.jar app.jar

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "app.jar"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  spring-ai-app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - SPRING_AI_OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - chroma
  
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chroma-data:/chroma/chroma

volumes:
  chroma-data:
```

## 9. 参考资料

### 9.1 官方文档

- Spring AI 官方: https://spring.io/projects/spring-ai
- Spring AI 2.0 文档: https://docs.spring.io/spring-ai/reference/
- Spring AI 2.0.0 GA 发布公告: https://spring.io/blog/2026/06/12/spring-ai-2-0-0-GA-available-now
- GitHub: https://github.com/spring-projects/spring-ai
- 升级指南: https://docs.spring.io/spring-ai/reference/upgrade-notes.html

### 9.2 教程与文章

- Spring Boot 4 and Spring AI 2.0: The New Java AI Stack: https://medium.com/@chrisvanbreeden/spring-boot-4-and-spring-ai-2-0-the-new-java-ai-stack-58e9577e7919
- Spring AI 2.0 + Spring Boot 4 Guide [2026]: https://usama.codes/blog/spring-ai-2-spring-boot-4-guide
- Spring AI 2.0 接入 AI 大模型完全指南: https://ofox.ai/zh/blog/spring-ai-java-llm-api-complete-guide-2026/
- Spring AI 2.0 + MCP: Building a Tool-Calling Agent in 50 Lines: https://www.javacodegeeks.com/2026/06/spring-ai-2-0-mcp-building-a-tool-calling-agent-in-50-lines.html
- Spring AI tutorial: How to develop AI agents with Spring: https://www.infoworld.com/article/4150199/spring-ai-tutorial-building-ai-agents-with-spring-ai.html
- Spring Boot 4, Spring AI, and AI-First Java Development: https://www.javacodegeeks.com/2026/03/spring-boot-4-spring-ai-and-ai-first-java-development.html

### 9.3 示例项目

- Spring AI Samples: https://github.com/spring-ai-community/spring-ai-samples
- Spring Initializr (AI 支持): https://start.spring.io
