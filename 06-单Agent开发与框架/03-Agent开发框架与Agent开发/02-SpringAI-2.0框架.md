# Spring AI 2.0 开发实战

> 📅 **更新时间**: 2026-06-29

---

## 目录

- [1. Spring AI 2.0 核心变化](#1-spring-ai-20-核心变化)
- [2. ChatClient API](#2-chatclient-api)
- [3. Tool Calling 实战](#3-tool-calling-实战)
- [4. Agent 设计模式](#4-agent-设计模式)
- [5. 记忆与上下文](#5-记忆与上下文)

---

## 1. Spring AI 2.0 核心变化

### 1.1 与 1.0 的区别

Spring AI 2.0（2026年GA）完成了从"模型客户端"到"AI原生运行时"的蜕变：

| 特性 | Spring AI 1.0 | Spring AI 2.0 |
|------|--------------|--------------|
| 核心 API | ChatModel | ChatClient |
| 工具调用 | 手动管理 | ToolCallingAdvisor 自动管理 |
| Agent 支持 | 无 | 原生支持（loop、advisor 链） |
| 架构 | 简单客户端 | 可组合的 Agentic 架构 |

### 1.2 ChatClient 统一入口

```java
// Spring AI 2.0：ChatClient 统一入口
ChatClient chatClient = ChatClient.builder(chatModel)
    .defaultSystem("你是有用的助手")
    .build();

// 发送请求
String response = chatClient.prompt()
    .user("解释 Spring AI 2.0 的新特性")
    .call()
    .content();
```

### 1.3 ToolCallingAdvisor 接管工具执行

```java
// 2.0：ToolCallingAdvisor 自动管理工具调用
ChatClient chatClient = ChatClient.builder(chatModel)
    .defaultAdvisors(
        new ToolCallingAdvisor()  // 自动处理工具调用循环
    )
    .defaultTools("searchWeb", "calculate")  // 注册工具
    .build();
```

---

## 2. ChatClient API

### 2.1 依赖配置

```xml
<dependencies>
    <!-- Spring AI 2.0 -->
    <dependency>
        <groupId>org.springframework.ai</groupId>
        <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
        <version>2.0.0</version>
    </dependency>
    
    <!-- Spring Boot 3.x -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
</dependencies>
```

### 2.2 基础 ChatClient（完整代码）

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@SpringBootApplication
@RestController
public class SpringAiApplication {

    private final ChatClient chatClient;

    public SpringAiApplication(ChatClient.Builder chatClientBuilder) {
        // 构建 ChatClient
        this.chatClient = chatClientBuilder
            .defaultSystem("你是专业的 Java 开发助手")
            .build();
    }

    @GetMapping("/chat")
    public String chat(@RequestParam String message) {
        // 发送请求
        return chatClient.prompt()
            .user(message)
            .call()
            .content();
    }

    public static void main(String[] args) {
        SpringApplication.run(SpringAiApplication.class, args);
    }
}
```

**application.yml 配置**：

```yaml
spring:
  ai:
    openai:
      api-key: ${OPENAI_API_KEY}
      chat:
        options:
          model: gpt-4o
          temperature: 0.7
```

### 2.3 流式输出（完整代码）

```java
import reactor.core.publisher.Flux;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class StreamingController {

    private final ChatClient chatClient;

    public StreamingController(ChatClient.Builder chatClientBuilder) {
        this.chatClient = chatClientBuilder.build();
    }

    @GetMapping("/chat/stream")
    public Flux<String> streamChat(@RequestParam String message) {
        // 流式输出
        return chatClient.prompt()
            .user(message)
            .stream()
            .content();
    }
}
```

**前端调用（JavaScript）**：

```javascript
const eventSource = new EventSource('/chat/stream?message=解释Java');

eventSource.onmessage = (event) => {
    console.log(event.data);  // 逐字输出
};
```

---

## 3. Tool Calling 实战

### 3.1 定义工具（@Tool 注解）

```java
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;

@Component
public class SearchTool {

    @Tool(description = "搜索网络获取最新信息")
    public String searchWeb(
        @ToolParam(description = "搜索关键词") String query) {
        
        // 实际应调用搜索 API
        return "搜索结果：" + query;
    }
}

@Component
public class CalculatorTool {

    @Tool(description = "执行数学计算")
    public String calculate(
        @ToolParam(description = "数学表达式") String expression) {
        
        try {
            // 注意：生产环境应使用安全的表达式解析器
            return "计算结果：" + eval(expression);
        } catch (Exception e) {
            return "计算错误：" + e.getMessage();
        }
    }
    
    private double eval(String expr) {
        // 简化实现
        return 0.0;
    }
}
```

### 3.2 工具注册与发现

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.method.MethodToolCallbackProvider;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class ToolConfig {

    @Bean
    public ChatClient chatClient(
        ChatClient.Builder builder,
        SearchTool searchTool,
        CalculatorTool calculatorTool) {
        
        return builder
            .defaultTools(
                new MethodToolCallbackProvider(searchTool, calculatorTool)
            )
            .build();
    }
}
```

### 3.3 完整示例（完整代码）

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.ToolCallingAdvisor;
import org.springframework.ai.tool.method.MethodToolCallbackProvider;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@SpringBootApplication
@RestController
public class AgentApplication {

    private final ChatClient chatClient;

    public AgentApplication(
        ChatClient.Builder chatClientBuilder,
        SearchTool searchTool,
        CalculatorTool calculatorTool) {
        
        this.chatClient = chatClientBuilder
            .defaultSystem("""
                你是有用的助手。
                
                你可以使用以下工具：
                - searchWeb：搜索网络获取信息
                - calculate：执行数学计算
                
                根据用户需求自动选择合适的工具。
                """)
            .defaultAdvisors(
                new ToolCallingAdvisor()  // 自动处理工具调用循环
            )
            .defaultTools(
                new MethodToolCallbackProvider(searchTool, calculatorTool)
            )
            .build();
    }

    @GetMapping("/agent")
    public String agent(@RequestParam String message) {
        return chatClient.prompt()
            .user(message)
            .call()
            .content();
    }

    public static void main(String[] args) {
        SpringApplication.run(AgentApplication.class, args);
    }
}
```

---

## 4. Agent 设计模式

### 4.1 ReAct Agent（完整代码）

Spring AI 2.0 通过 Advisor 链实现 ReAct：

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.ToolCallingAdvisor;
import org.springframework.ai.tool.method.MethodToolCallbackProvider;

// 构建 ReAct Agent
ChatClient reactAgent = ChatClient.builder(chatModel)
    .defaultSystem("""
        你是一个 ReAct Agent。
        
        工作流程：
        1. 思考：分析问题
        2. 行动：使用工具
        3. 观察：查看工具结果
        4. 重复，直到得出答案
        
        可用工具：
        - searchWeb
        - calculate
        """)
    .defaultAdvisors(
        new ToolCallingAdvisor()  // 自动循环调用工具
    )
    .defaultTools(
        new MethodToolCallbackProvider(searchTool, calculatorTool)
    )
    .build();

// 使用
String result = reactAgent.prompt()
    .user("2024年诺贝尔物理学奖是谁？如果是奇数年，计算年份乘以2")
    .call()
    .content();
```

### 4.2 多步骤工作流（完整代码）

使用多个 Advisor 实现复杂工作流：

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.*;

// 多步骤工作流
ChatClient workflowAgent = ChatClient.builder(chatModel)
    .defaultAdvisors(
        // 1. 验证输入
        new InputValidationAdvisor(),
        
        // 2. 工具调用
        new ToolCallingAdvisor(),
        
        // 3. 输出审核
        new OutputAuditAdvisor()
    )
    .defaultTools(tools)
    .build();

// 自定义 Advisor
public class InputValidationAdvisor implements Advisor {
    
    @Override
    public Object around(ProceedingJoinPoint pjp) throws Throwable {
        ChatClientRequest request = (ChatClientRequest) pjp.getArgs()[0];
        
        // 验证输入
        String userInput = request.prompt().getUserMessage().getText();
        if (userInput.length() > 10000) {
            throw new IllegalArgumentException("输入过长");
        }
        
        // 继续执行
        return pjp.proceed();
    }
}
```

### 4.3 人类介入（完整代码）

```java
import org.springframework.ai.chat.client.advisor.AbstractAdvisor;

public class HumanApprovalAdvisor extends AbstractAdvisor {
    
    @Override
    protected Object doAround(ProceedingJoinPoint pjp) throws Throwable {
        ChatClientResponse response = (ChatClientResponse) pjp.proceed();
        
        // 检查是否需要人类审批
        String content = response.result().getOutput().getText();
        if (requiresApproval(content)) {
            // 等待人类审批
            String approval = waitForHumanApproval(content);
            
            if (!approval.equals("approved")) {
                return ChatClientResponse.from("操作被拒绝");
            }
        }
        
        return response;
    }
    
    private boolean requiresApproval(String content) {
        // 检测敏感操作
        return content.contains("删除") || content.contains("DROP");
    }
    
    private String waitForHumanApproval(String content) {
        // 实际应通过 WebSocket 或 API 等待人类输入
        return "approved";
    }
}
```

---

## 5. 记忆与上下文

### 5.1 ChatMemory 使用

```java
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.InMemoryChatMemory;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;

// 创建记忆
ChatMemory chatMemory = new InMemoryChatMemory();

// 构建带记忆的 ChatClient
ChatClient chatClient = ChatClient.builder(chatModel)
    .defaultAdvisors(
        new MessageChatMemoryAdvisor(chatMemory, "user_123", 10)  // 保留最近 10 条
    )
    .build();

// 使用（自动记住对话历史）
String response1 = chatClient.prompt()
    .user("我喜欢 Python")
    .call()
    .content();

String response2 = chatClient.prompt()
    .user("我喜欢什么编程语言？")
    .call()
    .content();
// Agent 会回答："你喜欢 Python"
```

### 5.2 向量数据库集成

```java
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.PgVectorStore;
import org.springframework.ai.document.Document;
import org.springframework.ai.chat.client.advisor.QuestionAnswerAdvisor;

// 创建向量存储
VectorStore vectorStore = new PgVectorStore(dataSource);

// 添加文档
vectorStore.add(List.of(
    new Document("Spring AI 2.0 引入了 ChatClient API"),
    new Document("ToolCallingAdvisor 自动管理工具调用")
));

// 构建 RAG ChatClient
ChatClient ragClient = ChatClient.builder(chatModel)
    .defaultAdvisors(
        new QuestionAnswerAdvisor(vectorStore)  // RAG 顾问
    )
    .build();

// 使用
String response = ragClient.prompt()
    .user("Spring AI 2.0 有什么新特性？")
    .call()
    .content();
// Agent 会从向量库检索相关文档来回答
```

### 5.3 RAG 实现（完整代码）

完整的 RAG 应用：

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.QuestionAnswerAdvisor;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@SpringBootApplication
@RestController
public class RagApplication {

    private final ChatClient ragClient;

    public RagApplication(
        ChatClient.Builder chatClientBuilder,
        VectorStore vectorStore) {
        
        this.ragClient = chatClientBuilder
            .defaultAdvisors(
                new QuestionAnswerAdvisor(vectorStore)
            )
            .build();
    }

    @PostMapping("/documents")
    public String addDocuments(@RequestBody List<String> documents) {
        // 添加文档到向量库
        vectorStore.add(
            documents.stream()
                .map(Document::new)
                .toList()
        );
        return "文档已添加";
    }

    @GetMapping("/rag")
    public String rag(@RequestParam String question) {
        // RAG 查询
        return ragClient.prompt()
            .user(question)
            .call()
            .content();
    }

    @Bean
    public VectorStore vectorStore(EmbeddingModel embeddingModel) {
        return new SimpleVectorStore(embeddingModel);
    }

    public static void main(String[] args) {
        SpringApplication.run(RagApplication.class, args);
    }
}
```
