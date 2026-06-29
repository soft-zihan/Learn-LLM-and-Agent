package com.example.agent;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 提示词与上下文工程服务
 * 
 * 对应教程：05-提示词与上下文工程
 * 
 * 实现功能：
 * 1. CLEAR 提示词原则（教程05.1）
 * 2. 上下文组装技术（教程05.2）
 * 3. 上下文窗口管理：裁剪、压缩、优先级（教程05.3）
 * 4. 动态上下文注入（教程05.4）
 * 5. RAG 场景上下文注入（教程05.5）
 * 6. 长任务上下文保持（教程05.6）
 */
@Service
public class PromptEngineeringService {

    private final ChatClient chatClient;

    public PromptEngineeringService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    // ═══════════════════════════════════════════════════════
    // 教程05.1：CLEAR 提示词原则
    // ═══════════════════════════════════════════════════════

    /**
     * 创建符合 CLEAR 原则的提示词
     * 
     * C - Clear（清晰）：意图明确，无歧义
     * L - Limited（聚焦）：一次只做一件事
     * E - Explicit（显式）：明确输出格式和要求
     * A - Adapted（适配）：根据模型能力调整
     * R - Repeatable（可重复）：相同输入产生稳定输出
     */
    public String createClearPrompt(String role, String capabilities, 
                                     String constraints, String outputFormat, 
                                     String examples) {
        return """
            你是一个%s。
            
            ## 能力
            %s
            
            ## 约束
            %s
            
            ## 输出格式
            %s
            
            ## 示例
            %s
            """.formatted(role, capabilities, constraints, outputFormat, examples);
    }

    /**
     * 使用 CLEAR 提示词进行对话
     */
    public String chatWithClearPrompt(String role, String capabilities,
                                       String constraints, String outputFormat,
                                       String examples, String userMessage) {
        String systemPrompt = createClearPrompt(role, capabilities, constraints, 
                                                outputFormat, examples);
        return chatClient.prompt()
                .system(systemPrompt)
                .user(userMessage)
                .call()
                .content();
    }

    // ═══════════════════════════════════════════════════════
    // 教程05.2：上下文组装技术
    // ═══════════════════════════════════════════════════════

    /**
     * 组装多层次的上下文
     * 
     * 完整上下文结构：
     * ├── System Context（系统级）：角色定义、行为规则
     * ├── User Context（用户级）：用户偏好、历史交互
     * ├── Session Context（会话级）：对话历史、中间结果
     * └── Tool Context（工具级）：工具返回结果、外部数据
     */
    public List<Message> assembleContext(
            String systemPrompt,
            Map<String, Object> userProfile,
            List<Message> conversationHistory,
            List<String> retrievedDocs,
            String currentMessage) {
        
        List<Message> context = new ArrayList<>();
        
        // 1. System Context
        context.add(new SystemMessage(systemPrompt));
        
        // 2. User Context（用户画像）
        if (userProfile != null && !userProfile.isEmpty()) {
            String profileStr = userProfile.entrySet().stream()
                    .map(e -> "%s: %s".formatted(e.getKey(), e.getValue()))
                    .collect(Collectors.joining("\n"));
            context.add(new SystemMessage("用户画像：\n" + profileStr));
        }
        
        // 3. Session Context（最近5轮对话，即10条消息）
        if (conversationHistory != null && !conversationHistory.isEmpty()) {
            int startIdx = Math.max(0, conversationHistory.size() - 10);
            context.addAll(conversationHistory.subList(startIdx, conversationHistory.size()));
        }
        
        // 4. Tool Context（检索到的文档）
        if (retrievedDocs != null && !retrievedDocs.isEmpty()) {
            StringBuilder docsSb = new StringBuilder();
            for (int i = 0; i < retrievedDocs.size(); i++) {
                if (i > 0) docsSb.append("\n\n");
                docsSb.append("文档").append(i + 1).append("：").append(retrievedDocs.get(i));
            }
            context.add(new SystemMessage("参考文档：\n" + docsSb.toString()));
        }
        
        // 5. 当前用户消息
        context.add(new UserMessage(currentMessage));
        
        return context;
    }

    /**
     * 使用组装的上下文进行对话
     */
    public String chatWithContext(List<Message> context) {
        return chatClient.prompt()
                .messages(context)
                .call()
                .content();
    }

    // ═══════════════════════════════════════════════════════
    // 教程05.3：上下文窗口管理策略
    // ═══════════════════════════════════════════════════════

    /**
     * 策略1：上下文裁剪
     * 保留系统消息和最近N条消息
     */
    public List<Message> trimContext(List<Message> messages, int maxRecentMessages) {
        if (messages == null || messages.isEmpty()) {
            return messages;
        }
        
        // 分离系统消息和其他消息
        List<Message> systemMessages = messages.stream()
                .filter(m -> m instanceof SystemMessage)
                .toList();
        
        List<Message> otherMessages = messages.stream()
                .filter(m -> !(m instanceof SystemMessage))
                .toList();
        
        // 保留最近的N条消息
        int startIdx = Math.max(0, otherMessages.size() - maxRecentMessages);
        List<Message> recentMessages = otherMessages.subList(startIdx, otherMessages.size());
        
        // 合并：系统消息 + 最近的消息
        List<Message> result = new ArrayList<>(systemMessages);
        result.addAll(recentMessages);
        
        return result;
    }

    /**
     * 策略2：上下文压缩（摘要）
     * 使用 LLM 将长对话压缩为摘要
     */
    public String compressContext(List<Message> messages) {
        if (messages == null || messages.isEmpty()) {
            return "";
        }
        
        // 构建对话文本
        String conversationText = messages.stream()
                .map(m -> "%s: %s".formatted(m.getMessageType(), m.getText()))
                .collect(Collectors.joining("\n"));
        
        // 使用 LLM 生成摘要
        String summaryPrompt = """
            请总结以下对话的关键信息，保持简洁（200字以内）：
            
            %s
            
            摘要：
            """.formatted(conversationText);
        
        return chatClient.prompt()
                .system("你是一个对话摘要专家，擅长提取关键信息。")
                .user(summaryPrompt)
                .call()
                .content();
    }

    /**
     * 策略3：上下文优先级管理
     * 为不同类型的消息分配优先级
     */
    public List<Message> prioritizeContext(List<Message> messages, int maxTokens) {
        if (messages == null || messages.isEmpty()) {
            return messages;
        }
        
        // 定义优先级
        Map<Class<?>, Integer> priorities = Map.of(
                SystemMessage.class, 100,      // 最高优先级
                AssistantMessage.class, 40     // AI 回复
                // UserMessage 默认优先级 60
        );
        
        // 按优先级排序（降序）
        List<Message> sortedMessages = messages.stream()
                .sorted((m1, m2) -> {
                    int p1 = priorities.getOrDefault(m1.getClass(), 60);
                    int p2 = priorities.getOrDefault(m2.getClass(), 60);
                    return Integer.compare(p2, p1); // 降序
                })
                .toList();
        
        // 选择消息直到达到 token 限制
        List<Message> selected = new ArrayList<>();
        int totalTokens = 0;
        
        for (Message msg : sortedMessages) {
            int msgTokens = msg.getText().length(); // 简化：按字符数估算
            if (totalTokens + msgTokens <= maxTokens) {
                selected.add(msg);
                totalTokens += msgTokens;
            } else {
                break;
            }
        }
        
        return selected;
    }

    // ═══════════════════════════════════════════════════════
    // 教程05.4：动态上下文注入
    // ═══════════════════════════════════════════════════════

    /**
     * 动态注入上下文变量到系统消息中
     */
    public List<Message> injectDynamicContext(
            String baseSystemPrompt,
            Map<String, String> contextVars,
            String userMessage) {
        
        List<Message> messages = new ArrayList<>();
        
        // 1. 基础系统提示
        messages.add(new SystemMessage(baseSystemPrompt));
        
        // 2. 注入动态上下文变量
        if (contextVars != null && !contextVars.isEmpty()) {
            String contextStr = contextVars.entrySet().stream()
                    .map(e -> "- %s: %s".formatted(e.getKey(), e.getValue()))
                    .collect(Collectors.joining("\n"));
            
            messages.add(new SystemMessage("## 当前上下文\n" + contextStr));
        }
        
        // 3. 用户消息
        messages.add(new UserMessage(userMessage));
        
        return messages;
    }

    /**
     * 使用动态上下文进行对话
     */
    public String chatWithDynamicContext(
            String baseSystemPrompt,
            Map<String, String> contextVars,
            String userMessage) {
        
        List<Message> context = injectDynamicContext(baseSystemPrompt, contextVars, userMessage);
        return chatWithContext(context);
    }

    // ═══════════════════════════════════════════════════════
    // 教程05.5：RAG 场景上下文注入
    // ═══════════════════════════════════════════════════════

    /**
     * 构建 RAG 提示词
     */
    public String buildRagPrompt(List<String> documents, String question) {
        StringBuilder docsSb = new StringBuilder();
        for (int i = 0; i < documents.size(); i++) {
            if (i > 0) docsSb.append("\n\n");
            docsSb.append("## 文档 ").append(i + 1).append("\n").append(documents.get(i));
        }
        String docsStr = docsSb.toString();
        
        return """
            请根据以下参考文档回答问题。
            
            %s
            
            规则：
            1. 仅使用参考资料中的信息
            2. 如果资料不足，回答"根据现有资料无法确定"
            3. 引用来源
            4. 不要编造信息
            
            问题：%s
            """.formatted(docsStr, question);
    }

    /**
     * RAG 问答
     */
    public String ragQuery(List<String> documents, String question) {
        String ragPrompt = buildRagPrompt(documents, question);
        
        return chatClient.prompt()
                .system("你是知识问答助手，基于参考资料回答问题。")
                .user(ragPrompt)
                .call()
                .content();
    }

    // ═══════════════════════════════════════════════════════
    // 教程05.6：长任务中的上下文保持
    // ═══════════════════════════════════════════════════════

    /**
     * 长任务上下文管理
     * 当对话过长时，自动摘要并重建上下文
     */
    public String chatLongTask(
            List<Message> messageHistory,
            String newUserMessage,
            int maxMessages) {
        
        List<Message> context = new ArrayList<>();
        
        // 如果上下文太长，先摘要
        if (messageHistory.size() > maxMessages) {
            // 提取需要摘要的消息
            List<Message> toSummarize = messageHistory.subList(
                    0, messageHistory.size() - 5); // 保留最近5条
            
            // 生成摘要
            String summary = compressContext(toSummarize);
            
            // 重建上下文
            context.add(new SystemMessage("对话摘要：" + summary));
            context.addAll(messageHistory.subList(
                    Math.max(0, messageHistory.size() - 5), 
                    messageHistory.size()));
        } else {
            context.addAll(messageHistory);
        }
        
        // 添加新消息
        context.add(new UserMessage(newUserMessage));
        
        return chatWithContext(context);
    }
}
