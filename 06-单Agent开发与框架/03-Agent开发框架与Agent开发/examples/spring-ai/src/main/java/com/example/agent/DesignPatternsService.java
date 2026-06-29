package com.example.agent;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * 智能体设计模式服务
 * 
 * 对应教程：07-智能体设计模式
 * 
 * 实现功能：
 * 1. Reflection 模式（自我反思与改进）- 教程07.1
 * 2. 工具链设计 - 教程07.2
 * 3. 验证与护栏 - 教程07.3
 */
@Service
public class DesignPatternsService {

    private final ChatClient chatClient;
    private final ChatClient strongChatClient; // 强模型用于生成
    private final ChatClient weakChatClient;   // 弱模型用于评估

    public DesignPatternsService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
        // 实际项目中可以配置不同模型的 ChatClient
        this.strongChatClient = builder.build();
        this.weakChatClient = builder.build();
    }

    // ═══════════════════════════════════════════════════════
    // 教程07.1：Reflection 模式（反思循环）
    // ═══════════════════════════════════════════════════════

    /**
     * Reflection 模式执行结果
     */
    public record ReflectionResult(
            String draft,           // 初稿
            String critique,        // 评审意见
            String improvedDraft,   // 改进稿
            int iterations,         // 迭代次数
            int score              // 最终评分
    ) {}

    /**
     * 质量评分模型（对应 Python 的 QualityScore）
     */
    public record QualityScore(
            int score,              // 质量评分 1-10
            List<String> issues,    // 问题列表
            List<String> suggestions // 改进建议
    ) {}

    /**
     * 执行 Reflection 模式
     * 
     * 流程：
     * 生成草稿 → 评审质量 → 改进草稿 → 循环直到质量达标或达到最大迭代次数
     * 
     * @param task 任务描述
     * @param maxIterations 最大迭代次数
     * @return Reflection 执行结果
     */
    public ReflectionResult executeReflection(String task, int maxIterations) {
        String draft = "";
        String critique = "";
        String improvedDraft = "";
        int iterations = 0;
        int score = 0;

        for (int i = 0; i < maxIterations; i++) {
            iterations = i + 1;

            // 1. 生成器节点：撰写初稿或改进稿
            if (i == 0) {
                draft = generateDraft(task);
            } else {
                draft = improveDraft(task, improvedDraft, critique);
            }

            // 2. 评审节点：评估质量
            QualityScore qualityScore = critiqueDraft(task, draft);
            critique = "评分：%d/10\n问题：%s\n建议：%s".formatted(
                    qualityScore.score(),
                    String.join("、", qualityScore.issues()),
                    String.join("、", qualityScore.suggestions())
            );
            score = qualityScore.score();

            // 3. 改进节点：根据评审意见改进
            improvedDraft = refineDraft(draft, critique);

            // 4. 检查是否继续迭代
            if (score >= 8) {
                // 质量达标，提前结束
                break;
            }
        }

        return new ReflectionResult(draft, critique, improvedDraft, iterations, score);
    }

    /**
     * 生成器：撰写初稿
     */
    private String generateDraft(String task) {
        String prompt = """
            你是一个报告生成专家。请根据用户要求撰写高质量的报告。
            
            要求：
            1. 内容详实、结构清晰
            2. 包含关键发现和分析
            3. 提供建设性的建议
            
            任务：%s
            
            请生成详细的初稿。
            """.formatted(task);

        return chatClient.prompt()
                .system(prompt)
                .user("请开始撰写报告")
                .call()
                .content();
    }

    /**
     * 评审器：评估草稿质量
     */
    private QualityScore critiqueDraft(String task, String draft) {
        String prompt = """
            你是一个严格的评审专家。请评估以下报告的质量。
            
            原始任务：%s
            
            报告草稿：
            %s
            
            评估维度（0-10分）：
            1. 准确性：信息是否准确
            2. 完整性：是否覆盖所有要点
            3. 逻辑性：推理是否严密
            4. 可读性：表达是否清晰
            
            请以以下 JSON 格式返回评估结果：
            {
                "score": 0-10,
                "issues": ["问题1", "问题2"],
                "suggestions": ["建议1", "建议2"]
            }
            
            请严格评估，给出具体问题和改进建议。
            """.formatted(task, draft);

        String response = chatClient.prompt()
                .system(prompt)
                .user("请评估报告质量")
                .call()
                .content();

        // 简化解析：实际应使用 JSON 解析库
        // 这里提取关键信息
        int score = extractScore(response);
        List<String> issues = extractList(response, "issues");
        List<String> suggestions = extractList(response, "suggestions");

        return new QualityScore(score, issues, suggestions);
    }

    /**
     * 改进器：根据评审意见改进草稿
     */
    private String refineDraft(String draft, String critique) {
        String prompt = """
            你是一个改进专家。请根据评审意见改进报告。
            
            原草稿：
            %s
            
            评审意见：
            %s
            
            请生成改进后的版本，解决所有指出的问题。
            """.formatted(draft, critique);

        return chatClient.prompt()
                .system(prompt)
                .user("请改进报告")
                .call()
                .content();
    }

    /**
     * 改进草稿（迭代时使用）
     */
    private String improveDraft(String task, String currentDraft, String critique) {
        String prompt = """
            你是一个改进专家。根据以下评审意见改进报告。
            
            任务：%s
            
            当前草稿：
            %s
            
            评审意见：
            %s
            
            请生成改进后的版本。
            """.formatted(task, currentDraft, critique);

        return chatClient.prompt()
                .system(prompt)
                .user("请改进报告")
                .call()
                .content();
    }

    // ═══════════════════════════════════════════════════════
    // 教程07.2：工具链设计
    // ═══════════════════════════════════════════════════════

    /**
     * 工具链执行结果
     */
    public record ToolChainResult(
            List<String> results,
            boolean success,
            String message
    ) {}

    /**
     * 执行工具链（按顺序调用多个工具）
     * 
     * 流程：搜索 → 撰写报告 → 验证报告
     * 如果任何步骤失败，终止链
     * 
     * @param query 查询内容
     * @return 工具链执行结果
     */
    public ToolChainResult executeToolChain(String query) {
        List<String> results = new ArrayList<>();
        
        // 工具1：搜索信息
        String searchResult = searchInformation(query);
        results.add("搜索结果：" + searchResult);
        
        // 工具2：撰写报告
        String report = writeReport(searchResult);
        results.add("报告：" + report);
        
        // 工具3：验证报告
        String validation = validateReport(report);
        results.add("验证：" + validation);
        
        // 如果验证不合格，终止链
        if (validation.contains("不合格")) {
            return new ToolChainResult(results, false, "工具链终止：" + validation);
        }
        
        return new ToolChainResult(results, true, "工具链执行完成");
    }

    /**
     * 工具1：搜索信息
     */
    private String searchInformation(String query) {
        String prompt = """
            请搜索以下主题的信息：
            
            查询：%s
            
            请提供详细的搜索结果。
            """.formatted(query);

        return chatClient.prompt()
                .system(prompt)
                .user("请搜索相关信息")
                .call()
                .content();
    }

    /**
     * 工具2：撰写报告
     */
    private String writeReport(String content) {
        String prompt = """
            请根据以下内容撰写报告：
            
            %s
            
            报告要求：
            1. 包含关键发现
            2. 结构清晰
            3. 至少200字
            """.formatted(content);

        return chatClient.prompt()
                .system(prompt)
                .user("请撰写报告")
                .call()
                .content();
    }

    /**
     * 工具3：验证报告质量
     */
    private String validateReport(String report) {
        // 简单验证逻辑
        if (report.length() < 50) {
            return "不合格：报告太短";
        }
        if (!report.contains("关键发现")) {
            return "不合格：缺少关键发现";
        }
        return "合格";
    }

    // ═══════════════════════════════════════════════════════
    // 教程07.3：验证与护栏
    // ═══════════════════════════════════════════════════════

    /**
     * 输入验证结果
     */
    public record ValidationResult(
            boolean passed,
            String message
    ) {}

    /**
     * 输入验证护栏
     * 检查用户输入是否包含不安全内容
     */
    public ValidationResult validateInput(String userInput) {
        String lowerInput = userInput.toLowerCase();
        
        // 安全检查：敏感词列表
        String[] blockedWords = {"删除所有", "格式化", "密码", "黑客", "drop table", "delete from"};
        
        for (String word : blockedWords) {
            if (lowerInput.contains(word.toLowerCase())) {
                return new ValidationResult(
                        false,
                        "输入被拒绝：包含敏感词'%s'".formatted(word)
                );
            }
        }
        
        // 长度检查
        if (userInput.length() > 10000) {
            return new ValidationResult(false, "输入过长，请精简到10000字以内");
        }
        
        return new ValidationResult(true, "输入验证通过");
    }

    /**
     * 输出验证护栏
     * 检查 Agent 输出是否包含敏感信息
     */
    public ValidationResult validateOutput(String output) {
        String lowerOutput = output.toLowerCase();
        
        // 检查是否包含敏感信息
        String[] sensitiveWords = {"密码", "token", "secret", "password", "密钥"};
        
        for (String word : sensitiveWords) {
            if (lowerOutput.contains(word.toLowerCase())) {
                return new ValidationResult(
                        false,
                        "输出已过滤：包含敏感信息'%s'".formatted(word)
                );
            }
        }
        
        return new ValidationResult(true, "输出验证通过");
    }

    /**
     * 带护栏的对话
     * 先验证输入，再执行，最后验证输出
     */
    public String chatWithGuardrails(String userInput) {
        // 1. 输入验证
        ValidationResult inputValidation = validateInput(userInput);
        if (!inputValidation.passed()) {
            return "❌ " + inputValidation.message();
        }
        
        // 2. 执行对话
        String response = chatClient.prompt()
                .system("你是一个安全的AI助手。")
                .user(userInput)
                .call()
                .content();
        
        // 3. 输出验证
        ValidationResult outputValidation = validateOutput(response);
        if (!outputValidation.passed()) {
            return "⚠️ " + outputValidation.message();
        }
        
        return response;
    }

    // ═══════════════════════════════════════════════════════
    // 辅助方法
    // ═══════════════════════════════════════════════════════

    /**
     * 从响应中提取评分（简化实现）
     */
    private int extractScore(String response) {
        // 简化：查找 "score": X 模式
        try {
            String[] lines = response.split("\n");
            for (String line : lines) {
                if (line.contains("score")) {
                    String[] parts = line.split(":");
                    if (parts.length >= 2) {
                        return Integer.parseInt(parts[1].trim().replaceAll("[^0-9]", ""));
                    }
                }
            }
        } catch (Exception e) {
            // 默认返回中等评分
            return 5;
        }
        return 5;
    }

    /**
     * 从响应中提取列表（简化实现）
     */
    private List<String> extractList(String response, String key) {
        List<String> result = new ArrayList<>();
        // 简化：按行分割，提取包含关键词的行
        String[] lines = response.split("\n");
        boolean inList = false;
        
        for (String line : lines) {
            if (line.contains(key)) {
                inList = true;
                continue;
            }
            if (inList && line.contains("]")) {
                break;
            }
            if (inList && line.trim().startsWith("\"")) {
                String item = line.trim()
                        .replaceAll("^\"|\",?$", "")
                        .trim();
                if (!item.isEmpty()) {
                    result.add(item);
                }
            }
        }
        
        return result;
    }
}
