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
 * 循环与自演进服务
 * 
 * 对应教程：08-循环与自演进
 * 
 * 实现功能：
 * 1. 生成器-评估器-规划器架构（教程08.1）
 * 2. Loop 实战模式：代码审查循环（教程08.2）
 * 3. 理解债务防护（教程08.3）
 * 4. Token 成本优化（教程08.4）
 */
@Service
public class PlanExecutorService {

    private final ChatClient generatorClient;  // 强模型：生成
    private final ChatClient evaluatorClient;  // 弱模型：评估
    private final ChatClient plannerClient;    // 弱模型：规划
    private final ChatClient chatClient;       // 通用客户端

    public PlanExecutorService(ChatClient.Builder builder) {
        // 实际项目中可以配置不同成本的模型
        this.generatorClient = builder.build();  // GPT-4o 级别
        this.evaluatorClient = builder.build();  // GPT-4o-mini 级别
        this.plannerClient = builder.build();    // GPT-4o-mini 级别
        this.chatClient = builder.build();       // 通用客户端
    }

    // ═══════════════════════════════════════════════════════
    // 教程08.1：生成器-评估器-规划器架构
    // ═══════════════════════════════════════════════════════

    /**
     * 生成器-评估器-规划器执行结果
     */
    public record GeneratorEvaluatorResult(
            String draft,           // 最终草稿
            String evaluation,      // 最终评估
            double score,           // 最终评分
            int rounds,             // 迭代轮数
            List<String> history    // 评估历史
    ) {}

    /**
     * 评估结果模型
     */
    public record EvaluationResult(
            double score,           // 评分 0-10
            List<String> issues,    // 问题列表
            List<String> suggestions, // 改进建议
            boolean pass            // 是否通过
    ) {}

    /**
     * 执行生成器-评估器-规划器循环
     * 
     * 架构：
     * 生成器（创造性工作）→ 评估器（独立批判和打分）→ 规划器（决定下一步）
     * 
     * 核心思想：分开制造者和验证者，避免自我评估放水
     * 
     * @param task 任务描述
     * @param maxRounds 最大迭代轮数
     * @param threshold 通过阈值（评分）
     * @return 执行结果
     */
    public GeneratorEvaluatorResult executeLoop(String task, int maxRounds, double threshold) {
        String draft = "";
        String evaluation = "";
        double score = 0;
        int rounds = 0;
        List<String> history = new ArrayList<>();

        for (int i = 0; i < maxRounds; i++) {
            rounds = i + 1;

            // 1. 生成器节点：创造性工作
            if (i == 0) {
                draft = generatorNode(task, "");
            } else {
                draft = generatorNode(task, evaluation);
            }

            // 2. 评估器节点：独立批判和打分
            EvaluationResult evalResult = evaluatorNode(task, draft);
            score = evalResult.score();
            evaluation = "评分：%.1f/10\n问题：%s\n建议：%s".formatted(
                    score,
                    String.join("、", evalResult.issues()),
                    String.join("、", evalResult.suggestions())
            );
            history.add("第%d轮评分：%.1f".formatted(rounds, score));

            // 3. 规划器节点：决定下一步
            String decision = plannerNode(score, rounds, maxRounds, threshold);
            
            if ("finalize".equals(decision)) {
                // 质量达标或超过最大轮数，结束
                break;
            }
            // 否则继续循环改进
        }

        return new GeneratorEvaluatorResult(draft, evaluation, score, rounds, history);
    }

    /**
     * 生成器：负责创造性工作
     */
    private String generatorNode(String task, String feedback) {
        String prompt;
        if (feedback.isEmpty()) {
            prompt = """
                作为资深工程师，高质量完成以下任务：
                
                任务：%s
                
                请生成高质量的实现方案。
                """.formatted(task);
        } else {
            prompt = """
                作为资深工程师，根据反馈改进实现方案：
                
                任务：%s
                
                之前的反馈：
                %s
                
                请根据反馈改进实现。
                """.formatted(task, feedback);
        }

        return generatorClient.prompt()
                .system(prompt)
                .user("请生成或改进实现方案")
                .call()
                .content();
    }

    /**
     * 评估器：独立批判和打分
     */
    private EvaluationResult evaluatorNode(String task, String draft) {
        String prompt = """
            作为严格的代码审查专家，评估以下实现：
            
            任务：%s
            
            实现：
            %s
            
            评估维度（0-10分）：
            1. 正确性：功能是否正确
            2. 安全性：有无安全漏洞
            3. 性能：是否高效
            4. 可维护性：代码质量
            
            请以以下格式返回评估结果：
            {
                "score": 0-10,
                "issues": ["问题1", "问题2"],
                "suggestions": ["建议1", "建议2"],
                "pass": true/false
            }
            
            请严格评估，不要放水。
            """.formatted(task, draft);

        String response = evaluatorClient.prompt()
                .system(prompt)
                .user("请评估实现质量")
                .call()
                .content();

        // 解析评估结果（简化实现）
        double score = extractScoreFromResponse(response);
        List<String> issues = extractItemsFromResponse(response, "issues");
        List<String> suggestions = extractItemsFromResponse(response, "suggestions");
        boolean pass = score >= 8.0;

        return new EvaluationResult(score, issues, suggestions, pass);
    }

    /**
     * 规划器：决定下一步
     */
    private String plannerNode(double score, int round, int maxRounds, double threshold) {
        // 决策逻辑
        if (score >= threshold) {
            return "finalize";  // 高质量，通过
        } else if (round >= maxRounds) {
            return "finalize";  // 超过最大轮数
        } else {
            return "improve";   // 继续改进
        }
    }

    // ═══════════════════════════════════════════════════════
    // 教程08.2：Loop 实战模式 - 代码审查循环
    // ═══════════════════════════════════════════════════════

    /**
     * 代码审查循环结果
     */
    public record CodeReviewResult(
            String finalCode,         // 最终代码
            List<String> reviews,     // 审查历史
            int iterations,           // 修复轮数
            String summary            // 总结报告
    ) {}

    /**
     * 执行代码审查循环
     * 
     * 流程：
     * 安全审查 → 性能审查 → 修复问题 → 循环直到无问题或达到最大迭代次数
     * 
     * @param code 待审查的代码
     * @param maxIterations 最大迭代次数
     * @return 代码审查结果
     */
    public CodeReviewResult executeCodeReviewLoop(String code, int maxIterations) {
        List<String> reviews = new ArrayList<>();
        String currentCode = code;
        int iterations = 0;

        for (int i = 0; i < maxIterations; i++) {
            iterations = i + 1;

            // 1. 安全审查
            String securityReview = securityReviewNode(currentCode);
            reviews.add("安全审查（第%d轮）：%s".formatted(iterations, securityReview));

            // 2. 性能审查
            String performanceReview = performanceReviewNode(currentCode);
            reviews.add("性能审查（第%d轮）：%s".formatted(iterations, performanceReview));

            // 3. 判断是否需要继续
            if (!securityReview.contains("发现") && !performanceReview.contains("发现")) {
                // 没有问题，提前结束
                break;
            }

            // 4. 修复问题
            String allReviews = securityReview + "\n\n" + performanceReview;
            currentCode = fixIssuesNode(currentCode, allReviews);
        }

        String summary = """
            代码审查完成（经过 %d 轮修复）
            
            审查历史：
            %s
            
            最终代码已修复所有发现的问题。
            """.formatted(iterations, String.join("\n", reviews));

        return new CodeReviewResult(currentCode, reviews, iterations, summary);
    }

    /**
     * 安全审查节点
     */
    private String securityReviewNode(String code) {
        String prompt = """
            审查以下代码的安全问题：
            
            代码：
            %s
            
            检查项：
            1. SQL 注入
            2. XSS 攻击
            3. 缓冲区溢出
            4. 权限绕过
            5. 敏感信息泄露
            
            如果发现安全问题，请详细说明；如果没有问题，回复"未发现安全问题"。
            """.formatted(code);

        return chatClient.prompt()
                .system(prompt)
                .user("请进行安全审查")
                .call()
                .content();
    }

    /**
     * 性能审查节点
     */
    private String performanceReviewNode(String code) {
        String prompt = """
            审查以下代码的性能问题：
            
            代码：
            %s
            
            检查项：
            1. 时间复杂度
            2. 内存使用
            3. 不必要的计算
            4. 缓存机会
            5. 并发处理
            
            如果发现性能问题，请提供优化建议；如果没有问题，回复"未发现性能问题"。
            """.formatted(code);

        return chatClient.prompt()
                .system(prompt)
                .user("请进行性能审查")
                .call()
                .content();
    }

    /**
     * 修复问题节点
     */
    private String fixIssuesNode(String code, String reviews) {
        String prompt = """
            根据以下审查意见，修复代码中的问题：
            
            原代码：
            %s
            
            审查意见：
            %s
            
            请返回修复后的完整代码，确保所有问题都已解决。
            """.formatted(code, reviews);

        return chatClient.prompt()
                .system(prompt)
                .user("请修复代码")
                .call()
                .content();
    }

    // ═══════════════════════════════════════════════════════
    // 教程08.3：理解债务防护
    // ═══════════════════════════════════════════════════════

    /**
     * 理解债务防护系统
     * 监控 Loop 自动化程度，确保人类保持对代码库的理解
     */
    public static class UnderstandingDebtGuard {
        private int reviewedCount = 0;
        private int totalCount = 0;
        private final double minReviewRatio;  // 最低审查比例

        public UnderstandingDebtGuard(double minReviewRatio) {
            this.minReviewRatio = minReviewRatio;
        }

        /**
         * 判断是否应该人工审查
         */
        public boolean shouldReview() {
            double currentRatio = reviewedCount / Math.max(totalCount, 1);
            
            if (currentRatio < minReviewRatio) {
                return true;  // 强制审查
            }
            
            // 20% 随机抽查
            return Math.random() < 0.2;
        }

        /**
         * 记录总变更数
         */
        public void recordTotal(int count) {
            totalCount += count;
        }

        /**
         * 记录已审查数
         */
        public void recordReviewed(int count) {
            reviewedCount += count;
        }

        /**
         * 获取当前审查比例
         */
        public double getReviewRatio() {
            return reviewedCount / Math.max(totalCount, 1);
        }
    }

    // ═══════════════════════════════════════════════════════
    // 教程08.4：Token 成本优化
    // ═══════════════════════════════════════════════════════

    /**
     * Loop 成本分析器
     */
    public static class LoopCostAnalyzer {
        private final double generatorCost;  // 生成器每次成本
        private final double evaluatorCost;  // 评估器每次成本
        private final double plannerCost;    // 规划器每次成本

        public LoopCostAnalyzer(double generatorCost, double evaluatorCost, double plannerCost) {
            this.generatorCost = generatorCost;
            this.evaluatorCost = evaluatorCost;
            this.plannerCost = plannerCost;
        }

        /**
         * 估算每次迭代的成本
         */
        public double costPerIteration() {
            return generatorCost + evaluatorCost + plannerCost;
        }

        /**
         * 估算总成本
         */
        public double estimateTotalCost(int rounds) {
            return costPerIteration() * rounds;
        }

        /**
         * 获取成本报告
         */
        public String getCostReport(int rounds) {
            double perIteration = costPerIteration();
            double total = estimateTotalCost(rounds);
            
            return """
                Loop 成本分析
                =============
                
                每次迭代成本：
                - 生成器：$%.4f
                - 评估器：$%.4f
                - 规划器：$%.4f
                - 合计：$%.4f
                
                %d 轮总成本：$%.4f
                
                优化建议：
                1. 使用小模型做验证（评估器、规划器）
                2. 设置合理的最大轮数
                3. 跳过不必要的轮次
                4. 缓存中间结果
                """.formatted(
                    generatorCost, evaluatorCost, plannerCost, perIteration,
                    rounds, total
                );
        }
    }

    // ═══════════════════════════════════════════════════════
    // 辅助方法
    // ═══════════════════════════════════════════════════════

    /**
     * 从响应中提取评分
     */
    private double extractScoreFromResponse(String response) {
        try {
            String[] lines = response.split("\n");
            for (String line : lines) {
                if (line.contains("score")) {
                    String[] parts = line.split(":");
                    if (parts.length >= 2) {
                        return Double.parseDouble(parts[1].trim().replaceAll("[^0-9.]", ""));
                    }
                }
            }
        } catch (Exception e) {
            return 5.0;
        }
        return 5.0;
    }

    /**
     * 从响应中提取列表项
     */
    private List<String> extractItemsFromResponse(String response, String key) {
        List<String> result = new ArrayList<>();
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
