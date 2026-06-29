package com.example.agent;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * 子智能体与路由模式服务
 * 
 * 对应教程：09-子智能体与路由模式
 * 
 * 实现功能：
 * 1. 字典式子智能体（教程09.1）
 * 2. 并行子智能体（教程09.2）
 * 3. 任务路由器（教程09.3）
 * 4. 子智能体嵌套限制（教程09.4）
 */
@Service
public class SubAgentService {

    private final ChatClient chatClient;
    private final Map<String, SubAgent> subAgents;
    private final ExecutorService executorService;

    public SubAgentService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
        this.executorService = Executors.newFixedThreadPool(4);
        
        // 注册子智能体字典
        this.subAgents = new HashMap<>();
        
        // 网络搜索子智能体
        subAgents.put("search", new SubAgent(
                "search",
                "你是网络搜索专家，专注于获取最新的网络信息和新闻。",
                "搜索网络获取最新信息、新闻、文档等"
        ));
        
        // 数据库查询子智能体
        subAgents.put("database", new SubAgent(
                "database",
                "你是数据库专家，专注于查询结构化数据和执行SQL。",
                "查询数据库、执行SQL、获取数据"
        ));
        
        // 知识库搜索子智能体
        subAgents.put("knowledge", new SubAgent(
                "knowledge",
                "你是知识库专家，专注于搜索内部知识库和文档。",
                "搜索内部知识库、文档、FAQ"
        ));
        
        // 代码分析子智能体
        subAgents.put("code", new SubAgent(
                "code",
                "你是代码分析专家，专注于分析、审查和调试代码。",
                "分析代码、审查代码、调试"
        ));
    }

    // ═══════════════════════════════════════════════════════
    // 教程09.1：字典式子智能体
    // ═══════════════════════════════════════════════════════

    /**
     * 子智能体定义
     */
    public record SubAgent(
            String name,
            String systemPrompt,
            String description
    ) {}

    /**
     * 子智能体执行结果
     */
    public record SubAgentResult(
            String agentName,
            String result,
            long duration
    ) {}

    /**
     * 路由决策结果
     */
    public record RoutingDecision(
            String selectedAgent,
            String reason
    ) {}

    /**
     * 根据任务路由到合适的子智能体
     * 
     * 流程：
     * 用户问题 → 路由器选择子智能体 → 执行子智能体 → 返回结果
     * 
     * @param userQuery 用户查询
     * @return 子智能体执行结果
     */
    public SubAgentResult routeAndExecute(String userQuery) {
        long startTime = System.currentTimeMillis();
        
        // 1. 路由决策
        RoutingDecision decision = routeTask(userQuery);
        
        // 2. 获取选中的子智能体
        SubAgent agent = subAgents.get(decision.selectedAgent());
        if (agent == null) {
            return new SubAgentResult(
                    decision.selectedAgent(),
                    "未知的子智能体：" + decision.selectedAgent(),
                    System.currentTimeMillis() - startTime
            );
        }
        
        // 3. 执行子智能体
        String result = executeSubAgent(agent, userQuery);
        long duration = System.currentTimeMillis() - startTime;
        
        return new SubAgentResult(agent.name(), result, duration);
    }

    /**
     * 路由函数：根据任务类型选择子智能体
     */
    private RoutingDecision routeTask(String userQuery) {
        String query = userQuery.toLowerCase();
        
        // 使用 LLM 进行智能路由
        String prompt = """
            根据用户问题，选择最合适的子智能体：
            
            问题：%s
            
            可选子智能体：
            - search：%s
            - database：%s
            - knowledge：%s
            - code：%s
            
            只返回智能体名称（search/database/knowledge/code），不要其他内容。
            """.formatted(
                userQuery,
                subAgents.get("search").description(),
                subAgents.get("database").description(),
                subAgents.get("knowledge").description(),
                subAgents.get("code").description()
            );

        String response = chatClient.prompt()
                .system(prompt)
                .user("请选择最合适的子智能体")
                .call()
                .content()
                .trim()
                .toLowerCase();
        
        // 验证返回的智能体名称
        if (subAgents.containsKey(response)) {
            return new RoutingDecision(response, "LLM智能路由");
        }
        
        // 回退到关键词匹配
        return routeByKeywords(query);
    }

    /**
     * 基于关键词的路由（回退策略）
     */
    private RoutingDecision routeByKeywords(String query) {
        if (query.contains("搜索") || query.contains("查询") || 
            query.contains("新闻") || query.contains("最新")) {
            return new RoutingDecision("search", "关键词匹配");
        } else if (query.contains("数据库") || query.contains("sql") || 
                   query.contains("数据")) {
            return new RoutingDecision("database", "关键词匹配");
        } else if (query.contains("知识") || query.contains("文档") || 
                   query.contains("faq")) {
            return new RoutingDecision("knowledge", "关键词匹配");
        } else if (query.contains("代码") || query.contains("程序") || 
                   query.contains("分析")) {
            return new RoutingDecision("code", "关键词匹配");
        }
        
        // 默认使用搜索
        return new RoutingDecision("search", "默认路由");
    }

    /**
     * 执行子智能体
     */
    private String executeSubAgent(SubAgent agent, String userQuery) {
        return chatClient.prompt()
                .system(agent.systemPrompt())
                .user(userQuery)
                .call()
                .content();
    }

    // ═══════════════════════════════════════════════════════
    // 教程09.2：并行子智能体
    // ═══════════════════════════════════════════════════════

    /**
     * 并行子智能体结果
     */
    public record ParallelResult(
            Map<String, String> results,
            String aggregatedResult,
            long duration
    ) {}

    /**
     * 并行执行多个子智能体
     * 
     * 优势：
     * - 独立任务可以并行执行
     * - 减少总体执行时间
     * - 提高系统吞吐量
     * 
     * @param userQuery 用户查询
     * @param agentNames 要并行执行的子智能体名称列表
     * @return 并行执行结果
     */
    public ParallelResult executeParallel(List<String> agentNames, String userQuery) {
        long startTime = System.currentTimeMillis();
        
        // 创建并行任务
        List<CompletableFuture<SubAgentResult>> futures = agentNames.stream()
                .map(agentName -> CompletableFuture.supplyAsync(() -> {
                    SubAgent agent = subAgents.get(agentName);
                    if (agent == null) {
                        return new SubAgentResult(agentName, "未知的子智能体", 0);
                    }
                    long agentStart = System.currentTimeMillis();
                    String result = executeSubAgent(agent, userQuery);
                    long duration = System.currentTimeMillis() - agentStart;
                    return new SubAgentResult(agentName, result, duration);
                }, executorService))
                .toList();
        
        // 等待所有任务完成
        CompletableFuture<Void> allFutures = CompletableFuture.allOf(
                futures.toArray(new CompletableFuture[0])
        );
        
        // 收集结果
        Map<String, String> results = allFutures.thenApply(v ->
                futures.stream()
                        .map(CompletableFuture::join)
                        .collect(Collectors.toMap(
                                SubAgentResult::agentName,
                                SubAgentResult::result
                        ))
        ).join();
        
        // 汇总结果
        String aggregatedResult = aggregateResults(results);
        long duration = System.currentTimeMillis() - startTime;
        
        return new ParallelResult(results, aggregatedResult, duration);
    }

    /**
     * 汇总所有子智能体结果
     */
    private String aggregateResults(Map<String, String> results) {
        StringBuilder sb = new StringBuilder();
        sb.append("子智能体结果汇总：\n\n");
        
        for (Map.Entry<String, String> entry : results.entrySet()) {
            sb.append("【%s】\n%s\n\n".formatted(entry.getKey(), entry.getValue()));
        }
        
        // 使用 LLM 生成综合答案
        String prompt = """
            请综合以下子智能体的结果，生成一个完整的答案：
            
            %s
            
            请整合所有信息，提供全面的回答。
            """.formatted(sb.toString());

        return chatClient.prompt()
                .system(prompt)
                .user("请生成综合答案")
                .call()
                .content();
    }

    // ═══════════════════════════════════════════════════════
    // 教程09.3：任务路由器
    // ═══════════════════════════════════════════════════════

    /**
     * 任务类型枚举
     */
    public enum TaskType {
        RESEARCH,      // 研究和搜索
        CALCULATION,   // 数学计算
        CODING,        // 编写代码
        WRITING,       // 撰写文档
        GENERAL        // 通用
    }

    /**
     * 任务分类结果
     */
    public record TaskClassification(
            TaskType taskType,
            String confidence,
            String reasoning
    ) {}

    /**
     * 分类任务类型
     */
    public TaskClassification classifyTask(String userQuery) {
        String prompt = """
            分类以下任务类型：
            
            任务：%s
            
            类型选项：
            - RESEARCH：需要研究和搜索
            - CALCULATION：需要数学计算
            - CODING：需要编写代码
            - WRITING：需要撰写文档
            - GENERAL：其他通用任务
            
            请以以下格式返回：
            {
                "type": "任务类型",
                "confidence": "高/中/低",
                "reasoning": "分类理由"
            }
            """.formatted(userQuery);

        String response = chatClient.prompt()
                .system(prompt)
                .user("请分类任务类型")
                .call()
                .content();
        
        // 简化解析
        TaskType type = TaskType.GENERAL;
        if (response.contains("RESEARCH")) type = TaskType.RESEARCH;
        else if (response.contains("CALCULATION")) type = TaskType.CALCULATION;
        else if (response.contains("CODING")) type = TaskType.CODING;
        else if (response.contains("WRITING")) type = TaskType.WRITING;
        
        return new TaskClassification(type, "中", "基于关键词分析");
    }

    /**
     * 根据任务类型路由到专业节点
     */
    public String routeByTaskType(String userQuery) {
        TaskClassification classification = classifyTask(userQuery);
        
        return switch (classification.taskType()) {
            case RESEARCH -> researchNode(userQuery);
            case CALCULATION -> calculationNode(userQuery);
            case CODING -> codingNode(userQuery);
            case WRITING -> writingNode(userQuery);
            case GENERAL -> chatClient.prompt()
                    .user(userQuery)
                    .call()
                    .content();
        };
    }

    /**
     * 研究节点
     */
    private String researchNode(String query) {
        String prompt = """
            你是研究专家。请深入研究以下主题：
            
            %s
            
            请提供详细的研究结果和分析。
            """.formatted(query);

        return chatClient.prompt()
                .system(prompt)
                .user("请进行研究")
                .call()
                .content();
    }

    /**
     * 计算节点
     */
    private String calculationNode(String query) {
        String prompt = """
            你是数学专家。请执行以下计算：
            
            %s
            
            请提供详细的计算过程和结果。
            """.formatted(query);

        return chatClient.prompt()
                .system(prompt)
                .user("请执行计算")
                .call()
                .content();
    }

    /**
     * 编程节点
     */
    private String codingNode(String query) {
        String prompt = """
            你是编程专家。请编写代码实现以下需求：
            
            %s
            
            请提供完整的代码实现，包含注释和示例。
            """.formatted(query);

        return chatClient.prompt()
                .system(prompt)
                .user("请编写代码")
                .call()
                .content();
    }

    /**
     * 写作节点
     */
    private String writingNode(String query) {
        String prompt = """
            你是写作专家。请撰写以下内容：
            
            %s
            
            请提供结构清晰、内容详实的文档。
            """.formatted(query);

        return chatClient.prompt()
                .system(prompt)
                .user("请撰写内容")
                .call()
                .content();
    }

    // ═══════════════════════════════════════════════════════
    // 教程09.4：子智能体嵌套限制
    // ═══════════════════════════════════════════════════════

    /**
     * 子智能体路由器（带深度限制）
     * 防止无限嵌套调用
     */
    public static class SubAgentRouter {
        private final int maxDepth;

        public SubAgentRouter(int maxDepth) {
            this.maxDepth = maxDepth;
        }

        /**
         * 检查是否可以继续分发
         */
        public boolean canDispatch(int currentDepth) {
            return currentDepth < maxDepth;
        }

        /**
         * 分发任务（带深度检查）
         */
        public String dispatch(String task, int currentDepth) {
            if (!canDispatch(currentDepth)) {
                throw new IllegalStateException(
                        "达到最大嵌套深度 %d".formatted(maxDepth)
                );
            }
            
            // 实际执行逻辑应在外部实现
            // 这里只是深度检查的示例
            return "任务已分发，当前深度：%d".formatted(currentDepth);
        }
    }

    /**
     * 获取所有可用的子智能体
     */
    public Map<String, String> getAvailableAgents() {
        return subAgents.entrySet().stream()
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        e -> e.getValue().description()
                ));
    }

    /**
     * 关闭线程池
     */
    public void shutdown() {
        executorService.shutdown();
    }
}
