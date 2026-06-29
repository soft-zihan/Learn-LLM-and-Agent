package com.example.agent.test;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * 测试与可观测性服务
 * 
 * 对应教程：12-测试与可观测性
 * 
 * 实现功能：
 * 1. 基准测试套件（教程12.1）
 * 2. 评估指标体系（教程12.2）
 * 3. 结构化日志（教程12.3）
 * 4. 指标收集（教程12.4）
 * 5. 执行轨迹可视化（教程12.5）
 * 6. 性能分析工具（教程12.6）
 */
@Service
public class AgentMetricsService {

    private static final Logger logger = LoggerFactory.getLogger(AgentMetricsService.class);
    
    private final ChatClient chatClient;
    private final ConcurrentHashMap<String, ExecutionTrace> traces;
    private final AgentMetricsCollector metricsCollector;

    public AgentMetricsService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
        this.traces = new ConcurrentHashMap<>();
        this.metricsCollector = new AgentMetricsCollector();
    }

    // ═══════════════════════════════════════════════════════
    // 教程12.1：基准测试套件
    // ═══════════════════════════════════════════════════════

    /**
     * 基准测试任务
     */
    public record BenchmarkTask(
            String name,
            String inputData,
            String expectedOutput,
            String category,
            int difficulty
    ) {}

    /**
     * 基准测试结果
     */
    public record BenchmarkResult(
            int totalTasks,
            int passed,
            int failed,
            Map<String, CategoryStats> byCategory,
            List<TaskResult> details,
            double passRate
    ) {}

    /**
     * 分类统计
     */
    public record CategoryStats(
            int passed,
            int total,
            double passRate
    ) {}

    /**
     * 任务结果
     */
    public record TaskResult(
            String taskName,
            boolean passed,
            double duration,
            String error
    ) {}

    /**
     * 基准测试套件
     */
    public static class BenchmarkSuite {
        private final List<BenchmarkTask> tasks = new ArrayList<>();
        private final List<BenchmarkResult> results = new ArrayList<>();

        /**
         * 添加测试任务
         */
        public void addTask(BenchmarkTask task) {
            tasks.add(task);
        }

        /**
         * 加载标准基准测试
         */
        public void loadStandardBenchmarks() {
            // 工具使用测试
            addTask(new BenchmarkTask(
                    "calculator_basic",
                    "计算 256 * 128",
                    "32768",
                    "tool_use",
                    1
            ));

            addTask(new BenchmarkTask(
                    "search_fact",
                    "爱因斯坦哪一年获得诺贝尔奖？",
                    "1921",
                    "tool_use",
                    2
            ));

            // 多步推理测试
            addTask(new BenchmarkTask(
                    "multi_step_reasoning",
                    "如果 3 个人 3 天完成 3 个项目，9 个人 9 天完成多少？",
                    "27",
                    "reasoning",
                    3
            ));

            // 代码生成测试
            addTask(new BenchmarkTask(
                    "code_generation",
                    "写一个快速排序算法",
                    "快速排序",
                    "coding",
                    2
            ));

            // 写作测试
            addTask(new BenchmarkTask(
                    "writing_task",
                    "写一封商务邮件",
                    "邮件",
                    "writing",
                    1
            ));
        }

        /**
         * 运行基准测试
         */
        public BenchmarkResult runBenchmark(ChatClient chatClient) {
            int passed = 0;
            int failed = 0;
            Map<String, int[]> categoryStats = new HashMap<>();
            List<TaskResult> details = new ArrayList<>();

            for (BenchmarkTask task : tasks) {
                long startTime = System.currentTimeMillis();

                try {
                    // 执行任务
                    String actualOutput = chatClient.prompt()
                            .user(task.inputData())
                            .call()
                            .content();

                    double duration = (System.currentTimeMillis() - startTime) / 1000.0;

                    // 评估
                    boolean taskPassed = evaluateOutput(task.expectedOutput(), actualOutput);

                    if (taskPassed) {
                        passed++;
                    } else {
                        failed++;
                    }

                    // 按分类统计
                    categoryStats.putIfAbsent(task.category(), new int[]{0, 0});
                    int[] stats = categoryStats.get(task.category());
                    stats[1]++;
                    if (taskPassed) {
                        stats[0]++;
                    }

                    details.add(new TaskResult(task.name(), taskPassed, duration, null));

                } catch (Exception e) {
                    failed++;
                    details.add(new TaskResult(task.name(), false, 0, e.getMessage()));
                }
            }

            // 构建分类统计
            Map<String, CategoryStats> byCategory = categoryStats.entrySet().stream()
                    .collect(Collectors.toMap(
                            Map.Entry::getKey,
                            e -> new CategoryStats(
                                    e.getValue()[0],
                                    e.getValue()[1],
                                    (double) e.getValue()[0] / e.getValue()[1]
                            )
                    ));

            double passRate = (double) passed / tasks.size();

            return new BenchmarkResult(
                    tasks.size(),
                    passed,
                    failed,
                    byCategory,
                    details,
                    passRate
            );
        }

        /**
         * 评估输出
         */
        private boolean evaluateOutput(String expected, String actual) {
            Set<String> expectedKeywords = new HashSet<>(
                    Arrays.asList(expected.toLowerCase().split("\\s+"))
            );
            String actualLower = actual.toLowerCase();

            // 如果包含所有关键词（长度>2），认为通过
            return expectedKeywords.stream()
                    .filter(kw -> kw.length() > 2)
                    .allMatch(actualLower::contains);
        }

        /**
         * 生成测试报告
         */
        public String generateReport(BenchmarkResult result) {
            StringBuilder report = new StringBuilder();
            report.append("""
                    基准测试报告
                    ============
                    
                    总览：
                    - 总任务数：%d
                    - 通过：%d
                    - 失败：%d
                    - 通过率：%.2f%%
                    
                    按分类：
                    """.formatted(
                    result.totalTasks(),
                    result.passed(),
                    result.failed(),
                    result.passRate() * 100
            ));

            for (Map.Entry<String, CategoryStats> entry : result.byCategory().entrySet()) {
                report.append("- %s: %d/%d (%.2f%%)\n".formatted(
                        entry.getKey(),
                        entry.getValue().passed(),
                        entry.getValue().total(),
                        entry.getValue().passRate() * 100
                ));
            }

            return report.toString();
        }
    }

    // ═══════════════════════════════════════════════════════
    // 教程12.2：评估指标体系
    // ═══════════════════════════════════════════════════════

    /**
     * Agent 评估指标
     */
    public record AgentMetrics(
            double toolAccuracy,              // 工具使用准确率
            double toolSelectionPrecision,    // 工具选择精确率
            double taskCompletionRate,        // 任务完成率
            double averageIterations,         // 平均迭代次数
            double outputAccuracy,            // 输出准确率
            double outputCompleteness,        // 输出完整性
            double errorRate,                 // 错误率
            double recoverySuccessRate,       // 恢复成功率
            double averageTokens,             // 平均 token 数
            double costPerTask                // 每个任务的成本
    ) {
        public static AgentMetrics empty() {
            return new AgentMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        }
    }

    /**
     * 指标收集器
     */
    public static class AgentMetricsCollector {
        private final AtomicInteger requestsTotal = new AtomicInteger(0);
        private final AtomicInteger successTotal = new AtomicInteger(0);
        private final AtomicInteger errorTotal = new AtomicInteger(0);
        private final AtomicInteger tokensTotal = new AtomicInteger(0);
        private final List<Double> durations = Collections.synchronizedList(new ArrayList<>());
        private final Map<String, AtomicInteger> toolCalls = new ConcurrentHashMap<>();

        /**
         * 记录请求
         */
        public void recordRequest(boolean success, double duration) {
            requestsTotal.incrementAndGet();
            if (success) {
                successTotal.incrementAndGet();
            } else {
                errorTotal.incrementAndGet();
            }
            durations.add(duration);
        }

        /**
         * 记录 Token 使用
         */
        public void recordTokens(int count) {
            tokensTotal.addAndGet(count);
        }

        /**
         * 记录工具调用
         */
        public void recordToolCall(String toolName) {
            toolCalls.computeIfAbsent(toolName, k -> new AtomicInteger(0)).incrementAndGet();
        }

        /**
         * 获取指标
         */
        public AgentMetrics getMetrics() {
            int total = requestsTotal.get();
            if (total == 0) {
                return AgentMetrics.empty();
            }

            double avgDuration = durations.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            double errorRate = (double) errorTotal.get() / total;
            double successRate = (double) successTotal.get() / total;

            return new AgentMetrics(
                    successRate,              // toolAccuracy
                    successRate,              // toolSelectionPrecision
                    successRate,              // taskCompletionRate
                    1.0,                      // averageIterations
                    successRate,              // outputAccuracy
                    successRate,              // outputCompleteness
                    errorRate,                // errorRate
                    successRate,              // recoverySuccessRate
                    tokensTotal.get(),        // averageTokens
                    0.0                       // costPerTask
            );
        }

        /**
         * 获取工具调用统计
         */
        public Map<String, Integer> getToolCallStats() {
            return toolCalls.entrySet().stream()
                    .collect(Collectors.toMap(
                            Map.Entry::getKey,
                            e -> e.getValue().get()
                    ));
        }
    }

    // ═══════════════════════════════════════════════════════
    // 教程12.3：结构化日志
    // ═══════════════════════════════════════════════════════

    /**
     * 记录 Agent 启动日志
     */
    public void logAgentStart(String agentName, String userId, String task) {
        logger.info("Agent 启动 | agent_name={} | user_id={} | task={}", 
                agentName, userId, task);
    }

    /**
     * 记录工具调用日志
     */
    public void logToolCall(String toolName, String input, double durationMs, boolean success) {
        logger.info("工具调用 | tool_name={} | input={} | duration_ms={} | success={}", 
                toolName, input, durationMs, success);
    }

    /**
     * 记录 LLM 调用日志
     */
    public void logLlmCall(int promptLength, int responseLength, int tokens, double durationMs) {
        logger.info("LLM调用 | prompt_length={} | response_length={} | tokens={} | duration_ms={}", 
                promptLength, responseLength, tokens, durationMs);
    }

    /**
     * 记录错误日志
     */
    public void logError(String errorType, String message, String details) {
        logger.error("Agent错误 | error_type={} | message={} | details={}", 
                errorType, message, details);
    }

    // ═══════════════════════════════════════════════════════
    // 教程12.4：指标收集（Prometheus 风格）
    // ═══════════════════════════════════════════════════════

    /**
     * 记录请求指标
     */
    public void recordRequest(String agentName, String status, double duration) {
        metricsCollector.recordRequest("success".equals(status), duration);
        
        logger.info("请求指标 | agent_name={} | status={} | duration={}", 
                agentName, status, duration);
    }

    /**
     * 记录 Token 指标
     */
    public void recordTokens(int tokenCount, String tokenType) {
        metricsCollector.recordTokens(tokenCount);
        
        logger.info("Token指标 | type={} | count={}", tokenType, tokenCount);
    }

    /**
     * 记录错误指标
     */
    public void recordError(String errorType) {
        logger.error("错误指标 | error_type={}", errorType);
    }

    /**
     * 获取当前指标
     */
    public AgentMetrics getCurrentMetrics() {
        return metricsCollector.getMetrics();
    }

    // ═══════════════════════════════════════════════════════
    // 教程12.5：执行轨迹可视化
    // ═══════════════════════════════════════════════════════

    /**
     * 执行事件
     */
    public record ExecutionEvent(
            String eventType,     // start, tool_call, llm_call, end
            String timestamp,
            String tool,
            String input,
            String output,
            int tokens,
            String result
    ) {}

    /**
     * 执行轨迹
     */
    public record ExecutionTrace(
            String taskId,
            List<ExecutionEvent> events
    ) {
        public ExecutionTrace(String taskId) {
            this(taskId, new ArrayList<>());
        }
    }

    /**
     * 开始追踪
     */
    public void startTrace(String taskId, String taskName) {
        traces.put(taskId, new ExecutionTrace(taskId));
        
        traces.get(taskId).events.add(new ExecutionEvent(
                "start",
                Instant.now().toString(),
                null,
                null,
                null,
                0,
                taskName
        ));
        
        logger.info("开始追踪 | task_id={} | task={}", taskId, taskName);
    }

    /**
     * 记录工具调用
     */
    public void recordToolCall(String taskId, String toolName, String input, String output) {
        ExecutionTrace trace = traces.get(taskId);
        if (trace != null) {
            trace.events.add(new ExecutionEvent(
                    "tool_call",
                    Instant.now().toString(),
                    toolName,
                    input,
                    output != null ? output.substring(0, Math.min(200, output.length())) : null,
                    0,
                    null
            ));
            
            logger.info("工具调用追踪 | task_id={} | tool={}", taskId, toolName);
        }
    }

    /**
     * 记录 LLM 调用
     */
    public void recordLlmCall(String taskId, int promptLength, String response, int tokens) {
        ExecutionTrace trace = traces.get(taskId);
        if (trace != null) {
            trace.events.add(new ExecutionEvent(
                    "llm_call",
                    Instant.now().toString(),
                    null,
                    null,
                    response != null ? response.substring(0, Math.min(200, response.length())) : null,
                    tokens,
                    null
            ));
            
            logger.info("LLM调用追踪 | task_id={} | tokens={}", taskId, tokens);
        }
    }

    /**
     * 结束追踪
     */
    public void endTrace(String taskId, String result) {
        ExecutionTrace trace = traces.get(taskId);
        if (trace != null) {
            trace.events.add(new ExecutionEvent(
                    "end",
                    Instant.now().toString(),
                    null,
                    null,
                    null,
                    0,
                    result
            ));
            
            logger.info("结束追踪 | task_id={} | result={}", taskId, result);
        }
    }

    /**
     * 可视化轨迹
     */
    public String visualizeTrace(String taskId) {
        ExecutionTrace trace = traces.get(taskId);
        if (trace == null) {
            return "轨迹不存在";
        }

        StringBuilder visualization = new StringBuilder();
        visualization.append("执行轨迹：\n").append("=".repeat(60)).append("\n");

        for (ExecutionEvent event : trace.events()) {
            String timestamp = event.timestamp().split("T")[1].split("\\.")[0];

            visualization.append("[%s] ".formatted(timestamp));

            switch (event.eventType()) {
                case "start" -> visualization.append("🚀 开始：").append(event.result()).append("\n");
                case "tool_call" -> visualization.append("🔧 工具：").append(event.tool()).append("\n");
                case "llm_call" -> visualization.append("🧠 LLM（").append(event.tokens()).append(" tokens）\n");
                case "end" -> visualization.append("✅ 完成\n");
            }

            visualization.append("-".repeat(60)).append("\n");
        }

        return visualization.toString();
    }

    // ═══════════════════════════════════════════════════════
    // 教程12.6：性能分析工具
    // ═══════════════════════════════════════════════════════

    /**
     * 性能分析器
     */
    public static class PerformanceProfiler {
        private final Map<String, List<ProfileRecord>> profiles = new ConcurrentHashMap<>();

        /**
         * 性能记录
         */
        public record ProfileRecord(
                double duration,
                boolean success,
                String error
        ) {}

        /**
         * 记录性能数据
         */
        public void record(String funcName, double duration, boolean success, String error) {
            profiles.computeIfAbsent(funcName, k -> new ArrayList<>())
                    .add(new ProfileRecord(duration, success, error));
        }

        /**
         * 获取性能统计
         */
        public Map<String, Object> getStats(String funcName) {
            List<ProfileRecord> records = profiles.get(funcName);
            if (records == null || records.isEmpty()) {
                return Map.of();
            }

            List<Double> durations = records.stream()
                    .mapToDouble(ProfileRecord::duration)
                    .boxed()
                    .toList();

            double avgDuration = durations.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            double p95Duration = getPercentile(durations, 95);
            long successCount = records.stream().filter(ProfileRecord::success).count();

            return Map.of(
                    "call_count", records.size(),
                    "avg_duration", avgDuration,
                    "p95_duration", p95Duration,
                    "success_rate", (double) successCount / records.size()
            );
        }

        /**
         * 计算百分位数
         */
        private double getPercentile(List<Double> values, int percentile) {
            if (values.isEmpty()) return 0;
            
            List<Double> sorted = new ArrayList<>(values);
            Collections.sort(sorted);
            
            int index = (int) Math.ceil(percentile / 100.0 * sorted.size()) - 1;
            return sorted.get(Math.max(0, index));
        }

        /**
         * 获取所有函数的统计
         */
        public Map<String, Map<String, Object>> getAllStats() {
            return profiles.keySet().stream()
                    .collect(Collectors.toMap(
                            funcName -> funcName,
                            this::getStats
                    ));
        }
    }

    // ═══════════════════════════════════════════════════════
    // 辅助方法
    // ═══════════════════════════════════════════════════════

    /**
     * 获取所有轨迹
     */
    public Set<String> getAllTraces() {
        return traces.keySet();
    }

    /**
     * 清理已完成的轨迹
     */
    public void cleanupCompletedTraces() {
        // 保留最近的轨迹，清理旧的
        if (traces.size() > 100) {
            List<String> keys = new ArrayList<>(traces.keySet());
            for (int i = 0; i < keys.size() - 100; i++) {
                traces.remove(keys.get(i));
            }
        }
    }
}
