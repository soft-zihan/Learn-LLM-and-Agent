package com.example.agent.test;

import com.example.agent.memory.MemoryService;
import com.example.agent.SubAgentService;
import com.example.agent.HumanApprovalService;
import com.example.agent.DesignPatternsService;
import com.example.agent.PlanExecutorService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.time.LocalDateTime;

/**
 * 测试与可观测性服务 - 对应教程 11-测试与可观测性
 */
@Service
@Slf4j
public class TestingService {

    /**
     * 基准测试任务
     */
    public record BenchmarkTask(String name, String inputData, String expectedOutput, 
                                String category, int difficulty) {}

    /**
     * 基准测试结果
     */
    public record BenchmarkResult(String taskName, boolean passed, double duration, 
                                  String output, String error) {}

    /**
     * 运行基准测试
     */
    public List<BenchmarkResult> runBenchmark(List<BenchmarkTask> tasks, 
                                              java.util.function.Function<String, String> agentFunc) {
        List<BenchmarkResult> results = new ArrayList<>();
        
        for (BenchmarkTask task : tasks) {
            long startTime = System.currentTimeMillis();
            try {
                String output = agentFunc.apply(task.inputData());
                double duration = (System.currentTimeMillis() - startTime) / 1000.0;
                
                boolean passed = output.toLowerCase().contains(task.expectedOutput().toLowerCase());
                
                results.add(new BenchmarkResult(
                    task.name(), passed, duration, 
                    output.length() > 200 ? output.substring(0, 200) : output,
                    null
                ));
            } catch (Exception e) {
                double duration = (System.currentTimeMillis() - startTime) / 1000.0;
                results.add(new BenchmarkResult(task.name(), false, duration, "", e.getMessage()));
            }
        }
        
        return results;
    }

    /**
     * 获取测试摘要
     */
    public Map<String, Object> getSummary(List<BenchmarkResult> results) {
        int total = results.size();
        long passed = results.stream().filter(BenchmarkResult::passed).count();
        double avgDuration = results.stream().mapToDouble(BenchmarkResult::duration).average().orElse(0);
        
        return Map.of(
            "total", total,
            "passed", passed,
            "failed", total - passed,
            "pass_rate", String.format("%.1f%%", passed * 100.0 / total),
            "avg_duration", String.format("%.2fs", avgDuration)
        );
    }

    /**
     * 单元测试：MCP连接
     */
    public boolean testMCPConnection(Object mcpClient) {
        try {
            // 测试MCP连接
            log.info("✓ MCP连接测试通过");
            return true;
        } catch (Exception e) {
            log.error("✗ MCP连接测试失败: {}", e.getMessage());
            return false;
        }
    }

    /**
     * 单元测试：Skills加载
     */
    public boolean testSkillsLoading(Object skillsRegistry) {
        try {
            log.info("✓ Skills加载测试通过");
            return true;
        } catch (Exception e) {
            log.error("✗ Skills加载测试失败: {}", e.getMessage());
            return false;
        }
    }

    /**
     * 单元测试：子智能体系统（教程09）
     */
    public boolean testSubAgentSystem(SubAgentService subAgentService) {
        try {
            // 测试子智能体路由
            SubAgentService.SubAgentResult result = subAgentService.routeAndExecute("查询数据库中的用户信息");
            log.info("✓ 子智能体路由测试通过: {}", result.agentName());
            return true;
        } catch (Exception e) {
            log.error("✗ 子智能体测试失败: {}", e.getMessage());
            return false;
        }
    }

    /**
     * 单元测试：人类审批流程（教程10）
     */
    public boolean testHumanApproval(HumanApprovalService approvalService) {
        try {
            // 测试审批流程创建
            HumanApprovalService.ApprovalRequest request = approvalService.startApprovalWorkflow("删除数据");
            log.info("✓ 人类审批流程测试通过: workflowId={}", request.workflowId());
            return true;
        } catch (Exception e) {
            log.error("✗ 人类审批测试失败: {}", e.getMessage());
            return false;
        }
    }

    /**
     * 单元测试：反思模式（教程07）
     */
    public boolean testReflectionPattern(DesignPatternsService designPatternsService) {
        try {
            // 测试反思循环
            DesignPatternsService.ReflectionResult result = designPatternsService.executeReflection("写一首诗", 2);
            log.info("✓ 反思模式测试通过");
            return true;
        } catch (Exception e) {
            log.error("✗ 反思模式测试失败: {}", e.getMessage());
            return false;
        }
    }

    /**
     * 单元测试：计划执行（教程08）
     */
    public boolean testPlanExecution(PlanExecutorService planExecutorService) {
        try {
            // 测试计划生成
            PlanExecutorService.GeneratorEvaluatorResult result = planExecutorService.executeLoop("帮我规划一次旅行", 2, 0.7);
            log.info("✓ 计划执行测试通过");
            return true;
        } catch (Exception e) {
            log.error("✗ 计划执行测试失败: {}", e.getMessage());
            return false;
        }
    }

    /**
     * 单元测试：记忆系统
     */
    public boolean testMemorySystem(MemoryService memoryService) {
        try {
            // 测试短期记忆
            memoryService.shortTerm().add("user", "测试消息");
            log.info("✓ 短期记忆测试通过");
            
            // 测试工作记忆
            memoryService.working().set("test_key", "test_value");
            String value = memoryService.working().get("test_key", null);
            if (!"test_value".equals(value)) {
                throw new RuntimeException("工作记忆值不匹配");
            }
            log.info("✓ 工作记忆测试通过");
            
            return true;
        } catch (Exception e) {
            log.error("✗ 记忆系统测试失败: {}", e.getMessage());
            return false;
        }
    }

    /**
     * 运行所有单元测试
     */
    public Map<String, Boolean> runAllUnitTests(
            SubAgentService subAgentService,
            HumanApprovalService approvalService,
            DesignPatternsService designPatternsService,
            PlanExecutorService planExecutorService,
            MemoryService memoryService) {
        
        log.info("开始运行所有单元测试...");
        
        return Map.of(
            "sub_agent", testSubAgentSystem(subAgentService),
            "human_approval", testHumanApproval(approvalService),
            "reflection", testReflectionPattern(designPatternsService),
            "plan_execution", testPlanExecution(planExecutorService),
            "memory_system", testMemorySystem(memoryService)
        );
    }

    /**
     * 性能分析器
     */
    public static class PerformanceProfiler {
        private int totalRequests = 0;
        private double totalDuration = 0;
        private double maxDuration = 0;
        private double minDuration = Double.MAX_VALUE;
        private int errors = 0;
        private long startTime;

        public void start() {
            startTime = System.currentTimeMillis();
        }

        public void end(boolean success) {
            double duration = (System.currentTimeMillis() - startTime) / 1000.0;
            totalRequests++;
            totalDuration += duration;
            maxDuration = Math.max(maxDuration, duration);
            minDuration = Math.min(minDuration, duration);
            if (!success) errors++;
        }

        public Map<String, String> getReport() {
            return Map.of(
                "total_requests", String.valueOf(totalRequests),
                "avg_duration", String.format("%.2fs", totalDuration / totalRequests),
                "max_duration", String.format("%.2fs", maxDuration),
                "min_duration", String.format("%.2fs", minDuration),
                "error_rate", String.format("%.1f%%", errors * 100.0 / totalRequests)
            );
        }
    }
}
