package com.example.agent;

import com.example.agent.context.ContextService;
import com.example.agent.mcp.McpClient;
import com.example.agent.skills.SkillRegistry;
import com.example.agent.memory.MemoryService;
import com.example.agent.thread.ThreadService;
import com.example.agent.tools.BuiltinTools;
import com.example.agent.test.BehaviorTestService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

import java.util.*;

@SpringBootApplication
@Slf4j
public class AgentApplication {

    public static void main(String[] args) {
        SpringApplication.run(AgentApplication.class, args);
    }

    @Bean
    public ChatClient chatClient(ChatClient.Builder builder,
                                 McpClient mcpClient,
                                 BuiltinTools builtinTools) {
        ToolCallbackProvider mcpProvider = mcpClient.getToolProvider();
        return builder
                .defaultSystem("你是一个有用的AI助手。你可以使用工具来帮助用户完成任务。")
                .defaultTools(mcpProvider, builtinTools)
                .build();
    }
}

@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
@Slf4j
class AgentController {

    private final ChatClient chatClient;
    private final SkillRegistry skillRegistry;
    private final MemoryService memoryService;
    private final SubAgentService subAgentService;
    private final HumanApprovalService humanApprovalService;
    private final DesignPatternsService designPatternsService;
    private final PlanExecutorService planExecutorService;
    private final PromptEngineeringService promptEngineeringService;
    private final BehaviorTestService behaviorTestService;
    private final ThreadService threadService;
    private final ContextService contextService;
    private final McpClient mcpClient;
    private final BuiltinTools builtinTools;

    // ── 对话接口 ─────────────────────────────────────────────

    @PostMapping(value = "/chat", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> chat(@RequestBody ChatRequest request) {
        String threadId = request.threadId() != null ? request.threadId() : "default";
        log.info("收到请求: {} (线程: {})", request.message(), threadId);
        
        // 记录用户消息到线程
        threadService.addMessage(threadId, "user", request.message());
        memoryService.shortTerm().add("user", request.message());
        
        return chatClient.prompt()
                .user(request.message())
                .stream()
                .content()
                .doOnNext(c -> {
                    threadService.addMessage(threadId, "assistant", c);
                    memoryService.shortTerm().add("assistant", c);
                });
    }

    @PostMapping("/chat/sync")
    public ChatResponse chatSync(@RequestBody ChatRequest request) {
        String threadId = request.threadId() != null ? request.threadId() : "default";
        String response = chatClient.prompt().user(request.message()).call().content();
        threadService.addMessage(threadId, "assistant", response);
        memoryService.shortTerm().add("assistant", response);
        return new ChatResponse(response);
    }

    // ── 工具接口 ─────────────────────────────────────────────

    @GetMapping("/tools")
    public Map<String, Object> listTools() {
        // 返回内置工具列表（与 Python /api/tools 格式一致）
        List<Map<String, String>> tools = new ArrayList<>();
        
        // 内置工具
        tools.add(Map.of("name", "get_current_time", "description", "获取当前时间"));
        tools.add(Map.of("name", "calculate", "description", "执行数学计算"));
        tools.add(Map.of("name", "save_memory", "description", "保存信息到长期记忆"));
        tools.add(Map.of("name", "search_memory", "description", "从长期记忆中搜索"));
        tools.add(Map.of("name", "get_system_info", "description", "获取系统信息"));
        
        return Map.of("tools", tools);
    }

    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "ok");
    }

    // ── 会话管理（教程06：记忆隔离）──────────────────────────

    @GetMapping("/threads")
    public Map<String, Object> listThreads() {
        return Map.of("threads", threadService.listThreads());
    }

    @PostMapping("/threads")
    public Map<String, Object> createThread(@RequestBody(required = false) Map<String, String> request) {
        String name = request != null && request.containsKey("name") 
            ? request.get("name") 
            : "对话 " + (threadService.listThreads().size() + 1);
        
        ThreadService.ThreadInfo thread = threadService.createThread(name);
        
        Map<String, Object> result = new HashMap<>();
        result.put("status", "ok");
        result.put("thread_id", thread.threadId());
        result.put("name", thread.name());
        return result;
    }

    @DeleteMapping("/threads/{threadId}")
    public Map<String, Object> deleteThread(@PathVariable String threadId) {
        if ("default".equals(threadId)) {
            return Map.of("status", "error", "message", "默认线程不能删除");
        }
        
        boolean deleted = threadService.deleteThread(threadId);
        if (deleted) {
            return Map.of("status", "ok", "deleted", threadId);
        } else {
            return Map.of("status", "error", "message", "线程不存在");
        }
    }

    // ── 上下文管理（教程05：上下文工程）──────────────────────

    @GetMapping("/context/{threadId}")
    public Map<String, Object> getContext(@PathVariable String threadId) {
        ThreadService.ThreadInfo thread = threadService.getThread(threadId);
        if (thread == null) {
            return Map.of("error", "线程不存在");
        }
        
        ContextService.ContextStats stats = contextService.getStats(threadId);
        
        Map<String, Object> result = new HashMap<>();
        result.put("thread_id", threadId);
        result.put("stats", Map.of(
            "message_count", stats.messageCount(),
            "estimated_tokens", stats.estimatedTokens(),
            "max_tokens", stats.maxTokens(),
            "usage_percent", stats.usagePercent(),
            "is_near_limit", stats.isNearLimit(),
            "role_counts", stats.roleCounts()
        ));
        return result;
    }

    @PostMapping("/context/{threadId}/compress")
    public Map<String, Object> compressContext(@PathVariable String threadId) {
        ThreadService.ThreadInfo thread = threadService.getThread(threadId);
        if (thread == null) {
            return Map.of("error", "线程不存在");
        }
        
        ContextService.CompressResult result = contextService.compressContext(threadId);
        
        return Map.of(
            "status", "ok",
            "thread_id", threadId,
            "message", result.message()
        );
    }

    // ── 技能管理（教程04：Skills）────────────────────────────

    @GetMapping("/skills")
    public Map<String, Object> listSkills() {
        List<Map<String, String>> skills = new ArrayList<>();
        for (SkillRegistry.SkillDefinition skill : skillRegistry.listAll()) {
            skills.add(Map.of(
                "name", skill.name(),
                "description", skill.description() != null ? skill.description() : ""
            ));
        }
        return Map.of("skills", skills, "count", skills.size());
    }

    // ── MCP管理（教程03：MCP协议）────────────────────────────

    @GetMapping("/mcp/status")
    public Map<String, Object> mcpStatus() {
        return mcpClient.getStatus();
    }

    @GetMapping("/mcp/tools")
    public Map<String, Object> mcpTools() {
        return Map.of("tools", mcpClient.getToolsDetail());
    }

    // ── 其他功能接口 ─────────────────────────────────────────

    @PostMapping("/reflection")
    public Map<String, String> reflection(@RequestBody Map<String, String> request) {
        String result = designPatternsService.executeReflection(
            request.get("task"), 
            Integer.parseInt(request.getOrDefault("iterations", "3"))
        ).improvedDraft();
        return Map.of("result", result);
    }

    @PostMapping("/subagent")
    public Map<String, String> subAgent(@RequestBody Map<String, String> request) {
        SubAgentService.SubAgentResult result = subAgentService.routeAndExecute(request.get("task"));
        return Map.of("result", result.result());
    }

    @PostMapping("/approval")
    public Map<String, String> approval(@RequestBody Map<String, String> request) {
        HumanApprovalService.ApprovalRequest req = humanApprovalService.startApprovalWorkflow(
            request.get("action"));
        return Map.of("result", req.question() + "\n" + req.draft());
    }

    @PostMapping("/plan")
    public Map<String, String> plan(@RequestBody Map<String, String> request) {
        PlanExecutorService.GeneratorEvaluatorResult result = planExecutorService.executeLoop(
            request.get("task"), 3, 8.0);
        return Map.of("result", result.draft());
    }

    @PostMapping("/prompt")
    public Map<String, String> prompt(@RequestBody Map<String, String> request) {
        return Map.of("prompt", promptEngineeringService.createClearPrompt(
            request.get("role"), request.get("capabilities"),
            request.get("constraints"), request.get("outputFormat"),
            request.get("examples")));
    }

    @GetMapping("/test/behavior")
    public Map<String, Object> runBehaviorTests() {
        return behaviorTestService.runAllBehaviorTests();
    }

    // ── 请求/响应模型 ────────────────────────────────────────

    public record ChatRequest(String message, String threadId) {}

    public record ChatResponse(String content) {}
}
