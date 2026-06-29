package com.example.agent.context;

import com.example.agent.thread.ThreadService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * 上下文管理服务 - 对应教程05：上下文工程
 * 
 * 功能：
 * - 获取上下文统计（消息数、token估算、使用率）
 * - 上下文压缩（生成摘要）
 * - 上下文裁剪
 * 
 * 教程对应：
 * - 05-提示词与上下文工程：2.3 上下文窗口管理策略
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class ContextService {

    private final ThreadService threadService;

    // 默认最大 token 数
    private static final int MAX_TOKENS = 100000;

    /**
     * 获取上下文统计信息
     */
    public ContextStats getStats(String threadId) {
        List<Map<String, String>> messages = threadService.getMessages(threadId);
        
        int totalTokens = messages.stream()
            .mapToInt(m -> estimateTokens(m.getOrDefault("content", "")))
            .sum();
        
        double usagePercent = MAX_TOKENS > 0 ? (totalTokens * 100.0 / MAX_TOKENS) : 0;
        
        // 按角色统计
        Map<String, Integer> roleCounts = new HashMap<>();
        roleCounts.put("user", 0);
        roleCounts.put("assistant", 0);
        roleCounts.put("system", 0);
        roleCounts.put("tool", 0);
        
        for (Map<String, String> msg : messages) {
            String role = msg.getOrDefault("role", "unknown");
            roleCounts.merge(role, 1, Integer::sum);
        }
        
        return new ContextStats(
            messages.size(),
            totalTokens,
            MAX_TOKENS,
            Math.round(usagePercent * 10.0) / 10.0,
            usagePercent > 80,
            roleCounts
        );
    }

    /**
     * 估算 token 数
     * 简单估算：中文约 1.5 字符/token，英文约 4 字符/token
     * 这里用保守估计：总字符数 / 2
     */
    public int estimateTokens(String text) {
        if (text == null || text.isEmpty()) return 0;
        return text.length() / 2;
    }

    /**
     * 压缩上下文（生成摘要）
     * 实际应用中会调用 LLM 生成摘要
     */
    public CompressResult compressContext(String threadId) {
        List<Map<String, String>> messages = threadService.getMessages(threadId);
        
        if (messages.size() < 5) {
            return new CompressResult("消息太少，无需压缩", 0, 0);
        }
        
        int originalCount = messages.size();
        int originalTokens = messages.stream()
            .mapToInt(m -> estimateTokens(m.getOrDefault("content", "")))
            .sum();
        
        // 实际应用中：
        // 1. 取前 N 条消息
        // 2. 调用 LLM 生成摘要
        // 3. 用摘要替换原始消息
        
        // 这里简化处理：返回提示信息
        log.info("[Context] 压缩线程 {} 的上下文: {} 条消息, {} tokens", 
            threadId, originalCount, originalTokens);
        
        return new CompressResult(
            "上下文压缩功能需要实际对话中使用",
            originalCount,
            originalTokens
        );
    }

    /**
     * 上下文统计信息
     */
    public record ContextStats(
        int messageCount,
        int estimatedTokens,
        int maxTokens,
        double usagePercent,
        boolean isNearLimit,
        Map<String, Integer> roleCounts
    ) {}

    /**
     * 压缩结果
     */
    public record CompressResult(
        String message,
        int originalMessageCount,
        int originalTokens
    ) {}
}
