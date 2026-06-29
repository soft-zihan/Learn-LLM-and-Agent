package com.example.agent.thread;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 会话管理服务 - 对应教程06：记忆隔离
 * 
 * 功能：
 * - 管理所有对话线程
 * - 通过 thread_id 实现会话隔离
 * - 每个线程独立的消息历史
 * 
 * 教程对应：
 * - 06-记忆系统设计：thread_id 实现记忆隔离
 */
@Service
@Slf4j
public class ThreadService {

    // 线程注册表
    private final Map<String, ThreadInfo> threads = new ConcurrentHashMap<>();
    
    // 每个线程的消息历史
    private final Map<String, List<Map<String, String>>> threadMessages = new ConcurrentHashMap<>();

    public ThreadService() {
        // 创建默认线程
        createThread("default", "默认对话");
    }

    /**
     * 创建新线程
     */
    public ThreadInfo createThread(String threadId, String name) {
        ThreadInfo info = new ThreadInfo(
            threadId,
            name,
            LocalDateTime.now().toString(),
            0
        );
        threads.put(threadId, info);
        threadMessages.put(threadId, Collections.synchronizedList(new ArrayList<>()));
        log.info("[Thread] 创建线程: {} ({})", name, threadId);
        return info;
    }

    /**
     * 创建新线程（自动生成ID）
     */
    public ThreadInfo createThread(String name) {
        String threadId = UUID.randomUUID().toString().substring(0, 8);
        return createThread(threadId, name);
    }

    /**
     * 获取所有线程
     */
    public List<ThreadInfo> listThreads() {
        return new ArrayList<>(threads.values());
    }

    /**
     * 获取指定线程
     */
    public ThreadInfo getThread(String threadId) {
        return threads.get(threadId);
    }

    /**
     * 删除线程
     */
    public boolean deleteThread(String threadId) {
        if ("default".equals(threadId)) {
            return false;
        }
        threads.remove(threadId);
        threadMessages.remove(threadId);
        log.info("[Thread] 删除线程: {}", threadId);
        return true;
    }

    /**
     * 添加消息到线程
     */
    public void addMessage(String threadId, String role, String content) {
        threadMessages.computeIfAbsent(threadId, k -> Collections.synchronizedList(new ArrayList<>()));
        
        Map<String, String> msg = new HashMap<>();
        msg.put("role", role);
        msg.put("content", content);
        msg.put("timestamp", LocalDateTime.now().toString());
        threadMessages.get(threadId).add(msg);

        // 更新线程消息计数
        ThreadInfo info = threads.get(threadId);
        if (info != null) {
            threads.put(threadId, new ThreadInfo(
                info.threadId(),
                info.name(),
                info.createdAt(),
                threadMessages.get(threadId).size()
            ));
        }
    }

    /**
     * 获取线程的消息历史
     */
    public List<Map<String, String>> getMessages(String threadId) {
        return threadMessages.getOrDefault(threadId, new ArrayList<>());
    }

    /**
     * 获取线程消息数
     */
    public int getMessageCount(String threadId) {
        return getMessages(threadId).size();
    }

    /**
     * 线程信息
     */
    public record ThreadInfo(
        String threadId,
        String name,
        String createdAt,
        int messageCount
    ) {}
}
