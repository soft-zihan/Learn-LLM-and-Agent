package com.example.agent.memory;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import jakarta.annotation.PostConstruct;
import java.util.*;
import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;

/**
 * 记忆系统 - 对应教程 06-记忆系统设计
 * 
 * 实现三种记忆类型：
 * 1. 短期记忆：当前对话上下文
 * 2. 长期记忆：跨对话持久化
 * 3. 工作记忆：临时状态
 */
@Component
@Slf4j
public class MemoryService {

    private final ShortTermMemory shortTerm = new ShortTermMemory();
    private final LongTermMemory longTerm = new LongTermMemory();
    private final WorkingMemory working = new WorkingMemory();

    @PostConstruct
    public void init() {
        log.info("[Memory] 记忆系统初始化完成");
    }

    public ShortTermMemory shortTerm() { return shortTerm; }
    public LongTermMemory longTerm() { return longTerm; }
    public WorkingMemory working() { return working; }

    /**
     * 短期记忆
     */
    public static class ShortTermMemory {
        private final List<Map<String, String>> messages = new ArrayList<>();
        private final int maxMessages = 50;

        public void add(String role, String content) {
            Map<String, String> msg = new HashMap<>();
            msg.put("role", role);
            msg.put("content", content);
            msg.put("timestamp", LocalDateTime.now().toString());
            messages.add(msg);
            
            if (messages.size() > maxMessages) {
                messages.remove(0);
            }
        }

        public List<Map<String, String>> getRecent(int n) {
            return messages.subList(Math.max(0, messages.size() - n), messages.size());
        }

        public void clear() {
            messages.clear();
        }
    }

    /**
     * 长期记忆
     */
    public static class LongTermMemory {
        private final Path storageDir = Path.of("memory");

        public LongTermMemory() {
            try {
                Files.createDirectories(storageDir);
            } catch (IOException e) {
                throw new RuntimeException("创建记忆目录失败", e);
            }
        }

        public void saveFact(String userId, String fact) {
            Path file = storageDir.resolve(userId + ".json");
            List<Map<String, String>> facts = loadFacts(userId);
            
            Map<String, String> record = new HashMap<>();
            record.put("content", fact);
            record.put("timestamp", LocalDateTime.now().toString());
            facts.add(record);
            
            try {
                String json = "[" + facts.stream()
                        .map(f -> "{\"content\":\"" + f.get("content") + 
                                  "\",\"timestamp\":\"" + f.get("timestamp") + "\"}")
                        .reduce((a, b) -> a + "," + b).orElse("") + "]";
                Files.writeString(file, json);
            } catch (IOException e) {
                throw new RuntimeException("保存记忆失败", e);
            }
        }

        @SuppressWarnings("unchecked")
        public List<Map<String, String>> loadFacts(String userId) {
            Path file = storageDir.resolve(userId + ".json");
            if (!Files.exists(file)) return new ArrayList<>();
            
            try {
                String content = Files.readString(file);
                // 简单JSON解析
                List<Map<String, String>> facts = new ArrayList<>();
                if (content.startsWith("[")) {
                    String[] records = content.substring(1, content.length() - 1).split("\\},\\{");
                    for (String record : records) {
                        Map<String, String> fact = new HashMap<>();
                        if (record.contains("\"content\":\"")) {
                            int start = record.indexOf("\"content\":\"") + 11;
                            int end = record.indexOf("\"", start);
                            if (end > start) fact.put("content", record.substring(start, end));
                        }
                        facts.add(fact);
                    }
                }
                return facts;
            } catch (IOException e) {
                return new ArrayList<>();
            }
        }

        public List<Map<String, String>> search(String userId, String query) {
            return loadFacts(userId).stream()
                    .filter(f -> f.getOrDefault("content", "").toLowerCase().contains(query.toLowerCase()))
                    .toList();
        }
    }

    /**
     * 工作记忆
     */
    public static class WorkingMemory {
        private final Map<String, Object> data = new HashMap<>();

        public void set(String key, Object value) {
            data.put(key, value);
        }

        @SuppressWarnings("unchecked")
        public <T> T get(String key, T defaultValue) {
            return (T) data.getOrDefault(key, defaultValue);
        }

        public void delete(String key) {
            data.remove(key);
        }

        public void clear() {
            data.clear();
        }
    }
}
