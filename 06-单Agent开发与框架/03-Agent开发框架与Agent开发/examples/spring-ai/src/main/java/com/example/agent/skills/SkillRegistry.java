package com.example.agent.skills;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import jakarta.annotation.PostConstruct;
import java.util.*;
import java.io.*;
import java.nio.file.*;

/**
 * Skills 系统集成 - 对应教程 04-Skills系统集成
 * 
 * 功能：
 * 1. 从文件系统加载 SKILL.md
 * 2. 解析 YAML frontmatter
 * 3. 技能注册表
 * 4. 根据查询匹配相关技能
 */
@Component
@Slf4j
public class SkillRegistry {

    private final Map<String, SkillDefinition> skills = new HashMap<>();

    @PostConstruct
    public void init() {
        // SKILLS 目录在 06-单Agent开发与框架/02-Agent工具与协议/SKILLS/
        try {
            // 当前工作目录是 examples/spring-ai
            Path currentDir = Paths.get(System.getProperty("user.dir")).toAbsolutePath();
            // 向上到 examples/，再向上到 03-LangGraph与Agent工程化/，然后找到同级的 02-Agent工具与协议/SKILLS/
            Path skillsDir = currentDir
                .getParent()              // examples/
                .getParent()              // 03-LangGraph与Agent工程化/
                .resolve("../02-Agent工具与协议/SKILLS")
                .normalize();
            
            loadSkills(skillsDir.toString());
            log.info("[Skills] 技能目录: {}", skillsDir);
            log.info("[Skills] 已加载 {} 个技能", skills.size());
        } catch (Exception e) {
            log.warn("[Skills] 加载技能目录失败: {}", e.getMessage());
        }
    }

    /**
     * 从目录加载所有技能
     */
    public void loadSkills(String dirPath) {
        Path dir = Path.of(dirPath);
        if (!Files.exists(dir)) {
            log.warn("技能目录不存在: {}", dirPath);
            return;
        }

        try (var stream = Files.walk(dir)) {
            stream.filter(p -> p.getFileName().toString().equals("SKILL.md"))
                  .forEach(this::loadSkill);
        } catch (IOException e) {
            log.error("加载技能失败", e);
        }
    }

    /**
     * 加载单个技能文件
     */
    private void loadSkill(Path skillFile) {
        try {
            String content = Files.readString(skillFile);
            
            // 解析 YAML frontmatter
            if (content.startsWith("---")) {
                String[] parts = content.split("---", 3);
                if (parts.length >= 3) {
                    String frontmatter = parts[1].trim();
                    String markdown = parts[2].trim();
                    
                    // 简单解析 name 和 description
                    String name = extractField(frontmatter, "name");
                    String description = extractField(frontmatter, "description");
                    
                    if (name != null) {
                        SkillDefinition skill = new SkillDefinition(name, description, markdown);
                        skills.put(name, skill);
                        log.info("  加载技能: {}", name);
                    }
                }
            }
        } catch (IOException e) {
            log.error("加载技能文件失败: {}", skillFile, e);
        }
    }

    /**
     * 从 YAML 中提取字段
     */
    private String extractField(String yaml, String field) {
        for (String line : yaml.split("\n")) {
            if (line.startsWith(field + ":")) {
                return line.substring(field.length() + 1).trim();
            }
        }
        return null;
    }

    /**
     * 获取指定技能
     */
    public SkillDefinition getSkill(String name) {
        return skills.get(name);
    }

    /**
     * 列出所有技能
     */
    public List<SkillDefinition> listAll() {
        return new ArrayList<>(skills.values());
    }

    /**
     * 根据查询查找相关技能
     */
    public List<SkillDefinition> findRelevant(String query) {
        String q = query.toLowerCase();
        return skills.values().stream()
                .filter(s -> s.name().toLowerCase().contains(q) || 
                             s.description().toLowerCase().contains(q))
                .toList();
    }

    /**
     * 构建包含技能信息的系统提示
     */
    public String buildSystemPrompt(String query) {
        List<SkillDefinition> relevant = query.isEmpty() ? listAll() : findRelevant(query);
        if (relevant.isEmpty()) return "";

        StringBuilder sb = new StringBuilder("## 可用技能\n\n");
        relevant.stream().limit(5).forEach(s -> 
            sb.append("- **").append(s.name()).append("**: ").append(s.description()).append("\n")
        );
        return sb.toString();
    }

    /**
     * 技能定义
     */
    public record SkillDefinition(String name, String description, String content) {}
}
