package com.example.agent.tools;

import org.springframework.ai.chat.model.ToolContext;
import org.springframework.stereotype.Component;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.function.Function;

/**
 * 内置工具集 - 对应教程 03-MCP客户端接入
 * 
 * 提供常用内置工具，与 MCP 工具配合使用。
 */
@Component
public class BuiltinTools {

    /**
     * 获取当前时间
     */
    @org.springframework.ai.tool.annotation.Tool(
        name = "get_current_time",
        description = "获取当前时间，返回格式化的日期时间字符串"
    )
    public String getCurrentTime() {
        return LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy年MM月dd日 HH:mm:ss"));
    }

    /**
     * 执行数学计算
     */
    @org.springframework.ai.tool.annotation.Tool(
        name = "calculate",
        description = "执行数学计算，支持加减乘除和基础函数"
    )
    public String calculate(String expression) {
        try {
            // 简单的表达式计算
            javax.script.ScriptEngine engine = new javax.script.ScriptEngineManager()
                    .getEngineByName("JavaScript");
            Object result = engine.eval(expression);
            return String.valueOf(result);
        } catch (Exception e) {
            return "计算错误: " + e.getMessage();
        }
    }

    /**
     * 保存信息到记忆
     */
    @org.springframework.ai.tool.annotation.Tool(
        name = "save_memory",
        description = "保存信息到长期记忆"
    )
    public String saveMemory(String fact, ToolContext toolContext) {
        // 从上下文获取用户ID
        String userId = "default";
        // 这里需要注入 MemoryService，简化处理
        return "已保存: " + fact;
    }

    /**
     * 搜索记忆
     */
    @org.springframework.ai.tool.annotation.Tool(
        name = "search_memory",
        description = "从长期记忆中搜索相关信息"
    )
    public String searchMemory(String query, ToolContext toolContext) {
        return "未找到相关记忆";
    }

    /**
     * 获取系统信息
     */
    @org.springframework.ai.tool.annotation.Tool(
        name = "get_system_info",
        description = "获取系统配置和状态信息"
    )
    public String getSystemInfo() {
        return """
                {
                  "status": "running",
                  "timestamp": "%s",
                  "version": "1.0.0"
                }
                """.formatted(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
    }
}
