package com.example.agent.mcp;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.model.function.FunctionCallback;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * MCP 客户端 - 对应教程 03-MCP客户端接入
 * 
 * 功能：
 * 1. 使用Spring Boot自动配置MCP客户端
 * 2. 提供 ToolCallbackProvider 给 ChatClient 使用
 * 3. 提供 MCP 状态和工具详情
 * 
 * 教程对应：
 * - 03-MCP协议：MCP客户端接入、工具发现
 */
@Component
@Slf4j
public class McpClient {

    private final ToolCallbackProvider toolProvider;
    private boolean connected = false;

    public McpClient(ToolCallbackProvider toolProvider) {
        this.toolProvider = toolProvider;
        // 检查是否有工具来判断连接状态
        try {
            FunctionCallback[] tools = toolProvider.getToolCallbacks();
            this.connected = tools != null && tools.length > 0;
            log.info("[MCP] 客户端初始化完成，已连接 {} 个工具", 
                tools != null ? tools.length : 0);
        } catch (Exception e) {
            log.warn("[MCP] 获取工具列表失败: {}", e.getMessage());
            this.connected = false;
        }
    }

    /**
     * 获取工具回调提供者
     */
    public ToolCallbackProvider getToolProvider() {
        return toolProvider;
    }

    /**
     * 获取 MCP 连接状态
     */
    public Map<String, Object> getStatus() {
        Map<String, Object> status = new HashMap<>();
        status.put("connected", connected);
        
        int toolsCount = 0;
        try {
            FunctionCallback[] tools = toolProvider.getToolCallbacks();
            toolsCount = tools != null ? tools.length : 0;
        } catch (Exception e) {
            // ignore
        }
        status.put("tools_count", toolsCount);
        
        return status;
    }

    /**
     * 获取 MCP 工具详情列表
     */
    public List<Map<String, Object>> getToolsDetail() {
        List<Map<String, Object>> tools = new ArrayList<>();
        
        try {
            FunctionCallback[] callbacks = toolProvider.getToolCallbacks();
            if (callbacks != null) {
                for (FunctionCallback callback : callbacks) {
                    Map<String, Object> toolInfo = new HashMap<>();
                    toolInfo.put("name", callback.getName());
                    toolInfo.put("description", callback.getDescription());
                    tools.add(toolInfo);
                }
            }
        } catch (Exception e) {
            log.warn("[MCP] 获取工具详情失败: {}", e.getMessage());
        }
        
        return tools;
    }
}
