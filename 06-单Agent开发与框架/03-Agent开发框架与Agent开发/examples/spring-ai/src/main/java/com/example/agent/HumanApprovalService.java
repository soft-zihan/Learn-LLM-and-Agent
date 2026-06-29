package com.example.agent;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * 人机协作与中断恢复服务
 * 
 * 对应教程：10-人机协作与中断恢复
 * 
 * 实现功能：
 * 1. interrupt() 函数模拟（教程10.1）
 * 2. 中断恢复机制（教程10.2）
 * 3. Command 模式（教程10.3）
 * 4. 人类审批流程（教程10.4）
 * 5. 人类编辑工具参数（教程10.5）
 * 
 * 注意：Java/Spring AI 中没有 LangGraph 的 interrupt() 函数，
 * 这里使用回调和状态持久化来模拟相同的功能。
 */
@Service
public class HumanApprovalService {

    private final ChatClient chatClient;
    private final ConcurrentHashMap<String, WorkflowState> workflowStates;

    public HumanApprovalService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
        this.workflowStates = new ConcurrentHashMap<>();
    }

    // ═══════════════════════════════════════════════════════
    // 教程10.1：人类审批（interrupt 模拟）
    // ═══════════════════════════════════════════════════════

    /**
     * 工作流状态
     */
    public record WorkflowState(
            String workflowId,
            String status,            // RUNNING, WAITING_HUMAN, COMPLETED, CANCELLED
            String currentStep,
            String draft,
            String humanFeedback,
            List<String> history
    ) {
        public WorkflowState copyWith(String status, String currentStep, String draft, 
                                       String humanFeedback, List<String> history) {
            return new WorkflowState(
                    this.workflowId,
                    status != null ? status : this.status,
                    currentStep != null ? currentStep : this.currentStep,
                    draft != null ? draft : this.draft,
                    humanFeedback != null ? humanFeedback : this.humanFeedback,
                    history != null ? history : this.history
            );
        }
    }

    /**
     * 审批请求
     */
    public record ApprovalRequest(
            String workflowId,
            String question,
            String draft,
            List<String> options
    ) {}

    /**
     * 执行需要人类审批的工作流
     * 
     * 流程：
     * 生成草稿 → 中断等待审批 → 根据审批结果继续
     * 
     * @param task 任务描述
     * @return 工作流 ID 和当前状态
     */
    public ApprovalRequest startApprovalWorkflow(String task) {
        String workflowId = generateWorkflowId();
        
        // 1. 生成草稿
        String draft = generateDraft(task);
        
        // 2. 保存状态（模拟中断）
        WorkflowState state = new WorkflowState(
                workflowId,
                "WAITING_HUMAN",
                "human_review",
                draft,
                null,
                List.of("草稿已生成")
        );
        workflowStates.put(workflowId, state);
        
        // 3. 返回审批请求（中断点）
        return new ApprovalRequest(
                workflowId,
                "请审核以下草稿：",
                draft,
                List.of("approve", "revise", "reject")
        );
    }

    /**
     * 人类审批后恢复执行
     * 
     * @param workflowId 工作流 ID
     * @param feedback 人类反馈（approve/revise/reject）
     * @return 最终结果
     */
    public String resumeAfterApproval(String workflowId, String feedback) {
        WorkflowState state = workflowStates.get(workflowId);
        if (state == null) {
            return "错误：找不到工作流 " + workflowId;
        }
        
        // 更新状态
        List<String> history = new ArrayList<>(state.history());
        history.add("人类反馈：" + feedback);
        
        WorkflowState updatedState = state.copyWith(
                "RUNNING",
                "process_feedback",
                null,
                feedback,
                history
        );
        workflowStates.put(workflowId, updatedState);
        
        // 处理反馈
        return processFeedback(updatedState);
    }

    /**
     * 生成草稿
     */
    private String generateDraft(String task) {
        String prompt = """
            请起草一份关于以下主题的报告：
            
            %s
            
            请生成详细、结构清晰的草稿。
            """.formatted(task);

        return chatClient.prompt()
                .system(prompt)
                .user("请起草报告")
                .call()
                .content();
    }

    /**
     * 处理人类反馈
     */
    private String processFeedback(WorkflowState state) {
        String feedback = state.humanFeedback();
        
        return switch (feedback) {
            case "approve" -> {
                // 审批通过
                List<String> history = new ArrayList<>(state.history());
                history.add("审批通过");
                
                WorkflowState finalState = state.copyWith(
                        "COMPLETED",
                        "finalized",
                        null,
                        null,
                        history
                );
                workflowStates.put(state.workflowId(), finalState);
                
                yield "✅ 审核通过！\n\n" + state.draft();
            }
            
            case "revise" -> {
                // 需要修改
                String revisedDraft = reviseDraft(state.draft());
                
                List<String> history = new ArrayList<>(state.history());
                history.add("已修改草稿");
                
                WorkflowState updatedState = state.copyWith(
                        "WAITING_HUMAN",
                        "human_review",
                        revisedDraft,
                        null,
                        history
                );
                workflowStates.put(state.workflowId(), updatedState);
                
                // 返回新的审批请求
                yield "草稿已修改，请重新审核：\n\n" + revisedDraft;
            }
            
            case "reject" -> {
                // 拒绝
                List<String> history = new ArrayList<>(state.history());
                history.add("草稿被拒绝");
                
                WorkflowState finalState = state.copyWith(
                        "COMPLETED",
                        "rejected",
                        null,
                        null,
                        history
                );
                workflowStates.put(state.workflowId(), finalState);
                
                yield "❌ 草稿被拒绝，需要重新起草";
            }
            
            default -> "未知的反馈类型：" + feedback;
        };
    }

    /**
     * 修改草稿
     */
    private String reviseDraft(String draft) {
        String prompt = """
            请根据审核意见修改以下草稿：
            
            原草稿：
            %s
            
            请改进草稿的质量。
            """.formatted(draft);

        return chatClient.prompt()
                .system(prompt)
                .user("请修改草稿")
                .call()
                .content();
    }

    // ═══════════════════════════════════════════════════════
    // 教程10.2：中断恢复
    // ═══════════════════════════════════════════════════════

    /**
     * 长时间运行任务的状态
     */
    public record LongRunningTask(
            String taskId,
            String status,        // RUNNING, PAUSED, COMPLETED, CANCELLED
            int progress,         // 0-100
            String message,
            String result
    ) {}

    /**
     * 启动长时间运行的任务（可中断）
     */
    public LongRunningTask startLongRunningTask(String task) {
        String taskId = generateTaskId();
        
        // 模拟任务执行到50%时暂停
        LongRunningTask pausedTask = new LongRunningTask(
                taskId,
                "PAUSED",
                50,
                "任务执行到50%，是否继续？",
                null
        );
        
        workflowStates.put(taskId, new WorkflowState(
                taskId,
                "WAITING_HUMAN",
                "progress_check",
                null,
                null,
                List.of("任务已执行50%")
        ));
        
        return pausedTask;
    }

    /**
     * 恢复长时间运行的任务
     */
    public LongRunningTask resumeLongRunningTask(String taskId, String action) {
        WorkflowState state = workflowStates.get(taskId);
        if (state == null) {
            return new LongRunningTask(taskId, "ERROR", 0, "任务不存在", null);
        }
        
        return switch (action) {
            case "继续" -> {
                // 继续执行到完成
                List<String> history = new ArrayList<>(state.history());
                history.add("用户选择继续");
                history.add("任务已完成");
                
                workflowStates.put(taskId, new WorkflowState(
                        taskId,
                        "COMPLETED",
                        "completed",
                        null,
                        null,
                        history
                ));
                
                yield new LongRunningTask(taskId, "COMPLETED", 100, "任务完成", "任务已成功完成");
            }
            
            case "取消" -> {
                List<String> history = new ArrayList<>(state.history());
                history.add("用户取消任务");
                
                workflowStates.put(taskId, new WorkflowState(
                        taskId,
                        "CANCELLED",
                        "cancelled",
                        null,
                        null,
                        history
                ));
                
                yield new LongRunningTask(taskId, "CANCELLED", 50, "任务已取消", null);
            }
            
            default -> new LongRunningTask(taskId, "ERROR", 50, "未知操作", null);
        };
    }

    // ═══════════════════════════════════════════════════════
    // 教程10.3：Command 模式
    // ═══════════════════════════════════════════════════════

    /**
     * Command 类型
     */
    public enum CommandType {
        GOTO_FINALIZE,      // 跳转到完成
        GOTO_REVISE,        // 跳转到修改
        GOTO_END,           // 结束
        UPDATE_STATE        // 更新状态
    }

    /**
     * Command 对象
     */
    public record Command(
            CommandType type,
            String target,
            Map<String, Object> updates
    ) {}

    /**
     * 使用 Command 模式的审核节点
     */
    public Command reviewWithCommand(String workflowId, String feedback) {
        WorkflowState state = workflowStates.get(workflowId);
        if (state == null) {
            return new Command(CommandType.GOTO_END, null, 
                    Map.of("message", "工作流不存在"));
        }
        
        return switch (feedback) {
            case "approve" -> new Command(
                    CommandType.GOTO_FINALIZE,
                    "finalize",
                    Map.of("status", "COMPLETED")
            );
            
            case "revise" -> new Command(
                    CommandType.GOTO_REVISE,
                    "generate",
                    Map.of(
                            "status", "RUNNING",
                            "message", "请修改草稿"
                    )
            );
            
            default -> new Command(
                    CommandType.GOTO_END,
                    null,
                    Map.of(
                            "status", "COMPLETED",
                            "message", "草稿被拒绝"
                    )
            );
        };
    }

    // ═══════════════════════════════════════════════════════
    // 教程10.4：人类审批（代码提交场景）
    // ═══════════════════════════════════════════════════════

    /**
     * 代码提交审批状态
     */
    public record CodeCommitState(
            String commitId,
            String code,
            String commitMessage,
            boolean approved,
            String status
    ) {}

    /**
     * 代码提交审批流程
     * 
     * 流程：
     * 生成代码 → 等待人类审批 → 提交或拒绝
     */
    public CodeCommitState startCodeCommitApproval(String task) {
        String commitId = generateCommitId();
        
        // 1. 生成代码
        String code = generateCode(task);
        
        // 2. 创建审批状态
        CodeCommitState commitState = new CodeCommitState(
                commitId,
                code,
                "代码更新：" + task,
                false,
                "WAITING_APPROVAL"
        );
        
        workflowStates.put(commitId, new WorkflowState(
                commitId,
                "WAITING_HUMAN",
                "code_approval",
                code,
                null,
                List.of("代码已生成，等待审批")
        ));
        
        return commitState;
    }

    /**
     * 审批代码提交
     */
    public String approveCodeCommit(String commitId, boolean approved) {
        WorkflowState state = workflowStates.get(commitId);
        if (state == null) {
            return "错误：找不到代码提交 " + commitId;
        }
        
        if (approved) {
            // 提交代码
            List<String> history = new ArrayList<>(state.history());
            history.add("代码已批准并提交");
            
            workflowStates.put(commitId, new WorkflowState(
                    commitId,
                    "COMPLETED",
                    "committed",
                    null,
                    null,
                    history
            ));
            
            return "✅ 代码已提交：\n\n" + state.draft();
        } else {
            // 拒绝提交
            List<String> history = new ArrayList<>(state.history());
            history.add("代码提交被拒绝");
            
            workflowStates.put(commitId, new WorkflowState(
                    commitId,
                    "COMPLETED",
                    "rejected",
                    null,
                    null,
                    history
            ));
            
            return "❌ 代码提交被拒绝";
        }
    }

    /**
     * 生成代码
     */
    private String generateCode(String task) {
        String prompt = """
            请编写代码实现以下功能：
            
            %s
            
            请提供完整、可运行的代码实现。
            """.formatted(task);

        return chatClient.prompt()
                .system(prompt)
                .user("请编写代码")
                .call()
                .content();
    }

    // ═══════════════════════════════════════════════════════
    // 教程10.5：人类编辑工具参数
    // ═══════════════════════════════════════════════════════

    /**
     * 工具调用编辑请求
     */
    public record ToolCallEditRequest(
            String requestId,
            String toolName,
            Map<String, Object> originalParams,
            String message
    ) {}

    /**
     * 人类编辑工具参数
     * 
     * 流程：
     * Agent 决定调用工具 → 中断等待人类编辑 → 使用编辑后的参数执行
     */
    public ToolCallEditRequest requestToolCallEdit(String toolName, 
                                                     Map<String, Object> params) {
        String requestId = generateRequestId();
        
        // 保存状态
        workflowStates.put(requestId, new WorkflowState(
                requestId,
                "WAITING_HUMAN",
                "tool_edit",
                null,
                null,
                List.of("等待人类编辑工具参数")
        ));
        
        return new ToolCallEditRequest(
                requestId,
                toolName,
                params,
                "请确认或编辑工具调用参数"
        );
    }

    /**
     * 使用编辑后的参数执行工具
     */
    public String executeWithEditedParams(String requestId, 
                                            Map<String, Object> editedParams) {
        WorkflowState state = workflowStates.get(requestId);
        if (state == null) {
            return "错误：找不到请求 " + requestId;
        }
        
        // 使用编辑后的参数执行
        // 实际应调用具体的工具
        String result = "使用编辑后的参数执行工具：\n" + editedParams;
        
        List<String> history = new ArrayList<>(state.history());
        history.add("工具已执行，参数：" + editedParams);
        
        workflowStates.put(requestId, new WorkflowState(
                requestId,
                "COMPLETED",
                "executed",
                null,
                null,
                history
        ));
        
        return result;
    }

    // ═══════════════════════════════════════════════════════
    // 敏感操作确认
    // ═══════════════════════════════════════════════════════

    /**
     * 敏感操作检测结果
     */
    public record SensitiveOperationResult(
            boolean isSensitive,
            String operation,
            String confirmationRequired
    ) {}

    /**
     * 检测并确认敏感操作
     */
    public SensitiveOperationResult checkSensitiveOperation(String operation) {
        String lowerOp = operation.toLowerCase();
        
        // 检测敏感关键词
        String[] sensitiveKeywords = {"删除", "delete", "drop", "发送", "send", 
                                       "格式化", "清空"};
        
        boolean isSensitive = false;
        for (String keyword : sensitiveKeywords) {
            if (lowerOp.contains(keyword.toLowerCase())) {
                isSensitive = true;
                break;
            }
        }
        
        if (isSensitive) {
            return new SensitiveOperationResult(
                    true,
                    operation,
                    "⚠️ 这是敏感操作，确认继续？"
            );
        }
        
        return new SensitiveOperationResult(false, operation, null);
    }

    // ═══════════════════════════════════════════════════════
    // 辅助方法
    // ═══════════════════════════════════════════════════════

    private String generateWorkflowId() {
        return "workflow_" + System.currentTimeMillis();
    }

    private String generateTaskId() {
        return "task_" + System.currentTimeMillis();
    }

    private String generateCommitId() {
        return "commit_" + System.currentTimeMillis();
    }

    private String generateRequestId() {
        return "request_" + System.currentTimeMillis();
    }

    /**
     * 获取工作流状态
     */
    public WorkflowState getWorkflowState(String workflowId) {
        return workflowStates.get(workflowId);
    }

    /**
     * 清理已完成的工作流
     */
    public void cleanupCompletedWorkflows() {
        workflowStates.entrySet().removeIf(entry -> 
                "COMPLETED".equals(entry.getValue().status()) ||
                "CANCELLED".equals(entry.getValue().status())
        );
    }
}
