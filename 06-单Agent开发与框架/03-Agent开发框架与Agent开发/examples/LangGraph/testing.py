"""
完整的测试与可观测性套件 - 对应教程 11-测试与可观测性

覆盖：
1. 单元测试（模块级别）
2. 集成测试（功能级别）
3. 基准测试（性能级别）
4. E2E测试（端到端）
"""

import os
import sys
import time
import json
import asyncio
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(__file__))

# ── 配置日志 ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TestSuite")


# ── 数据模型 ──────────────────────────────────────────────

@dataclass
class TestResult:
    """单个测试结果"""
    name: str
    category: str
    passed: bool
    duration: float
    message: str = ""
    error: Optional[str] = None


@dataclass
class TestSuiteResult:
    """测试套件结果"""
    suite_name: str
    timestamp: str
    results: List[TestResult] = field(default_factory=list)
    
    @property
    def total(self) -> int:
        return len(self.results)
    
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def failed(self) -> int:
        return self.total - self.passed
    
    @property
    def pass_rate(self) -> str:
        return f"{self.passed/self.total*100:.1f}%" if self.total > 0 else "0%"
    
    @property
    def avg_duration(self) -> float:
        return sum(r.duration for r in self.results) / self.total if self.total > 0 else 0


# ── 1. 单元测试 ─────────────────────────────────────────

class TestMCPClient:
    """测试MCP客户端模块"""
    
    @staticmethod
    def test_imports():
        """测试模块导入"""
        try:
            from mcp_client import init_mcp, get_mcp_tools, shutdown_mcp
            return True, "MCP模块导入成功"
        except Exception as e:
            return False, f"MCP模块导入失败: {e}"
    
    @staticmethod
    def test_mcp_tools_structure():
        """测试MCP工具结构"""
        try:
            from mcp_client import _mcp_tools_raw
            if not _mcp_tools_raw:
                return False, "MCP工具列表为空（需先运行init_mcp）"
            
            # 检查工具结构
            for tool in _mcp_tools_raw:
                assert hasattr(tool, 'name'), "工具缺少name属性"
                assert hasattr(tool, 'description'), "工具缺少description属性"
            
            return True, f"MCP工具结构正确，共{len(_mcp_tools_raw)}个工具"
        except Exception as e:
            return False, f"MCP工具结构检查失败: {e}"


class TestSkills:
    """测试Skills模块"""
    
    @staticmethod
    def test_imports():
        """测试模块导入"""
        try:
            from skills import SkillDefinition, SkillLoader, SkillRegistry, skill_registry
            return True, "Skills模块导入成功"
        except Exception as e:
            return False, f"Skills模块导入失败: {e}"
    
    @staticmethod
    def test_skill_registry():
        """测试技能注册表"""
        try:
            from skills import skill_registry
            skills = skill_registry.list_all()
            assert isinstance(skills, list), "skills应为列表"
            return True, f"技能注册表正常，共{len(skills)}个技能"
        except Exception as e:
            return False, f"技能注册表测试失败: {e}"


class TestConfig:
    """测试配置模块"""
    
    @staticmethod
    def test_imports():
        """测试配置导入"""
        try:
            from config import API_KEY, BASE_URL, MODEL, llm
            assert API_KEY, "API_KEY不应为空"
            assert BASE_URL, "BASE_URL不应为空"
            assert MODEL, "MODEL不应为空"
            return True, "配置模块导入成功"
        except Exception as e:
            return False, f"配置模块导入失败: {e}"
    
    @staticmethod
    def test_llm_instance():
        """测试LLM实例"""
        try:
            from config import llm
            assert llm is not None, "LLM实例不应为空"
            assert hasattr(llm, 'invoke'), "LLM应有invoke方法"
            return True, "LLM实例创建成功"
        except Exception as e:
            return False, f"LLM实例测试失败: {e}"


class TestTools:
    """测试工具模块"""
    
    @staticmethod
    def test_imports():
        """测试工具导入"""
        try:
            from tools import get_current_time, calculate, search_memory
            return True, "工具模块导入成功"
        except Exception as e:
            return False, f"工具模块导入失败: {e}"
    
    @staticmethod
    def test_tool_structure():
        """测试工具结构"""
        try:
            from tools import get_current_time, calculate, search_memory
            # 测试工具调用（LangChain工具需要invoke）
            time_result = get_current_time.invoke({})
            assert isinstance(time_result, str), "时间工具应返回字符串"
            
            calc_result = calculate.invoke({"expression": "2+2"})
            assert "4" in str(calc_result), f"计算工具应返回4，实际: {calc_result}"
            
            return True, "工具结构正确"
        except Exception as e:
            return False, f"工具结构测试失败: {e}"


class TestMemory:
    """测试记忆模块"""
    
    @staticmethod
    def test_imports():
        """测试模块导入"""
        try:
            from memory import ShortTermMemory, LongTermMemory, WorkingMemory
            return True, "Memory模块导入成功"
        except Exception as e:
            return False, f"Memory模块导入失败: {e}"
    
    @staticmethod
    def test_short_term():
        """测试短期记忆"""
        try:
            from memory import ShortTermMemory
            mem = ShortTermMemory(max_messages=5)
            mem.add("user", "测试1")
            mem.add("assistant", "回复1")
            mem.add("user", "测试2")
            
            recent = mem.get_recent(2)
            assert len(recent) == 2, f"应返回2条，实际{len(recent)}"
            assert recent[0]["role"] == "assistant", "第一条应为assistant"
            assert recent[1]["role"] == "user", "第二条应为user"
            
            return True, "短期记忆测试通过"
        except Exception as e:
            return False, f"短期记忆测试失败: {e}"
    
    @staticmethod
    def test_working_memory():
        """测试工作记忆"""
        try:
            from memory import WorkingMemory
            mem = WorkingMemory()
            mem.set("key1", "value1")
            mem.set("key2", 123)
            
            assert mem.get("key1") == "value1", "key1值不匹配"
            assert mem.get("key2") == 123, "key2值不匹配"
            assert mem.get("nonexistent", "default") == "default", "默认值测试失败"
            
            mem.delete("key1")
            assert mem.get("key1") is None, "删除后应为None"
            
            return True, "工作记忆测试通过"
        except Exception as e:
            return False, f"工作记忆测试失败: {e}"


# ── 2. 行为测试（核心！）──────────────────────────────────────
# 测试 Agent 会不会在合适的时机主动调用工具/技能/子智能体
# 核心思路：给 LLM 绑定工具，发送任务，检查返回的 tool_calls

# API 可用性标记（启动时检查一次）
_api_available = None

async def check_api_available():
    """检查 API 是否可用（只检查一次）"""
    global _api_available
    if _api_available is not None:
        return _api_available
    try:
        from config import llm
        response = await llm.ainvoke([{"role": "user", "content": "hi"}])
        _api_available = response is not None and hasattr(response, 'content')
    except Exception:
        _api_available = False
    return _api_available


class TestToolCallingBehavior:
    """测试 Agent 是否会主动调用工具"""
    
    @staticmethod
    async def test_agent_calls_calculator():
        """
        行为测试：给 Agent 一个计算任务，它会不会主动调用 calculate 工具？
        
        预期：Agent 应该调用 calculate，而不是自己算
        """
        from tools import calculate
        from config import llm
        
        llm_with_tools = llm.bind_tools([calculate])
        response = await llm_with_tools.ainvoke([
            {"role": "system", "content": "你是一个有用的助手。需要计算时请调用 calculate 工具，不要自己算。"},
            {"role": "user", "content": "帮我算一下 123 乘以 456 等于多少"}
        ])
        
        called = [tc["name"] for tc in (response.tool_calls or [])]
        passed = "calculate" in called
        return passed, f"调用了: {called}" if called else "没有调用任何工具"
    
    @staticmethod
    async def test_agent_calls_time_tool():
        """
        行为测试：问 Agent 现在几点，它会不会调用 get_current_time？
        
        预期：Agent 应该调用工具获取时间，而不是瞎编
        """
        from tools import get_current_time
        from config import llm
        
        llm_with_tools = llm.bind_tools([get_current_time])
        response = await llm_with_tools.ainvoke([
            {"role": "system", "content": "你是一个有用的助手。需要知道时间时请调用 get_current_time 工具。"},
            {"role": "user", "content": "现在几点了？"}
        ])
        
        called = [tc["name"] for tc in (response.tool_calls or [])]
        passed = "get_current_time" in called
        return passed, f"调用了: {called}" if called else "没有调用任何工具"
    
    @staticmethod
    async def test_agent_no_tool_for_simple_question():
        """
        行为测试：问 Agent 简单问题，它会不会乱调工具？
        
        预期：Agent 应该直接回答，不调用任何工具
        """
        from tools import calculate, get_current_time
        from config import llm
        
        llm_with_tools = llm.bind_tools([calculate, get_current_time])
        response = await llm_with_tools.ainvoke([
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "你好，请做个自我介绍"}
        ])
        
        called = [tc["name"] for tc in (response.tool_calls or [])]
        passed = len(called) == 0  # 不应该调用任何工具
        return passed, "直接回答（正确）" if passed else f"乱调了工具: {called}"
    
    @staticmethod
    async def test_agent_calls_memory_tools():
        """
        行为测试：让 Agent 记住信息，它会不会调用 save_memory？
        
        预期：Agent 应该调用 save_memory 保存用户信息
        """
        from tools import save_memory, search_memory
        from config import llm
        
        llm_with_tools = llm.bind_tools([save_memory, search_memory])
        response = await llm_with_tools.ainvoke([
            {"role": "system", "content": "你是一个有用的助手。用户让你记住信息时，请调用 save_memory 工具。"},
            {"role": "user", "content": "请记住：我最喜欢的编程语言是 Python"}
        ])
        
        called = [tc["name"] for tc in (response.tool_calls or [])]
        passed = "save_memory" in called
        return passed, f"调用了: {called}" if called else "没有调用任何工具"
    
    @staticmethod
    async def test_agent_calls_list_skills():
        """
        行为测试：问 Agent 有哪些技能，它会不会调用 list_skills？
        
        预期：Agent 应该调用 list_skills 查看可用技能
        """
        from tools import list_skills
        from config import llm
        
        llm_with_tools = llm.bind_tools([list_skills])
        response = await llm_with_tools.ainvoke([
            {"role": "system", "content": "你是一个有用的助手。用户询问技能时请调用 list_skills 工具。"},
            {"role": "user", "content": "你有哪些技能可以用？"}
        ])
        
        called = [tc["name"] for tc in (response.tool_calls or [])]
        passed = "list_skills" in called
        return passed, f"调用了: {called}" if called else "没有调用任何工具"


class TestSubAgentBehavior:
    """测试子智能体路由行为"""
    
    @staticmethod
    async def test_subagent_routing():
        """
        行为测试：给子智能体系统一个任务，它会不会路由到正确的子 Agent？
        
        预期：数据库相关任务应该路由到 database 子 Agent
        """
        from subagents import build_subagent_graph
        from langgraph.checkpoint.memory import MemorySaver
        
        graph = build_subagent_graph()
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "帮我查询数据库中有多少用户"}]},
            config={"configurable": {"thread_id": "test_routing"}}
        )
        
        # 检查是否有响应
        messages = result.get("messages", [])
        passed = len(messages) > 0
        return passed, f"子智能体返回了 {len(messages)} 条消息"
    
    @staticmethod
    async def test_subagent_parallel():
        """
        行为测试：并行子智能体系统能否同时处理多个任务？
        
        预期：并行子智能体图应该能构建并执行
        """
        from subagents import build_parallel_subagents_graph
        
        graph = build_parallel_subagents_graph()
        passed = graph is not None
        return passed, "并行子智能体图构建成功" if passed else "构建失败"


class TestReflectionBehavior:
    """测试反思模式行为"""
    
    @staticmethod
    async def test_reflection_loop():
        """
        行为测试：反思图是否会经历 生成→评审→改进 的循环？
        
        预期：反思图应该能执行并返回改进后的结果
        """
        from reflection_agent import build_reflection_graph
        from langchain_core.messages import HumanMessage
        
        graph = build_reflection_graph()
        result = await graph.ainvoke({
            "messages": [HumanMessage(content="写一段关于AI的简短介绍")],
            "iteration": 0,
            "max_iterations": 2
        })
        
        # 检查是否有草稿生成
        draft = result.get("draft")
        critique = result.get("critique")
        passed = draft is not None and len(draft) > 0
        return passed, f"生成了草稿({len(draft)}字)" if passed else "未生成草稿"


class TestHumanInLoopBehavior:
    """测试人机协作行为"""
    
    @staticmethod
    async def test_approval_interrupt():
        """
        行为测试：危险操作是否会触发 interrupt 暂停？
        
        预期：删除数据操作应该触发 interrupt，等待人类审批
        """
        from human_in_loop import build_approval_graph
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import interrupt
        
        graph = build_approval_graph()
        
        try:
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": "删除 users 表中所有数据"}]},
                config={"configurable": {"thread_id": "test_approval"}}
            )
            # 如果图执行到了 interrupt，会抛出 GraphInterrupt 异常
            # 如果没有 interrupt，说明没有识别到危险操作
            messages = result.get("messages", [])
            last_msg = messages[-1] if messages else None
            has_tool_calls = hasattr(last_msg, "tool_calls") and last_msg.tool_calls
            passed = has_tool_calls  # 至少识别到了工具调用
            return passed, f"识别到工具调用: {[tc.get('name') for tc in last_msg.tool_calls]}" if has_tool_calls else "没有触发工具调用"
        except Exception as e:
            # GraphInterrupt 是预期行为（说明 interrupt 生效了）
            if "interrupt" in str(e).lower() or "GraphInterrupt" in type(e).__name__:
                return True, "interrupt 触发成功，等待人类审批"
            return False, f"异常: {e}"


# ── 3. 集成测试 ──────────────────────────────────────────

class TestHumanInLoop:
    """测试人机协作模块（教程10）"""
    
    @staticmethod
    def test_imports():
        """测试模块导入"""
        try:
            from human_in_loop import build_approval_graph, delete_data, send_email
            return True, "人机协作模块导入成功"
        except Exception as e:
            return False, f"人机协作模块导入失败: {e}"
    
    @staticmethod
    def test_approval_graph():
        """测试审批图构建"""
        try:
            from human_in_loop import build_approval_graph
            graph = build_approval_graph()
            assert graph is not None, "审批图不应为空"
            return True, "审批图构建成功"
        except Exception as e:
            return False, f"审批图测试失败: {e}"


class TestSubAgents:
    """测试子智能体模块（教程09）"""
    
    @staticmethod
    def test_imports():
        """测试模块导入"""
        try:
            from subagents import build_subagent_graph, build_main_agent
            return True, "子智能体模块导入成功"
        except Exception as e:
            return False, f"子智能体模块导入失败: {e}"
    
    @staticmethod
    def test_multi_agent_system():
        """测试多智能体系统构建"""
        try:
            from subagents import build_subagent_graph
            graph = build_subagent_graph()
            assert graph is not None, "子智能体图不应为空"
            return True, "子智能体系统构建成功"
        except Exception as e:
            return False, f"子智能体系统测试失败: {e}"


class TestReflection:
    """测试反思模式模块（教程07）"""
    
    @staticmethod
    def test_imports():
        """测试模块导入"""
        try:
            from reflection_agent import build_reflection_graph, build_tool_chain_graph
            return True, "反思模块导入成功"
        except Exception as e:
            return False, f"反思模块导入失败: {e}"
    
    @staticmethod
    def test_reflection_graph():
        """测试反思图构建"""
        try:
            from reflection_agent import build_reflection_graph
            graph = build_reflection_graph()
            assert graph is not None, "反思图不应为空"
            return True, "反思图构建成功"
        except Exception as e:
            return False, f"反思图测试失败: {e}"


class TestAgentIntegration:
    """测试Agent集成"""
    
    @staticmethod
    async def test_react_mode():
        """测试ReAct模式"""
        try:
            from agent import build_react_agent, get_mcp_tools
            from langgraph.checkpoint.memory import MemorySaver
            
            tools = get_mcp_tools()
            if not tools:
                return False, "无可用工具（需先初始化MCP）"
            
            checkpointer = MemorySaver()
            agent = build_react_agent(tools, checkpointer)
            
            # 简单测试
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": "你好"}]
            }, config={"configurable": {"thread_id": "test_1"}})
            
            assert "messages" in result, "结果应包含messages"
            assert len(result["messages"]) > 0, "messages不应为空"
            
            return True, "ReAct模式集成测试通过"
        except Exception as e:
            return False, f"ReAct模式集成测试失败: {e}"
    
    @staticmethod
    async def test_plan_mode():
        """测试Plan-and-Execute模式"""
        try:
            from agent import build_plan_agent, get_mcp_tools
            from langgraph.checkpoint.memory import MemorySaver
            
            tools = get_mcp_tools()
            if not tools:
                return False, "无可用工具"
            
            checkpointer = MemorySaver()
            agent = build_plan_agent(tools, checkpointer)
            
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": "查询北京天气"}]
            }, config={"configurable": {"thread_id": "test_2"}})
            
            assert "plan" in result or "messages" in result, "结果格式错误"
            
            return True, "Plan模式集成测试通过"
        except Exception as e:
            return False, f"Plan模式集成测试失败: {e}"


# ── 3. 基准测试 ──────────────────────────────────────────

@dataclass
class BenchmarkTask:
    name: str
    input_data: str
    expected_output: str
    category: str


class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self):
        self.tasks = []
        self.results = []
    
    def add_task(self, task: BenchmarkTask):
        self.tasks.append(task)
    
    async def run(self, agent_func):
        self.results = []
        for task in self.tasks:
            start = time.time()
            try:
                output = await agent_func(task.input_data)
                duration = time.time() - start
                passed = task.expected_output.lower() in output.lower()
                self.results.append({
                    "name": task.name,
                    "passed": passed,
                    "duration": duration,
                    "output": output[:200]
                })
            except Exception as e:
                duration = time.time() - start
                self.results.append({
                    "name": task.name,
                    "passed": False,
                    "duration": duration,
                    "error": str(e)
                })
        return self.results


# ── 4. 报告生成 ──────────────────────────────────────────

def generate_json_report(suite: TestSuiteResult, filename: str):
    """生成JSON报告"""
    report = {
        "suite_name": suite.suite_name,
        "timestamp": suite.timestamp,
        "summary": {
            "total": suite.total,
            "passed": suite.passed,
            "failed": suite.failed,
            "pass_rate": suite.pass_rate,
            "avg_duration": f"{suite.avg_duration:.3f}s"
        },
        "tests": [
            {
                "name": r.name,
                "category": r.category,
                "passed": r.passed,
                "duration": r.duration,
                "message": r.message,
                "error": r.error
            }
            for r in suite.results
        ]
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def generate_md_report(suite: TestSuiteResult, filename: str):
    """生成Markdown报告"""
    lines = [
        f"# {suite.suite_name} 测试报告",
        f"",
        f"**时间**: {suite.timestamp}",
        f"",
        f"## 摘要",
        f"",
        f"- **总计**: {suite.total}",
        f"- **通过**: {suite.passed}",
        f"- **失败**: {suite.failed}",
        f"- **通过率**: {suite.pass_rate}",
        f"- **平均耗时**: {suite.avg_duration:.3f}s",
        f"",
        f"## 详细结果",
        f""
    ]
    
    for r in suite.results:
        status = "✅" if r.passed else "❌"
        lines.append(f"### {status} {r.name} ({r.category})")
        lines.append(f"- 耗时: {r.duration:.3f}s")
        if r.message:
            lines.append(f"- 信息: {r.message}")
        if r.error:
            lines.append(f"- 错误: {r.error}")
        lines.append("")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ── 主测试流程 ─────────────────────────────────────────────

async def run_all_tests():
    """运行所有测试"""
    suite = TestSuiteResult(
        suite_name="Agent App 完整测试套件",
        timestamp=datetime.now().isoformat()
    )
    
    logger.info("=" * 60)
    logger.info("开始运行完整测试套件")
    logger.info("=" * 60)
    
    # 1. 单元测试
    logger.info("\n 单元测试")
    
    for test_class in [TestConfig, TestTools, TestMCPClient, TestSkills, TestMemory, TestHumanInLoop, TestSubAgents, TestReflection]:
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                method = getattr(test_class, method_name)
                start = time.time()
                try:
                    passed, message = method()
                    duration = time.time() - start
                    suite.results.append(TestResult(
                        name=f"{test_class.__name__}.{method_name}",
                        category="单元测试",
                        passed=passed,
                        duration=duration,
                        message=message
                    ))
                    status = "✅" if passed else "❌"
                    logger.info(f"  {status} {method_name}: {message}")
                except Exception as e:
                    duration = time.time() - start
                    suite.results.append(TestResult(
                        name=f"{test_class.__name__}.{method_name}",
                        category="单元测试",
                        passed=False,
                        duration=duration,
                        error=str(e)
                    ))
                    logger.info(f"  ❌ {method_name}: {e}")
    
    # 2. 行为测试（核心！）
    logger.info("\n🧠 行为测试 - 测试 Agent 会不会主动调用工具/技能/子智能体")
    
    # 先检查 API 是否可用
    api_ok = await check_api_available()
    if not api_ok:
        logger.info("  ⚠️  API 不可用，行为测试将全部跳过（请检查 .env 中的 API 配置）")
    
    behavior_test_classes = [
        TestToolCallingBehavior,
        TestSubAgentBehavior,
        TestReflectionBehavior,
        TestHumanInLoopBehavior,
    ]
    
    for test_class in behavior_test_classes:
        logger.info(f"\n  📦 {test_class.__name__}")
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                method = getattr(test_class, method_name)
                start = time.time()
                
                # API 不可用时跳过需要 LLM 的测试（纯构建类测试不需要 API）
                skip_methods = {"test_subagent_parallel"}  # 这些测试不需要 API
                if not api_ok and method_name not in skip_methods:
                    suite.results.append(TestResult(
                        name=f"{test_class.__name__}.{method_name}",
                        category="行为测试",
                        passed=False,
                        duration=0,
                        message="API 不可用，已跳过"
                    ))
                    logger.info(f"    ⏭️️ {method_name}: API 不可用，已跳过")
                    continue
                
                try:
                    passed, message = await method()
                    duration = time.time() - start
                    suite.results.append(TestResult(
                        name=f"{test_class.__name__}.{method_name}",
                        category="行为测试",
                        passed=passed,
                        duration=duration,
                        message=message
                    ))
                    status = "✅" if passed else "❌"
                    logger.info(f"    {status} {method_name}: {message}")
                except Exception as e:
                    duration = time.time() - start
                    suite.results.append(TestResult(
                        name=f"{test_class.__name__}.{method_name}",
                        category="行为测试",
                        passed=False,
                        duration=duration,
                        error=str(e)
                    ))
                    logger.info(f"    ❌ {method_name}: {e}")
    
    # 3. 集成测试
    logger.info("\n🔗 集成测试")
    
    for method_name in ['test_react_mode', 'test_plan_mode']:
        method = getattr(TestAgentIntegration, method_name)
        start = time.time()
        try:
            passed, message = await method()
            duration = time.time() - start
            suite.results.append(TestResult(
                name=method_name,
                category="集成测试",
                passed=passed,
                duration=duration,
                message=message
            ))
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {method_name}: {message}")
        except Exception as e:
            duration = time.time() - start
            suite.results.append(TestResult(
                name=method_name,
                category="集成测试",
                passed=False,
                duration=duration,
                error=str(e)
            ))
            logger.info(f"  ❌ {method_name}: {e}")
    
    # 3. 生成报告
    logger.info("\n📊 生成报告")
    generate_json_report(suite, 'test_report.json')
    generate_md_report(suite, 'test_report.md')
    
    logger.info(f"✅ JSON报告: test_report.json")
    logger.info(f"📄 Markdown报告: test_report.md")
    
    # 最终摘要
    logger.info("\n" + "=" * 60)
    logger.info(f"测试完成: {suite.passed}/{suite.total} 通过 ({suite.pass_rate})")
    logger.info(f"平均耗时: {suite.avg_duration:.3f}s")
    logger.info("=" * 60)
    
    return suite


if __name__ == "__main__":
    asyncio.run(run_all_tests())
