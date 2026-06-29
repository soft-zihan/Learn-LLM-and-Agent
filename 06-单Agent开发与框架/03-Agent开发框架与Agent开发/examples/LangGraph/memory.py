"""
记忆系统 - 对应教程 06-记忆系统设计

本模块实现了三种记忆类型，模拟人类的记忆机制：

1. 短期记忆（ShortTermMemory）
   - 存储当前对话的消息列表
   - 类似人类的"工作记忆"
   - 有容量限制（默认50条），超出后丢弃最早的消息

2. 长期记忆（LongTermMemory）
   - 跨对话持久化存储用户相关的事实
   - 类似人类的"长期记忆"
   - 使用 JSON 文件持久化到磁盘
   - 支持按用户ID隔离、关键词搜索

3. 工作记忆（WorkingMemory）
   - 临时键值存储，用于任务执行中的中间状态
   - 类似人类的"便签纸"
   - 不持久化，应用重启后清空

教程对应：
- 06-记忆系统设计：三种记忆类型的架构设计
- 01-LangGraph核心概念：MemorySaver 检查点（对话级记忆）
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


class ShortTermMemory:
    """
    短期记忆：存储当前对话的消息列表
    
    特点：
    - 内存存储，速度快
    - 有容量限制（FIFO 淘汰策略）
    - 应用重启后清空
    
    使用场景：
    - 保持对话连贯性
    - 为 LLM 提供上下文窗口
    """
    
    def __init__(self, max_messages: int = 50):
        """
        Args:
            max_messages: 最大消息数量，超出后丢弃最早的消息
        """
        self.max_messages = max_messages
        self._messages: list[dict] = []
    
    def add(self, role: str, content: str):
        """
        添加一条消息到短期记忆
        
        Args:
            role: 消息角色（"user"、"assistant"、"system"）
            content: 消息内容
        """
        self._messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # 超出容量时，丢弃最早的消息（FIFO）
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]
    
    def get_recent(self, n: int = 10) -> list:
        """获取最近 N 条消息"""
        return self._messages[-n:]
    
    def get_all(self) -> list:
        """获取所有消息"""
        return self._messages
    
    def clear(self):
        """清空所有消息"""
        self._messages = []


class LongTermMemory:
    """
    长期记忆：跨对话持久化存储用户事实
    
    特点：
    - JSON 文件持久化，应用重启后保留
    - 按 user_id 隔离（每个用户一个文件）
    - 支持关键词搜索
    
    使用场景：
    - 记住用户偏好（"我喜欢简洁的回复"）
    - 记住用户信息（"我叫张三"）
    - 跨对话保持个性化
    
    安全设计：
    - user_id 经过安全过滤，防止路径穿越攻击
    """
    
    def __init__(self, storage_dir: str = None):
        """
        Args:
            storage_dir: 存储目录，默认为模块同级的 memory/ 目录
        """
        if storage_dir is None:
            storage_dir = os.path.join(os.path.dirname(__file__), "memory_data")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file(self, user_id: str) -> Path:
        """
        获取用户记忆文件路径
        
        安全设计：对 user_id 进行安全过滤，防止路径穿越攻击
        例如：user_id = "../../etc/passwd" 会被过滤为 "etcpasswd"
        
        Args:
            user_id: 用户唯一标识
        
        Returns:
            安全的文件路径
        
        Raises:
            ValueError: 如果过滤后的 user_id 为空
        """
        # 防止路径穿越：只允许字母、数字、连字符、下划线、点
        safe_name = re.sub(r'[^\w\-.]', '', user_id)
        if not safe_name:
            raise ValueError(f"Invalid user_id: {user_id}")
        return self.storage_dir / f"{safe_name}.json"
    
    def save_fact(self, user_id: str, fact: str):
        """
        保存一条事实到长期记忆
        
        Args:
            user_id: 用户唯一标识
            fact: 要保存的事实内容
        """
        file = self._get_file(user_id)
        facts = self.load_facts(user_id)
        facts.append({
            "content": fact,
            "timestamp": datetime.now().isoformat()
        })
        file.write_text(json.dumps(facts, ensure_ascii=False, indent=2))
    
    def load_facts(self, user_id: str) -> list:
        """
        加载用户的所有记忆事实
        
        Args:
            user_id: 用户唯一标识
        
        Returns:
            事实列表，每条包含 content 和 timestamp
        """
        file = self._get_file(user_id)
        if file.exists():
            return json.loads(file.read_text(encoding='utf-8'))
        return []
    
    def search(self, user_id: str, query: str) -> list:
        """
        关键词搜索用户记忆
        
        Args:
            user_id: 用户唯一标识
            query: 搜索关键词（不区分大小写）
        
        Returns:
            匹配的事实列表
        """
        facts = self.load_facts(user_id)
        query_lower = query.lower()
        return [f for f in facts if query_lower in f["content"].lower()]
    
    def get_summary(self, user_id: str) -> str:
        """
        获取用户记忆的摘要（用于注入到系统提示中）
        
        Args:
            user_id: 用户唯一标识
        
        Returns:
            格式化的记忆摘要字符串
        """
        facts = self.load_facts(user_id)
        if not facts:
            return "暂无记忆"
        lines = ["## 用户记忆", ""]
        for fact in facts[-10:]:  # 只展示最近10条
            lines.append(f"- {fact['content']}")
        return "\n".join(lines)


class WorkingMemory:
    """
    工作记忆：临时键值存储
    
    特点：
    - 内存存储，不持久化
    - 简单的 key-value 接口
    - 用于任务执行中的中间状态
    
    使用场景：
    - 存储当前正在处理的子任务结果
    - 存储临时的计算中间值
    - 跨节点传递临时数据
    """
    
    def __init__(self):
        self._data: dict = {}
    
    def set(self, key: str, value):
        """存储一个键值对"""
        self._data[key] = value
    
    def get(self, key: str, default=None):
        """获取一个值"""
        return self._data.get(key, default)
    
    def delete(self, key: str):
        """删除一个键"""
        self._data.pop(key, None)
    
    def clear(self):
        """清空所有数据"""
        self._data = {}
    
    def to_dict(self) -> dict:
        """导出为字典"""
        return dict(self._data)


# ── 全局实例（模块级别单例）──────────────────────────────────
# 这些实例在应用启动时创建，所有模块共享

short_term_memory = ShortTermMemory()
long_term_memory = LongTermMemory()
working_memory = WorkingMemory()


def init_memory():
    """初始化记忆系统（应用启动时调用）"""
    print("[Memory] 记忆系统初始化完成")
    print(f"  - 短期记忆容量：{short_term_memory.max_messages} 条")
    print(f"  - 长期记忆目录：{long_term_memory.storage_dir}")
