# ReAct 框架与设计模式

> 📅 **更新时间**: 2026-06-17  

---

## 目录

- [1. 记忆系统设计](#1-记忆系统设计)
- [2. 工具使用深度实践](#2-工具使用深度实践)
- [3. 规划与推理](#3-规划与推理)

---

## 1. 记忆系统设计

记忆系统是 Agent 的核心组件之一，决定了 Agent 能否进行长期学习和个性化服务。

### 1.1 短期记忆

#### 对话历史

```python
# 对话历史管理

from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, 
    trim_messages
)
from langchain_core.language_models import BaseChatModel

class ConversationHistory:
    """对话历史管理器"""
    
    def __init__(
        self,
        max_messages: int = 20,
        max_tokens: int = 4000,
        include_system_prompt: bool = True
    ):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.include_system_prompt = include_system_prompt
        self.messages: List[BaseMessage] = []
        self.system_prompt: Optional[SystemMessage] = None
    
    def set_system_prompt(self, prompt: str):
        """设置系统提示"""
        self.system_prompt = SystemMessage(content=prompt)
    
    def add_message(self, message: BaseMessage):
        """添加消息"""
        self.messages.append(message)
        self._enforce_limits()
    
    def add_user_message(self, content: str):
        """添加用户消息"""
        self.add_message(HumanMessage(content=content))
    
    def add_ai_message(self, content: str):
        """添加 AI 消息"""
        self.add_message(AIMessage(content=content))
    
    def get_messages(self) -> List[BaseMessage]:
        """获取消息列表"""
        messages = []
        if self.include_system_prompt and self.system_prompt:
            messages.append(self.system_prompt)
        messages.extend(self.messages)
        return messages
    
    def _enforce_limits(self):
        """强制执行限制"""
        # 限制消息数量
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        # 限制 token 数量
        while self._estimate_tokens() > self.max_tokens:
            if len(self.messages) <= 2:
                break
            # 移除最早的消息（保留最近的）
            self.messages.pop(0)
    
    def _estimate_tokens(self) -> int:
        """估算 token 数量"""
        total_chars = sum(len(msg.content) for msg in self.messages)
        return int(total_chars / 4)  # 粗略估算：4 字符 ≈ 1 token
    
    def clear(self):
        """清空历史"""
        self.messages = []
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "message_count": len(self.messages),
            "estimated_tokens": self._estimate_tokens(),
            "max_messages": self.max_messages,
            "max_tokens": self.max_tokens
        }

# 使用 LangChain 的 trim_messages
def trim_conversation(messages: List[BaseMessage], max_tokens: int = 4000) -> List[BaseMessage]:
    """裁剪对话历史"""
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",  # 保留最后的消息
        token_counter=len,  # 简单计数
        include_system=True
    )
```

#### 摘要压缩

```python
# 对话摘要压缩

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5.2", temperature=0)

# 摘要 Prompt
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个对话摘要专家。请将以下对话历史压缩为简洁的摘要。

要求：
1. 保留关键信息和用户意图
2. 保留重要的事实和数据
3. 忽略寒暄和重复内容
4. 摘要应该连贯、易读
5. 使用中文

输出摘要："""),
    ("human", "{conversation}")
])

summarizer = summary_prompt | llm

class SummarizingMemory:
    """带摘要的记忆"""
    
    def __init__(
        self,
        llm: BaseChatModel,
        summary_threshold: int = 15,  # 消息数超过此值时触发摘要
        recent_messages: int = 5  # 保留最近的消息数量
    ):
        self.llm = llm
        self.summary_threshold = summary_threshold
        self.recent_messages = recent_messages
        self.messages: List[BaseMessage] = []
        self.summary: str = ""
        self.summarizer = summary_prompt | llm
    
    def add_message(self, message: BaseMessage):
        """添加消息"""
        self.messages.append(message)
        
        # 检查是否需要摘要
        if len(self.messages) > self.summary_threshold:
            self._summarize()
    
    def get_context(self) -> List[BaseMessage]:
        """获取上下文"""
        context = []
        
        # 添加摘要
        if self.summary:
            context.append(SystemMessage(content=f"之前的对话摘要：\n{self.summary}"))
        
        # 添加最近的消息
        context.extend(self.messages[-self.recent_messages:])
        
        return context
    
    def _summarize(self):
        """生成摘要"""
        # 需要摘要的消息（排除最近的消息）
        old_messages = self.messages[:-self.recent_messages]
        messages_text = "\n".join([
            f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in old_messages
        ])
        
        # 生成新摘要
        if self.summary:
            # 如果有旧摘要，合并生成
            new_summary = self.summarizer.invoke({
                "conversation": f"之前的摘要：\n{self.summary}\n\n新对话：\n{messages_text}"
            }).content
        else:
            new_summary = self.summarizer.invoke({
                "conversation": messages_text
            }).content
        
        # 更新摘要
        self.summary = new_summary
        
        # 保留最近的消息
        self.messages = self.messages[-self.recent_messages:]
        
        print(f"对话已摘要，摘要长度：{len(self.summary)} 字符")
    
    def clear(self):
        """清空记忆"""
        self.messages = []
        self.summary = ""

# 使用示例
memory = SummarizingMemory(llm, summary_threshold=10, recent_messages=5)

# 模拟对话
for i in range(20):
    memory.add_message(HumanMessage(content=f"问题 {i+1}"))
    memory.add_message(AIMessage(content=f"回答 {i+1}"))

# 获取上下文
context = memory.get_context()
print(f"上下文消息数量：{len(context)}")
print(f"摘要：{memory.summary}")
```

#### 滑动窗口

```python
# 滑动窗口记忆实现

from collections import deque
from langchain_core.messages import BaseMessage

class SlidingWindowMemory:
    """滑动窗口记忆"""
    
    def __init__(
        self,
        window_size: int = 10,
        return_messages: bool = True
    ):
        self.window_size = window_size
        self.return_messages = return_messages
        self.buffer = deque(maxlen=window_size)
    
    def add_message(self, message: BaseMessage):
        """添加消息"""
        self.buffer.append(message)
    
    def get_messages(self) -> List[BaseMessage]:
        """获取消息"""
        return list(self.buffer)
    
    def clear(self):
        """清空"""
        self.buffer.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "window_size": self.window_size,
            "message_count": len(self.buffer),
            "messages": [m.dict() for m in self.buffer]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SlidingWindowMemory":
        """从字典恢复"""
        memory = cls(window_size=data["window_size"])
        for msg_data in data["messages"]:
            # 恢复消息
            if msg_data["type"] == "human":
                memory.add_message(HumanMessage(content=msg_data["content"]))
            elif msg_data["type"] == "ai":
                memory.add_message(AIMessage(content=msg_data["content"]))
        return memory

# 动态滑动窗口（根据 token 数量调整）
class DynamicSlidingWindowMemory:
    """动态滑动窗口记忆"""
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.messages: List[BaseMessage] = []
    
    def add_message(self, message: BaseMessage):
        """添加消息"""
        self.messages.append(message)
        self._adjust_window()
    
    def _adjust_window(self):
        """调整窗口大小"""
        while self._estimate_tokens() > self.max_tokens and len(self.messages) > 2:
            self.messages.pop(0)  # 移除最早的消息
    
    def _estimate_tokens(self) -> int:
        """估算 token"""
        return sum(len(m.content) // 4 for m in self.messages)
    
    def get_messages(self) -> List[BaseMessage]:
        """获取消息"""
        return self.messages
    
    def get_current_window_size(self) -> int:
        """获取当前窗口大小"""
        return len(self.messages)
```

### 1.2 长期记忆

#### 向量数据库

```python
# 基于向量数据库的长期记忆

from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
import hashlib

class VectorMemory:
    """向量数据库长期记忆"""
    
    def __init__(
        self,
        persist_directory: str = "./long_term_memory",
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "agent_memory"
    ):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # 初始化向量存储
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
    
    def store(
        self,
        key: str,
        value: str,
        metadata: Optional[dict] = None,
        importance: float = 0.5
    ):
        """存储记忆"""
        # 生成唯一 ID
        doc_id = hashlib.md5(key.encode()).hexdigest()
        
        # 构建元数据
        doc_metadata = {
            "key": key,
            "importance": importance,
            "timestamp": self._get_timestamp(),
            **(metadata or {})
        }
        
        # 创建文档
        doc = Document(
            page_content=value,
            metadata=doc_metadata
        )
        
        # 添加到向量存储
        self.vectorstore.add_documents([doc], ids=[doc_id])
        
        print(f"记忆已存储：{key[:50]}...")
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_importance: float = 0.0
    ) -> List[Document]:
        """检索记忆"""
        # 相似度搜索
        docs = self.vectorstore.similarity_search(query, k=k * 2)
        
        # 过滤低重要性记忆
        filtered_docs = [
            doc for doc in docs
            if doc.metadata.get("importance", 0) >= min_importance
        ]
        
        # 按相关性排序
        return filtered_docs[:k]
    
    def update(self, key: str, new_value: str, new_metadata: Optional[dict] = None):
        """更新记忆"""
        # 删除旧记忆
        doc_id = hashlib.md5(key.encode()).hexdigest()
        try:
            self.vectorstore.delete([doc_id])
        except:
            pass
        
        # 存储新记忆
        metadata = {
            "key": key,
            "updated": True,
            **(new_metadata or {})
        }
        self.store(key, new_value, metadata)
    
    def delete(self, key: str) -> bool:
        """删除记忆"""
        doc_id = hashlib.md5(key.encode()).hexdigest()
        try:
            self.vectorstore.delete([doc_id])
            return True
        except:
            return False
    
    def get_all_memories(self, limit: int = 100) -> List[Document]:
        """获取所有记忆"""
        # Chroma 不支持直接获取所有，需要查询
        return self.vectorstore.similarity_search("", k=limit)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        all_docs = self.get_all_memories(limit=10000)
        return {
            "total_memories": len(all_docs),
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()

# 使用示例
memory = VectorMemory(persist_directory="./memory")

# 存储记忆
memory.store(
    key="user_preference",
    value="用户喜欢简洁的回答，不喜欢冗长的解释",
    metadata={"type": "preference", "user_id": "user_001"},
    importance=0.8
)

memory.store(
    key="fact_001",
    value="Python 是一种解释型、面向对象、动态数据类型的高级程序设计语言",
    metadata={"type": "fact", "category": "programming"},
    importance=0.6
)

# 检索记忆
query = "用户有什么偏好？"
relevant_memories = memory.retrieve(query, k=3)

for doc in relevant_memories:
    print(f"记忆：{doc.page_content}")
    print(f"重要性：{doc.metadata.get('importance')}")
    print("---")
```

#### 知识图谱

```python
# 基于知识图谱的长期记忆（简化实现）

from typing import Dict, List, Set, Tuple
import json

class KnowledgeGraphMemory:
    """知识图谱记忆"""
    
    def __init__(self, persist_file: str = "./knowledge_graph.json"):
        self.persist_file = persist_file
        # 实体：{实体名：{属性}}
        self.entities: Dict[str, Dict[str, str]] = {}
        # 关系：{(实体 1, 关系，实体 2): 属性}
        self.relationships: Dict[Tuple[str, str, str], Dict] = {}
        
        # 加载已有数据
        self._load()
    
    def add_entity(self, name: str, attributes: Dict[str, str]):
        """添加实体"""
        if name not in self.entities:
            self.entities[name] = {}
        self.entities[name].update(attributes)
        self._save()
    
    def add_relationship(
        self,
        entity1: str,
        relationship: str,
        entity2: str,
        attributes: Dict = None
    ):
        """添加关系"""
        # 确保实体存在
        if entity1 not in self.entities:
            self.entities[entity1] = {}
        if entity2 not in self.entities:
            self.entities[entity2] = {}
        
        # 存储关系
        self.relationships[(entity1, relationship, entity2)] = attributes or {}
        self._save()
    
    def query_entity(self, name: str) -> Optional[Dict]:
        """查询实体"""
        return self.entities.get(name)
    
    def query_relationships(
        self,
        entity: str,
        relationship: str = None
    ) -> List[Tuple[str, str, str]]:
        """查询关系"""
        results = []
        for (e1, rel, e2), attrs in self.relationships.items():
            if e1 == entity and (relationship is None or rel == relationship):
                results.append((e1, rel, e2, attrs))
            elif e2 == entity and (relationship is None or rel == relationship):
                results.append((e1, rel, e2, attrs))
        return results
    
    def find_path(self, entity1: str, entity2: str, max_depth: int = 3) -> List:
        """查找两个实体间的路径"""
        # BFS 搜索
        from collections import deque
        
        queue = deque([(entity1, [entity1])])
        visited = {entity1}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            # 查找邻居
            for (e1, rel, e2), attrs in self.relationships.items():
                neighbor = None
                if e1 == current:
                    neighbor = e2
                elif e2 == current:
                    neighbor = e1
                
                if neighbor and neighbor not in visited:
                    new_path = path + [(rel, neighbor)]
                    if neighbor == entity2:
                        return new_path
                    
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        
        return []
    
    def _save(self):
        """保存到文件"""
        data = {
            "entities": self.entities,
            "relationships": {
                f"{k[0]}|{k[1]}|{k[2]}": v
                for k, v in self.relationships.items()
            }
        }
        
        with open(self.persist_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load(self):
        """从文件加载"""
        import os
        if not os.path.exists(self.persist_file):
            return
        
        with open(self.persist_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.entities = data.get("entities", {})
        self.relationships = {
            tuple(k.split("|")): v
            for k, v in data.get("relationships", {}).items()
        }

# 使用 LLM 自动构建知识图谱
from langchain_core.prompts import ChatPromptTemplate

kg_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识图谱构建专家。请从以下文本中提取实体和关系。

输出格式（JSON）：
```json
{{
    "entities": [
        {{"name": "实体名", "type": "实体类型", "attributes": {{}}}}
    ],
    "relationships": [
        {{"entity1": "实体 1", "relationship": "关系", "entity2": "实体 2"}}
    ]
}}
```

实体类型：PERSON（人）, ORGANIZATION（组织）, LOCATION（地点）, CONCEPT（概念）, EVENT（事件）

关系类型：WORKS_AT（工作于）, LOCATED_IN（位于）, KNOWS（认识）, STUDIED（学习）, CREATED（创建）"""),
    ("human", "{text}")
])

kg_extractor = kg_extraction_prompt | llm

def build_kg_from_text(text: str, kg_memory: KnowledgeGraphMemory):
    """从文本构建知识图谱"""
    import json
    
    # 提取实体和关系
    response = kg_extractor.invoke({"text": text})
    
    try:
        json_str = response.content.split("```json")[1].split("```")[0]
        data = json.loads(json_str.strip())
        
        # 添加实体
        for entity in data.get("entities", []):
            kg_memory.add_entity(
                entity["name"],
                {"type": entity["type"], **entity.get("attributes", {})}
            )
        
        # 添加关系
        for rel in data.get("relationships", []):
            kg_memory.add_relationship(
                rel["entity1"],
                rel["relationship"],
                rel["entity2"]
            )
        
        print(f"已提取 {len(data.get('entities', []))} 个实体和 {len(data.get('relationships', []))} 个关系")
    except Exception as e:
        print(f"提取失败：{e}")

# 使用示例
kg = KnowledgeGraphMemory()

text = """
马斯克是特斯拉和 SpaceX 的 CEO。特斯拉总部位于美国德州奥斯汀。
SpaceX 致力于火星探索。马斯克还在 2022 年收购了 Twitter。
"""

build_kg_from_text(text, kg)

# 查询
print(kg.query_entity("马斯克"))
print(kg.query_relationships("马斯克"))
```

#### 记忆检索

```python
# 高级记忆检索策略

from langchain_core.documents import Document
from typing import List, Optional
import numpy as np

class AdvancedMemoryRetriever:
    """高级记忆检索器"""
    
    def __init__(self, vector_memory: VectorMemory):
        self.vector_memory = vector_memory
    
    def retrieve_with_recency(
        self,
        query: str,
        k: int = 5,
        recency_weight: float = 0.3
    ) -> List[Document]:
        """结合相关性和新近度的检索"""
        from datetime import datetime
        
        # 获取候选记忆
        candidates = self.vector_memory.retrieve(query, k=k * 2)
        
        # 计算新近度分数
        now = datetime.now()
        scored_docs = []
        for doc in candidates:
            timestamp = doc.metadata.get("timestamp", "")
            if timestamp:
                try:
                    doc_time = datetime.fromisoformat(timestamp)
                    days_diff = (now - doc_time).days
                    recency_score = max(0, 1 - days_diff / 365)  # 一年内新近度
                except:
                    recency_score = 0.5
            else:
                recency_score = 0.5
            
            # 综合分数（简化：假设文档已按相关性排序）
            combined_score = (1 - recency_weight) * 1.0 + recency_weight * recency_score
            scored_docs.append((doc, combined_score))
        
        # 按综合分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:k]]
    
    def retrieve_with_importance(
        self,
        query: str,
        k: int = 5,
        importance_threshold: float = 0.5
    ) -> List[Document]:
        """结合重要性的检索"""
        candidates = self.vector_memory.retrieve(query, k=k * 3)
        
        # 过滤并排序
        filtered = [
            doc for doc in candidates
            if doc.metadata.get("importance", 0) >= importance_threshold
        ]
        
        # 按重要性排序
        filtered.sort(
            key=lambda x: x.metadata.get("importance", 0),
            reverse=True
        )
        
        return filtered[:k]
    
    def retrieve_diverse(
        self,
        query: str,
        k: int = 5,
        diversity_threshold: float = 0.7
    ) -> List[Document]:
        """多样化检索（避免重复）"""
        from sentence_transformers import util
        
        candidates = self.vector_memory.retrieve(query, k=k * 3)
        
        if not candidates:
            return []
        
        # 选择第一个（最相关）
        selected = [candidates[0]]
        
        # 迭代选择多样化的文档
        for doc in candidates[1:]:
            if len(selected) >= k:
                break
            
            # 计算与已选文档的相似度
            max_similarity = 0
            for sel_doc in selected:
                # 简化：使用文本重叠度
                similarity = self._text_overlap(doc.page_content, sel_doc.page_content)
                max_similarity = max(max_similarity, similarity)
            
            # 如果足够不同，加入
            if max_similarity < diversity_threshold:
                selected.append(doc)
        
        return selected
    
    def _text_overlap(self, text1: str, text2: str) -> float:
        """计算文本重叠度（简化版）"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)

# 使用示例
retriever = AdvancedMemoryRetriever(memory)

# 新近度加权检索
recent_relevant = retriever.retrieve_with_recency(
    "用户偏好",
    k=3,
    recency_weight=0.4
)

# 重要性检索
important = retriever.retrieve_with_importance(
    "重要信息",
    k=5,
    importance_threshold=0.7
)

# 多样化检索
diverse = retriever.retrieve_diverse(
    "主题",
    k=5,
    diversity_threshold=0.6
)
```

### 1.3 记忆管理策略

#### 重要性评分

```python
# 记忆重要性自动评分

from langchain_core.prompts import ChatPromptTemplate

importance_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个记忆重要性评估专家。请评估以下信息的重要性（0-1）。

评估标准：
1. **实用性**：对未来任务是否有帮助
2. **独特性**：是否是独特或罕见的信息
3. **时效性**：信息的有效期
4. **可检索性**：是否容易被重新获取

输出 JSON：
```json
{{
    "importance": 0.8,
    "reasons": ["原因 1", "原因 2"],
    "expiry_days": 365  // 建议过期天数
}}
```"""),
    ("human", "信息：{content}")
])

importance_scorer = importance_prompt | llm

class ImportanceScorer:
    """重要性评分器"""
    
    def __init__(self, llm, cache_importance: bool = True):
        self.llm = llm
        self.cache = {} if cache_importance else None
        self.scorer = importance_prompt | llm
    
    def score(self, content: str, force_recalculate: bool = False) -> dict:
        """评分"""
        # 检查缓存
        if self.cache is not None and not force_recalculate:
            content_hash = hash(content)
            if content_hash in self.cache:
                return self.cache[content_hash]
        
        # 调用 LLM 评分
        import json
        try:
            response = self.scorer.invoke({"content": content})
            json_str = response.content.split("```json")[1].split("```")[0]
            result = json.loads(json_str.strip())
            
            # 缓存
            if self.cache is not None:
                self.cache[hash(content)] = result
            
            return result
        except:
            return {"importance": 0.5, "reasons": ["评分失败"], "expiry_days": 365}
    
    def batch_score(self, contents: List[str]) -> List[dict]:
        """批量评分"""
        return [self.score(c) for c in contents]

# 使用示例
scorer = ImportanceScorer(llm)

memories = [
    "用户喜欢简洁的回答",
    "今天是晴天",
    "Python 是一种编程语言",
    "用户的项目截止日期是 2024-12-31"
]

for mem in memories:
    result = scorer.score(mem)
    print(f"记忆：{mem[:30]}...")
    print(f"重要性：{result['importance']}")
    print(f"原因：{result['reasons']}")
    print("---")
```

#### 遗忘机制

```python
# 记忆遗忘机制实现

from datetime import datetime, timedelta
from langchain_core.documents import Document

class ForgettingMechanism:
    """记忆遗忘机制"""
    
    def __init__(
        self,
        vector_memory: VectorMemory,
        forgetting_curve_half_life: int = 30,  # 半衰期（天）
        min_importance: float = 0.1,
        cleanup_interval_days: int = 7
    ):
        self.vector_memory = vector_memory
        self.half_life = forgetting_curve_half_life
        self.min_importance = min_importance
        self.cleanup_interval = timedelta(days=cleanup_interval_days)
        self.last_cleanup = datetime.now()
    
    def calculate_retention_strength(self, memory: Document) -> float:
        """计算记忆保留强度（基于艾宾浩斯遗忘曲线）"""
        timestamp = memory.metadata.get("timestamp")
        importance = memory.metadata.get("importance", 0.5)
        
        if not timestamp:
            return importance
        
        try:
            mem_time = datetime.fromisoformat(timestamp)
            days_passed = (datetime.now() - mem_time).days
            
            # 艾宾浩斯遗忘曲线：R = e^(-t/S)
            # S 是半衰期
            import math
            retention = math.exp(-days_passed / self.half_life)
            
            # 结合初始重要性
            return importance * retention
        except:
            return importance
    
    def should_forget(self, memory: Document) -> bool:
        """判断是否应该遗忘"""
        strength = self.calculate_retention_strength(memory)
        return strength < self.min_importance
    
    def cleanup(self):
        """清理遗忘的记忆"""
        # 检查是否需要清理
        if datetime.now() - self.last_cleanup < self.cleanup_interval:
            return
        
        print("开始记忆清理...")
        
        # 获取所有记忆
        all_memories = self.vector_memory.get_all_memories(limit=10000)
        
        forgotten_count = 0
        for memory in all_memories:
            if self.should_forget(memory):
                # 删除记忆
                doc_id = memory.metadata.get("doc_id")
                if doc_id:
                    try:
                        self.vector_memory.delete(memory.metadata["key"])
                        forgotten_count += 1
                    except:
                        pass
        
        self.last_cleanup = datetime.now()
        print(f"清理完成，遗忘了 {forgotten_count} 条记忆")
    
    def get_memory_decay_report(self) -> dict:
        """获取记忆衰减报告"""
        all_memories = self.vector_memory.get_all_memories(limit=10000)
        
        strengths = [self.calculate_retention_strength(m) for m in all_memories]
        
        if not strengths:
            return {"total_memories": 0}
        
        import numpy as np
        return {
            "total_memories": len(all_memories),
            "average_strength": float(np.mean(strengths)),
            "min_strength": float(np.min(strengths)),
            "max_strength": float(np.max(strengths)),
            "at_risk_count": sum(1 for s in strengths if s < self.min_importance * 2)
        }

# 使用示例
forgetter = ForgettingMechanism(memory, half_life=30, min_importance=0.1)

# 查看衰减报告
report = forgetter.get_memory_decay_report()
print(f"总记忆数：{report['total_memories']}")
print(f"平均强度：{report['average_strength']:.2f}")
print(f"濒临遗忘：{report['at_risk_count']}")

# 执行清理
forgetter.cleanup()
```

#### 记忆整合

```python
# 记忆整合（合并相似记忆）

from langchain_core.documents import Document
from typing import List

class MemoryIntegrator:
    """记忆整合器"""
    
    def __init__(self, llm, similarity_threshold: float = 0.8):
        self.llm = llm
        self.similarity_threshold = similarity_threshold
    
    def integrate_similar_memories(
        self,
        memories: List[Document],
        key_field: str = "key"
    ) -> List[Document]:
        """整合相似记忆"""
        # 按相似度分组
        groups = self._cluster_similar_memories(memories)
        
        integrated = []
        for group in groups:
            if len(group) == 1:
                integrated.extend(group)
            else:
                # 合并相似记忆
                merged = self._merge_memories(group)
                integrated.append(merged)
        
        return integrated
    
    def _cluster_similar_memories(
        self,
        memories: List[Document]
    ) -> List[List[Document]]:
        """聚类相似记忆"""
        clusters = []
        visited = set()
        
        for i, mem1 in enumerate(memories):
            if i in visited:
                continue
            
            cluster = [mem1]
            visited.add(i)
            
            for j, mem2 in enumerate(memories[i+1:], i+1):
                if j in visited:
                    continue
                
                similarity = self._calculate_similarity(mem1, mem2)
                if similarity >= self.similarity_threshold:
                    cluster.append(mem2)
                    visited.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_similarity(self, doc1: Document, doc2: Document) -> float:
        """计算相似度"""
        # 简化：使用 Jaccard 相似度
        words1 = set(doc1.page_content.lower().split())
        words2 = set(doc2.page_content.lower().split())
        
        if not words1 or not words2:
            return 0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _merge_memories(self, memories: List[Document]) -> Document:
        """合并多条记忆"""
        import json
        
        # 使用 LLM 合并
        merge_prompt = """
请整合以下相关记忆，保留所有重要信息，去除重复内容：

{memories}

输出整合后的内容：
"""
        
        memories_text = "\n---\n".join([m.page_content for m in memories])
        merged_content = self.llm.invoke([
            HumanMessage(content=merge_prompt.format(memories=memories_text))
        ]).content
        
        # 合并元数据
        merged_metadata = {
            "merged": True,
            "source_count": len(memories),
            "importance": max(m.metadata.get("importance", 0.5) for m in memories),
            "timestamp": max(m.metadata.get("timestamp", "") for m in memories)
        }
        
        return Document(
            page_content=merged_content,
            metadata=merged_metadata
        )

# 使用示例
integrator = MemoryIntegrator(llm, similarity_threshold=0.7)

# 模拟重复记忆
memories = [
    Document(page_content="Python 是一种解释型编程语言", metadata={"key": "mem1"}),
    Document(page_content="Python 是解释型语言", metadata={"key": "mem2"}),
    Document(page_content="Java 是编译型语言", metadata={"key": "mem3"})
]

integrated = integrator.integrate_similar_memories(memories)
print(f"整合前：{len(memories)} 条")
print(f"整合后：{len(integrated)} 条")
```

#### 索引优化

```python
# 记忆索引优化策略

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
import hashlib

class OptimizedMemoryIndex:
    """优化的记忆索引"""
    
    def __init__(self, persist_directory: str = "./optimized_memory"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 分层索引
        self.category_indices: Dict[str, Chroma] = {}
        self.global_index = Chroma(
            persist_directory=f"{persist_directory}/global",
            embedding_function=self.embeddings
        )
        
        self.persist_directory = persist_directory
    
    def store(self, content: str, category: str, metadata: dict = None):
        """存储到分层索引"""
        doc_id = hashlib.md5(content.encode()).hexdigest()
        doc_metadata = {
            "category": category,
            "doc_id": doc_id,
            **(metadata or {})
        }
        
        from langchain_core.documents import Document
        doc = Document(page_content=content, metadata=doc_metadata)
        
        # 添加到全局索引
        self.global_index.add_documents([doc], ids=[doc_id])
        
        # 添加到分类索引
        if category not in self.category_indices:
            self.category_indices[category] = Chroma(
                persist_directory=f"{self.persist_directory}/categories/{category}",
                embedding_function=self.embeddings
            )
        
        self.category_indices[category].add_documents([doc], ids=[doc_id])
    
    def retrieve(
        self,
        query: str,
        category: str = None,
        k: int = 5
    ) -> List[Document]:
        """检索"""
        if category and category in self.category_indices:
            # 从分类索引检索（更快、更准）
            return self.category_indices[category].similarity_search(query, k=k)
        else:
            # 从全局索引检索
            return self.global_index.similarity_search(query, k=k)
    
    def retrieve_hierarchical(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """分层检索：先确定类别，再检索"""
        # 步骤 1：识别相关类别
        category_query = f"以下查询属于哪个类别？{query}"
        categories = list(self.category_indices.keys())
        
        # 简化：从所有类别检索
        all_results = []
        for category in categories:
            results = self.category_indices[category].similarity_search(query, k=k)
            all_results.extend(results)
        
        # 排序并返回前 k 个
        all_results.sort(key=lambda x: len(x.page_content))  # 简化排序
        return all_results[:k]
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "categories": list(self.category_indices.keys()),
            "category_count": len(self.category_indices)
        }

# 使用示例
opt_memory = OptimizedMemoryIndex()

# 存储分类记忆
opt_memory.store("Python 基础语法", "programming", {"importance": 0.8})
opt_memory.store("用户喜欢简洁回答", "preference", {"importance": 0.9})
opt_memory.store("机器学习算法", "ai", {"importance": 0.7})

# 检索
results = opt_memory.retrieve("编程知识", category="programming", k=3)
```

---

## 2. 工具使用深度实践

### 2.1 工具定义规范

#### JSON Schema 定义

```python
# 工具定义的完整规范

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import Enum

# 1. 基础工具定义
class CalculatorInput(BaseModel):
    """计算器输入"""
    expression: str = Field(
        description="数学表达式，例如：2 + 3 * 4",
        min_length=1,
        max_length=100
    )

# 2. 带枚举的工具定义
class SearchCategory(str, Enum):
    """搜索类别"""
    GENERAL = "general"
    NEWS = "news"
    ACADEMIC = "academic"
    IMAGE = "image"

class AdvancedSearchInput(BaseModel):
    """高级搜索输入"""
    query: str = Field(
        description="搜索关键词",
        min_length=1,
        max_length=200
    )
    category: SearchCategory = Field(
        default=SearchCategory.GENERAL,
        description="搜索类别"
    )
    limit: int = Field(
        default=10,
        description="返回结果数量",
        ge=1,
        le=100
    )
    language: str = Field(
        default="zh",
        description="语言代码",
        pattern="^[a-z]{2}$"
    )
    start_date: Optional[str] = Field(
        default=None,
        description="开始日期（YYYY-MM-DD）",
        pattern="^\\d{4}-\\d{2}-\\d{2}$"
    )

# 3. 复杂嵌套结构
class FilterCondition(BaseModel):
    """过滤条件"""
    field: str = Field(description="字段名")
    operator: Literal["eq", "ne", "gt", "lt", "contains"] = Field(
        description="操作符"
    )
    value: str = Field(description="值")

class DatabaseQueryInput(BaseModel):
    """数据库查询输入"""
    table: str = Field(
        description="表名",
        pattern="^[a-zA-Z_][a-zA-Z0-9_]*$"
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="查询列（None 表示所有列）"
    )
    filters: Optional[List[FilterCondition]] = Field(
        default=None,
        description="过滤条件"
    )
    order_by: Optional[str] = Field(
        default=None,
        description="排序字段"
    )
    limit: int = Field(
        default=100,
        description="限制数量",
        ge=1,
        le=1000
    )

# 4. 工具输出 Schema
class SearchResult(BaseModel):
    """搜索结果"""
    title: str = Field(description="标题")
    url: str = Field(description="URL")
    snippet: str = Field(description="摘要")
    source: str = Field(description="来源")
    published_date: Optional[str] = Field(default=None, description="发布日期")

class SearchOutput(BaseModel):
    """搜索输出"""
    query: str = Field(description="查询词")
    total_results: int = Field(description="总结果数")
    results: List[SearchResult] = Field(description="结果列表")
    execution_time: float = Field(description="执行时间（秒）")

# 5. 创建结构化输出工具
from langchain_core.tools import StructuredTool

def advanced_search(
    query: str,
    category: str = "general",
    limit: int = 10,
    language: str = "zh",
    start_date: str = None
) -> str:
    """高级搜索工具实现"""
    import requests
    import time
    
    start_time = time.time()
    
    # 模拟搜索逻辑
    results = [
        SearchResult(
            title=f"结果 {i+1}: {query}",
            url=f"https://example.com/result{i+1}",
            snippet=f"这是关于 {query} 的第 {i+1} 个搜索结果...",
            source="Example Search",
            published_date="2024-01-01"
        )
        for i in range(min(limit, 10))
    ]
    
    output = SearchOutput(
        query=query,
        total_results=len(results),
        results=results,
        execution_time=time.time() - start_time
    )
    
    import json
    return json.dumps(output.dict(), ensure_ascii=False, indent=2)

# 创建工具
search_tool = StructuredTool.from_function(
    func=advanced_search,
    name="advanced_search",
    description="高级搜索工具，支持分类、日期过滤、多语言",
    args_schema=AdvancedSearchInput,
    return_direct=False
)
```

#### 函数描述最佳实践

```python
# 工具描述的最佳实践

from langchain_core.tools import StructuredTool

# ❌ 错误示例：描述不清晰
def bad_tool(x: str) -> str:
    """处理数据"""  # 太模糊
    pass

bad_tool_def = StructuredTool.from_function(
    func=bad_tool,
    name="process",  # 名称不明确
    description="处理一些东西"  # 没有说明何时使用
)

# ✅ 正确示例：描述详细清晰
def good_tool(query: str, category: str = "all") -> str:
    """
    搜索公司内部知识库。
    
    当用户询问公司产品、政策、流程等信息时使用此工具。
    支持按类别筛选：tech（技术）、hr（人事）、finance（财务）。
    
    返回格式：JSON 包含标题、内容、相关度评分。
    注意：每次查询最多返回 10 条结果。
    """
    pass

good_tool_def = StructuredTool.from_function(
    func=good_tool,
    name="knowledge_base_search",  # 名称清晰
    description="""搜索公司内部知识库。

适用场景：
- 查询公司产品信息
- 查询公司政策和流程
- 查找技术文档

参数说明：
- query: 搜索关键词（必填）
- category: 类别筛选（可选：all/tech/hr/finance）

返回格式：JSON，包含 title、content、relevance_score
限制：最多返回 10 条结果""",
    args_schema=AdvancedSearchInput
)

# 工具描述模板
TOOL_DESCRIPTION_TEMPLATE = """
{工具名称}

功能：{一句话描述功能}

适用场景：
- {场景 1}
- {场景 2}
- {场景 3}

参数说明：
- {参数 1}: {说明}（{必填/可选}）
- {参数 2}: {说明}（{必填/可选}）

返回格式：{描述返回的数据结构}

注意事项：
- {注意 1}
- {注意 2}

示例：
输入：{示例输入}
输出：{示例输出}
"""

# 使用模板
weather_tool_description = TOOL_DESCRIPTION_TEMPLATE.format(
    工具名称="weather_query",
    一句话描述="查询指定城市的实时天气信息",
    场景 1="用户询问某地天气",
    场景 2="需要比较多个城市天气",
    场景 3="规划出行需要天气信息",
    参数 1="city",
    说明 1="城市名称",
    必填 1="必填",
    参数 2="unit",
    说明 2="温度单位",
    必填 2="可选：celsius/fahrenheit，默认 celsius",
    返回格式="JSON，包含 temperature、humidity、weather_condition",
    注意 1="城市名称请使用中文或英文",
    注意 2="数据更新延迟约 5 分钟",
    示例输入="北京",
    示例输出='{"temperature": 25, "humidity": 60, "condition": "晴"}'
)
```

#### 参数验证

```python
# 完善的参数验证

from pydantic import BaseModel, Field, validator
from typing import Optional
import re

class EmailInput(BaseModel):
    """邮件发送输入"""
    to: str = Field(description="收件人邮箱")
    subject: str = Field(description="邮件主题", max_length=100)
    body: str = Field(description="邮件正文", max_length=5000)
    cc: Optional[str] = Field(default=None, description="抄送邮箱")
    
    @validator("to", "cc")
    def validate_email(cls, v):
        """验证邮箱格式"""
        if v is None:
            return v
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(pattern, v):
            raise ValueError(f"无效的邮箱地址：{v}")
        return v
    
    @validator("subject")
    def validate_subject(cls, v):
        """验证主题"""
        if not v.strip():
            raise ValueError("邮件主题不能为空")
        return v.strip()

class DateRangeInput(BaseModel):
    """日期范围输入"""
    start_date: str = Field(description="开始日期")
    end_date: str = Field(description="结束日期")
    
    @validator("start_date", "end_date")
    def validate_date_format(cls, v):
        """验证日期格式"""
        pattern = r"^\d{4}-\d{2}-\d{2}$"
        if not re.match(pattern, v):
            raise ValueError(f"日期格式错误，应为 YYYY-MM-DD: {v}")
        return v
    
    @validator("end_date")
    def validate_date_range(cls, v, values):
        """验证日期范围"""
        if "start_date" in values:
            from datetime import datetime
            start = datetime.fromisoformat(values["start_date"])
            end = datetime.fromisoformat(v)
            
            if end < start:
                raise ValueError("结束日期不能早于开始日期")
            
            if (end - start).days > 365:
                raise ValueError("日期范围不能超过 365 天")
        
        return v

# 使用验证的工具
def send_email_tool(to: str, subject: str, body: str, cc: str = None) -> str:
    """发送邮件"""
    try:
        # Pydantic 会自动验证
        input_data = EmailInput(to=to, subject=subject, body=body, cc=cc)
        
        # 实际发送逻辑
        print(f"发送邮件到：{input_data.to}")
        print(f"主题：{input_data.subject}")
        
        return "邮件发送成功"
    except Exception as e:
        return f"邮件发送失败：{str(e)}"

email_tool = StructuredTool.from_function(
    func=send_email_tool,
    name="send_email",
    description="发送邮件给指定收件人",
    args_schema=EmailInput
)
```

#### 错误处理

```python
# 工具错误处理完整方案

from typing import Union
import traceback

class ToolExecutionError(Exception):
    """工具执行错误"""
    def __init__(self, message: str, error_type: str = "unknown", retryable: bool = False):
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable

def safe_tool_execution(func, **kwargs) -> dict:
    """安全执行工具"""
    try:
        result = func(**kwargs)
        return {
            "success": True,
            "result": result,
            "error": None
        }
    except ToolExecutionError as e:
        return {
            "success": False,
            "result": None,
            "error": {
                "type": e.error_type,
                "message": str(e),
                "retryable": e.retryable
            }
        }
    except Exception as e:
        # 记录详细错误信息
        error_detail = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        
        return {
            "success": False,
            "result": None,
            "error": error_detail
        }

# 工具错误分类
class APIRateLimitError(ToolExecutionError):
    """API 限流错误"""
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message, error_type="rate_limit", retryable=True)
        self.retry_after = retry_after

class AuthenticationError(ToolExecutionError):
    """认证错误"""
    def __init__(self, message: str):
        super().__init__(message, error_type="authentication", retryable=False)

class NotFoundError(ToolExecutionError):
    """未找到错误"""
    def __init__(self, resource: str):
        super().__init__(f"未找到资源：{resource}", error_type="not_found", retryable=False)

class ValidationError(ToolExecutionError):
    """验证错误"""
    def __init__(self, message: str):
        super().__init__(message, error_type="validation", retryable=False)

# 使用示例
def api_call_tool(endpoint: str, params: dict = None) -> str:
    """调用 API"""
    import requests
    
    try:
        response = requests.get(f"https://api.example.com/{endpoint}", params=params)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise APIRateLimitError("API 调用频率超限", retry_after)
        elif response.status_code == 401:
            raise AuthenticationError("API 密钥无效")
        elif response.status_code == 404:
            raise NotFoundError(endpoint)
        else:
            raise ToolExecutionError(f"API 错误：{response.status_code}")
    
    except requests.exceptions.Timeout:
        raise ToolExecutionError("API 请求超时", error_type="timeout", retryable=True)
    except requests.exceptions.ConnectionError:
        raise ToolExecutionError("网络连接错误", error_type="connection", retryable=True)

# 带错误恢复的工具执行
def execute_with_retry(tool_func, max_retries: int = 3, **kwargs):
    """带重试的工具执行"""
    import time
    
    last_error = None
    for attempt in range(max_retries + 1):
        result = safe_tool_execution(tool_func, **kwargs)
        
        if result["success"]:
            return result
        
        error = result["error"]
        last_error = error
        
        # 检查是否可重试
        if not error.get("retryable", False):
            return result
        
        # 检查是否是限流错误
        if error.get("type") == "rate_limit":
            retry_after = error.get("retry_after", 60)
            print(f"API 限流，等待 {retry_after} 秒后重试...")
            time.sleep(retry_after)
        else:
            # 指数退避
            wait_time = 2 ** attempt
            print(f"第 {attempt + 1} 次重试，等待 {wait_time} 秒...")
            time.sleep(wait_time)
    
    return {
        "success": False,
        "result": None,
        "error": {
            "type": "max_retries_exceeded",
            "message": f"重试 {max_retries} 次后仍失败",
            "last_error": last_error
        }
    }
```

### 2.2 工具选择策略

#### 单工具调用

```python
# 单工具调用：最简单的方式

from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor

# 工具列表
tools = [
    Tool(name="search", func=search_func, description="搜索信息"),
    Tool(name="calculator", func=calc_func, description="数学计算"),
    Tool(name="weather", func=weather_func, description="查询天气")
]

# ReAct Agent 会自动选择单个工具
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# 执行：每次只调用一个工具
result = executor.invoke({"input": "北京今天天气如何？"})
# Agent 会自动选择 weather 工具
```

#### 多工具并行

```python
# 多工具并行执行

import asyncio
from typing import List, Dict
from langchain_core.tools import BaseTool

class ParallelToolExecutor:
    """并行工具执行器"""
    
    def __init__(self, tools: List[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}
    
    async def execute_parallel(
        self,
        tool_calls: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """并行执行多个工具调用"""
        tasks = []
        
        for call in tool_calls:
            tool_name = call["name"]
            tool_input = call["input"]
            
            if tool_name not in self.tools:
                tasks.append(asyncio.sleep(0))  # 占位
                continue
            
            tool = self.tools[tool_name]
            task = asyncio.create_task(
                self._execute_single(tool, tool_input)
            )
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 整理结果
        output = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                output.append({
                    "tool": tool_calls[i]["name"],
                    "success": False,
                    "error": str(result)
                })
            else:
                output.append({
                    "tool": tool_calls[i]["name"],
                    "success": True,
                    "result": result
                })
        
        return output
    
    async def _execute_single(self, tool, tool_input):
        """执行单个工具"""
        # 如果是异步工具
        if asyncio.iscoroutinefunction(tool.run):
            return await tool.run(tool_input=tool_input)
        else:
            # 同步工具，在线程池中执行
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: tool.run(tool_input=tool_input)
            )

# 使用示例
async def parallel_execution_example():
    """并行执行示例"""
    executor = ParallelToolExecutor(tools)
    
    # 定义多个工具调用
    tool_calls = [
        {"name": "weather", "input": "北京"},
        {"name": "weather", "input": "上海"},
        {"name": "weather", "input": "广州"}
    ]
    
    # 并行执行
    results = await executor.execute_parallel(tool_calls)
    
    for result in results:
        print(f"工具：{result['tool']}")
        print(f"成功：{result['success']}")
        if result["success"]:
            print(f"结果：{result['result']}")
        else:
            print(f"错误：{result['error']}")
        print("---")

# 在 Agent 中使用并行工具
def create_parallel_tool(llm, tools):
    """创建支持并行工具的 Agent"""
    
    # 自定义工具，内部实现并行逻辑
    def parallel_weather(cities: str) -> str:
        """查询多个城市天气"""
        import json
        
        city_list = [c.strip() for c in cities.split(",")]
        
        # 并行查询
        async def query():
            executor = ParallelToolExecutor(tools)
            tool_calls = [
                {"name": "weather", "input": city}
                for city in city_list
            ]
            return await executor.execute_parallel(tool_calls)
        
        results = asyncio.run(query())
        
        # 格式化输出
        output = []
        for result in results:
            if result["success"]:
                output.append(f"{result['tool']}: {result['result']}")
            else:
                output.append(f"{result['tool']}: 失败 - {result['error']}")
        
        return "\n".join(output)
    
    parallel_tool = Tool(
        name="parallel_weather",
        func=parallel_weather,
        description="查询多个城市的天气，城市用逗号分隔"
    )
    
    return parallel_tool
```

#### 工具链编排

```python
# 工具链编排：按顺序执行多个工具

from typing import List, Callable
from functools import reduce

class ToolChain:
    """工具链"""
    
    def __init__(self):
        self.steps: List[Callable] = []
    
    def add_step(self, tool: Callable, name: str = ""):
        """添加步骤"""
        self.steps.append({
            "tool": tool,
            "name": name or tool.__name__
        })
        return self
    
    def execute(self, initial_input) -> dict:
        """执行工具链"""
        result = initial_input
        execution_log = []
        
        for i, step in enumerate(self.steps):
            try:
                print(f"执行步骤 {i+1}: {step['name']}")
                result = step["tool"](result)
                
                execution_log.append({
                    "step": i + 1,
                    "name": step["name"],
                    "success": True,
                    "output": str(result)[:200]
                })
            except Exception as e:
                execution_log.append({
                    "step": i + 1,
                    "name": step["name"],
                    "success": False,
                    "error": str(e)
                })
                raise e
        
        return {
            "final_result": result,
            "execution_log": execution_log
        }

# 示例：数据分析工具链
def fetch_data(query: str) -> dict:
    """获取数据"""
    print(f"获取数据：{query}")
    return {"data": [1, 2, 3, 4, 5], "query": query}

def clean_data(data_dict: dict) -> dict:
    """清洗数据"""
    print("清洗数据...")
    data_dict["data"] = [x * 2 for x in data_dict["data"]]
    return data_dict

def analyze_data(data_dict: dict) -> dict:
    """分析数据"""
    print("分析数据...")
    data_dict["statistics"] = {
        "mean": sum(data_dict["data"]) / len(data_dict["data"]),
        "max": max(data_dict["data"]),
        "min": min(data_dict["data"])
    }
    return data_dict

def visualize_result(data_dict: dict) -> str:
    """可视化结果"""
    print("生成可视化...")
    return f"分析报告：{data_dict['statistics']}"

# 构建工具链
analysis_chain = ToolChain()
analysis_chain.add_step(fetch_data, "获取数据")
analysis_chain.add_step(clean_data, "清洗数据")
analysis_chain.add_step(analyze_data, "分析数据")
analysis_chain.add_step(visualize_result, "生成报告")

# 执行
result = analysis_chain.execute("销售数据 2024")
print(f"最终结果：{result['final_result']}")
```

#### 动态选择

```python
# 动态工具选择：LLM 决定使用哪些工具

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(model="gpt-5.2", temperature=0)

# 工具选择器
tool_selector_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个工具选择专家。根据用户请求，选择最合适的工具或工具组合。

可用工具：
{tools_description}

输出 JSON 格式：
```json
{{
    "selected_tools": ["tool1", "tool2"],
    "execution_mode": "sequential|parallel",
    "reason": "选择原因"
}}
```"""),
    ("human", "{user_input}")
])

tool_selector = tool_selector_prompt | llm

class DynamicToolSelector:
    """动态工具选择器"""
    
    def __init__(self, llm, tools: List[Tool]):
        self.llm = llm
        self.tools = tools
        self.selector = tool_selector_prompt | llm
    
    def select_tools(self, user_input: str) -> dict:
        """选择工具"""
        # 构建工具描述
        tools_desc = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])
        
        # LLM 选择
        response = self.selector.invoke({
            "tools_description": tools_desc,
            "user_input": user_input
        })
        
        # 解析结果
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            selection = json.loads(json_str.strip())
            return selection
        except:
            return {
                "selected_tools": [],
                "execution_mode": "sequential",
                "reason": "选择失败"
            }
    
    def execute_with_selection(self, user_input: str) -> str:
        """选择并执行工具"""
        # 选择工具
        selection = self.select_tools(user_input)
        selected_tools = selection.get("selected_tools", [])
        
        print(f"选择的工具：{selected_tools}")
        print(f"执行模式：{selection['execution_mode']}")
        print(f"原因：{selection['reason']}")
        
        # 执行（简化）
        results = []
        for tool_name in selected_tools:
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool:
                result = tool.run(user_input)
                results.append(result)
        
        return "\n".join(results)

# 使用示例
tools = [
    Tool(name="search", func=lambda x: f"搜索：{x}", description="搜索网络信息"),
    Tool(name="calculator", func=lambda x: f"计算：{x}", description="数学计算"),
    Tool(name="database", func=lambda x: f"查询数据库：{x}", description="查询数据库")
]

selector = DynamicToolSelector(llm, tools)
result = selector.execute_with_selection("查询 2024 年销售数据并计算增长率")
```

### 2.3 工具执行优化

#### 超时控制

```python
# 工具超时控制

import signal
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError

class TimeoutException(Exception):
    """超时异常"""
    pass

def timeout(seconds: int):
    """超时装饰器（仅适用于 Unix）"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutException(f"函数执行超时：{seconds}秒")
            
            # 设置信号
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # 取消闹钟
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        return wrapper
    return decorator

# 跨平台超时控制（使用线程池）
def execute_with_timeout(func, timeout_seconds: int, *args, **kwargs):
    """带超时控制的执行"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return {
                "success": True,
                "result": result,
                "timeout": False
            }
        except TimeoutError:
            return {
                "success": False,
                "result": None,
                "timeout": True,
                "error": f"执行超时（{timeout_seconds}秒）"
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "timeout": False,
                "error": str(e)
            }

# 使用示例
@timeout(5)  # 5 秒超时
def slow_api_call():
    """慢速 API 调用"""
    import time
    time.sleep(10)  # 模拟慢操作
    return "完成"

# 方式 1：装饰器
try:
    result = slow_api_call()
except TimeoutException as e:
    print(f"超时：{e}")

# 方式 2：函数调用
result = execute_with_timeout(slow_api_call, 5)
if result["timeout"]:
    print(f"超时：{result['error']}")
elif result["success"]:
    print(f"成功：{result['result']}")
else:
    print(f"错误：{result['error']}")

# 在 LangChain 工具中使用
from langchain_core.tools import Tool

def create_timeout_tool(func, timeout_seconds: int, **kwargs):
    """创建带超时的工具"""
    def wrapped_func(*args, **kwargs):
        result = execute_with_timeout(func, timeout_seconds, *args, **kwargs)
        
        if result["timeout"]:
            return f"工具执行超时：{result['error']}"
        elif result["success"]:
            return result["result"]
        else:
            return f"工具执行失败：{result['error']}"
    
    return Tool(
        func=wrapped_func,
        **kwargs
    )
```

#### 重试机制

```python
# 完善的重试机制

import time
import random
from typing import Callable, Optional

class RetryConfig:
    """重试配置"""
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (Exception,)
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions

def retry_with_config(func: Callable, config: RetryConfig, *args, **kwargs):
    """带配置的重试"""
    last_exception = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e
            
            if attempt == config.max_retries:
                raise last_exception
            
            # 计算延迟
            delay = min(
                config.base_delay * (config.exponential_base ** attempt),
                config.max_delay
            )
            
            # 添加抖动
            if config.jitter:
                delay = delay * (0.5 + random.random() * 0.5)
            
            print(f"第 {attempt + 1} 次重试，等待 {delay:.2f}秒...")
            time.sleep(delay)
    
    raise last_exception

# 重试装饰器
def retry(config: RetryConfig):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return retry_with_config(func, config, *args, **kwargs)
        return wrapper
    return decorator

# 使用示例
@retry(RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=(ConnectionError, TimeoutError)
))
def unreliable_api():
    """不可靠的 API"""
    import random
    if random.random() < 0.7:
        raise ConnectionError("连接失败")
    return "成功"

# 执行
try:
    result = unreliable_api()
    print(f"结果：{result}")
except Exception as e:
    print(f"最终失败：{e}")
```

#### 结果缓存

```python
# 工具结果缓存

import hashlib
import json
from typing import Any, Optional
from functools import wraps
import time

class ToolCache:
    """工具缓存管理器"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl  # Time To Live（秒）
        self.cache: dict = {}
        self.access_times: dict = {}
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        key_data = {
            "func": func_name,
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if key not in self.cache:
            return None
        
        # 检查是否过期
        access_time = self.access_times[key]
        if time.time() - access_time > self.ttl:
            del self.cache[key]
            del self.access_times[key]
            return None
        
        return self.cache[key]
    
    def set(self, key: str, value: Any):
        """设置缓存"""
        # 检查大小限制
        if len(self.cache) >= self.max_size:
            self._evict()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def _evict(self):
        """淘汰最旧的缓存"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times, key=self.access_times.get)
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl
        }

# 全局缓存
tool_cache = ToolCache(max_size=1000, ttl=3600)

def cached_tool(ttl: int = None):
    """工具缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成键
            key = tool_cache._generate_key(func.__name__, args, kwargs)
            
            # 尝试从缓存获取
            cached_result = tool_cache.get(key)
            if cached_result is not None:
                print(f"缓存命中：{func.__name__}")
                return cached_result
            
            # 执行并缓存
            print(f"缓存未命中，执行：{func.__name__}")
            result = func(*args, **kwargs)
            
            # 使用自定义 TTL 或默认 TTL
            cache_ttl = ttl if ttl is not None else tool_cache.ttl
            tool_cache.set(key, result)
            
            return result
        return wrapper
    return decorator

# 使用示例
@cached_tool(ttl=1800)  # 缓存 30 分钟
def expensive_api_call(query: str):
    """耗时的 API 调用"""
    import time
    time.sleep(2)  # 模拟耗时操作
    return f"结果：{query}"

# 第一次调用（缓存未命中）
result1 = expensive_api_call("测试查询")
print(f"结果 1: {result1}")

# 第二次调用（缓存命中）
result2 = expensive_api_call("测试查询")
print(f"结果 2: {result2}")

# 查看缓存统计
print(f"缓存统计：{tool_cache.get_stats()}")
```

#### 降级策略

```python
# 工具降级策略

from typing import Callable, Optional

class FallbackChain:
    """降级链"""
    
    def __init__(self):
        self.tools: list = []
    
    def add_tool(self, tool: Callable, priority: int = 0):
        """添加工具"""
        self.tools.append({
            "tool": tool,
            "priority": priority
        })
        # 按优先级排序（高优先级在前）
        self.tools.sort(key=lambda x: x["priority"], reverse=True)
        return self
    
    def execute(self, *args, **kwargs) -> dict:
        """执行降级链"""
        errors = []
        
        for tool_info in self.tools:
            tool = tool_info["tool"]
            try:
                result = tool(*args, **kwargs)
                return {
                    "success": True,
                    "result": result,
                    "tool_used": tool.__name__,
                    "fallback_count": len(errors)
                }
            except Exception as e:
                errors.append({
                    "tool": tool.__name__,
                    "error": str(e)
                })
                print(f"工具 {tool.__name__} 失败，尝试降级...")
        
        # 所有工具都失败
        return {
            "success": False,
            "result": None,
            "errors": errors
        }

# 使用示例
def primary_search(query: str):
    """主搜索引擎"""
    import random
    if random.random() < 0.3:
        raise ConnectionError("主搜索引擎不可用")
    return f"主搜索引擎结果：{query}"

def secondary_search(query: str):
    """备用搜索引擎"""
    import random
    if random.random() < 0.5:
        raise ConnectionError("备用搜索引擎也不可用")
    return f"备用搜索引擎结果：{query}"

def local_cache_search(query: str):
    """本地缓存搜索"""
    return f"本地缓存结果：{query}"

# 构建降级链
search_chain = FallbackChain()
search_chain.add_tool(primary_search, priority=3)
search_chain.add_tool(secondary_search, priority=2)
search_chain.add_tool(local_cache_search, priority=1)

# 执行
result = search_chain.execute("查询关键词")
if result["success"]:
    print(f"成功，使用工具：{result['tool_used']}")
    print(f"降级次数：{result['fallback_count']}")
    print(f"结果：{result['result']}")
else:
    print(f"所有工具都失败：{result['errors']}")
```

---

## 3. 规划与推理

### 3.1 任务规划

#### 目标分解

```python
# 智能任务分解

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
import json

llm = ChatOpenAI(model="gpt-5.2", temperature=0)

class SubTask(BaseModel):
    """子任务"""
    id: int = Field(description="任务 ID")
    description: str = Field(description="任务描述")
    required_tools: List[str] = Field(description="需要的工具")
    estimated_complexity: int = Field(description="预估复杂度 1-5", ge=1, le=5)
    dependencies: List[int] = Field(description="依赖的任务 ID", default_factory=list)
    success_criteria: str = Field(description="成功标准")

class TaskPlan(BaseModel):
    """任务计划"""
    goal: str = Field(description="目标")
    subtasks: List[SubTask] = Field(description="子任务列表")
    execution_order: List[int] = Field(description="执行顺序")
    estimated_total_time: str = Field(description="预估总时间")
    risks: List[str] = Field(description="潜在风险", default_factory=list)

# 任务分解 Prompt
task_decomposition_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的任务规划专家。请将复杂目标分解为可执行的子任务。

分解原则：
1. **MECE 原则**：相互独立，完全穷尽
2. **原子性**：每个子任务应该是不可再分的原子操作
3. **可验证性**：每个子任务应该有明确的完成标准
4. **可并行性**：识别可以并行执行的任务
5. **依赖关系**：明确任务间的依赖

输出格式为 JSON，包含：
- goal: 目标描述
- subtasks: 子任务列表
- execution_order: 执行顺序
- estimated_total_time: 预估总时间
- risks: 潜在风险"""),
    ("human", "目标：{goal}\n\n可用工具：{tools}")
])

# 使用结构化输出
decomposer = task_decomposition_prompt | llm.with_structured_output(TaskPlan)

# 使用示例
def decompose_task(goal: str, available_tools: List[str]) -> TaskPlan:
    """分解任务"""
    tools_text = "\n".join([f"- {tool}" for tool in available_tools])
    
    plan = decomposer.invoke({
        "goal": goal,
        "tools": tools_text
    })
    
    return plan

# 执行分解
goal = "分析公司 2024 年销售数据，生成可视化报告并提出改进建议"
tools = ["database_query", "data_analysis", "chart_generation", "report_writing"]

plan = decompose_task(goal, tools)

print(f"目标：{plan.goal}")
print(f"\n子任务：")
for task in plan.subtasks:
    print(f"  {task.id}. {task.description}")
    print(f"     工具：{', '.join(task.required_tools)}")
    print(f"     依赖：{task.dependencies}")
    print(f"     复杂度：{task.estimated_complexity}")
    print(f"     成功标准：{task.success_criteria}")

print(f"\n执行顺序：{plan.execution_order}")
print(f"预估时间：{plan.estimated_total_time}")
print(f"风险：{plan.risks}")
```

#### 步骤生成

```python
# 动态步骤生成

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class StepGenerator:
    """步骤生成器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_steps(
        self,
        task: str,
        completed_steps: List[dict] = None,
        context: dict = None
    ) -> dict:
        """生成下一步"""
        prompt = f"""
当前任务：{task}

已完成步骤：
{self._format_steps(completed_steps)}

上下文信息：
{json.dumps(context or {}, ensure_ascii=False, indent=2)}

请决定：
1. 下一步应该做什么？
2. 需要使用什么工具？
3. 工具的参数是什么？
4. 预期的结果是什么？

输出 JSON：
```json
{{
    "next_step": "步骤描述",
    "tool": "工具名称",
    "tool_input": {{}},
    "expected_output": "预期结果",
    "reason": "选择此步骤的原因",
    "is_final_step": true/false
}}
```
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except:
            return {
                "next_step": "完成任务",
                "tool": None,
                "is_final_step": True
            }
    
    def _format_steps(self, steps: List[dict]) -> str:
        """格式化已完成的步骤"""
        if not steps:
            return "无"
        
        return "\n".join([
            f"{i+1}. {step.get('description', '未知')} - {'成功' if step.get('success') else '失败'}"
            for i, step in enumerate(steps)
        ])

# 使用示例
generator = StepGenerator(llm)

task = "查询北京天气并生成报告"
completed_steps = []
context = {"user_location": "北京"}

# 动态生成步骤
for i in range(10):
    step = generator.generate_steps(task, completed_steps, context)
    
    print(f"\n步骤 {i+1}: {step['next_step']}")
    print(f"工具：{step.get('tool')}")
    print(f"原因：{step.get('reason')}")
    
    # 模拟执行
    completed_steps.append({
        "description": step["next_step"],
        "success": True
    })
    
    if step.get("is_final_step"):
        print("\n任务完成！")
        break
```

#### 资源评估

```python
# 资源评估与成本估算

class ResourceEstimator:
    """资源评估器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def estimate_resources(self, task_plan: TaskPlan) -> dict:
        """评估资源需求"""
        prompt = f"""
请评估以下任务计划所需的资源：

目标：{task_plan.goal}
子任务数量：{len(task_plan.subtasks)}
总预估时间：{task_plan.estimated_total_time}

子任务详情：
{json.dumps([t.dict() for t in task_plan.subtasks], ensure_ascii=False, indent=2)}

请评估：
1. **LLM 调用次数**：预估需要调用多少次 LLM
2. **Token 消耗**：预估总 token 数量
3. **工具调用次数**：各工具调用次数
4. **执行时间**：实际执行时间（考虑并行）
5. **成本估算**：按 $0.01/1K tokens 计算

输出 JSON：
```json
{{
    "llm_calls": 10,
    "estimated_tokens": 50000,
    "tool_calls": {{
        "tool1": 5,
        "tool2": 3
    }},
    "estimated_execution_minutes": 15,
    "estimated_cost_usd": 0.50,
    "optimization_suggestions": ["优化建议"]
}}
```
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except:
            return {"error": "评估失败"}

# 使用示例
estimator = ResourceEstimator(llm)
resources = estimator.estimate_resources(plan)

print(f"LLM 调用次数：{resources['llm_calls']}")
print(f"预估 Token 消耗：{resources['estimated_tokens']}")
print(f"预估执行时间：{resources['estimated_execution_minutes']} 分钟")
print(f"预估成本：${resources['estimated_cost_usd']}")
print(f"优化建议：{resources['optimization_suggestions']}")
```

#### 风险识别

```python
# 风险识别与应对策略

class RiskAnalyzer:
    """风险分析器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_risks(self, task_plan: TaskPlan) -> List[dict]:
        """分析风险"""
        prompt = f"""
请分析以下任务计划的潜在风险：

{json.dumps(task_plan.dict(), ensure_ascii=False, indent=2)}

对每个风险，请评估：
1. 风险描述
2. 发生概率（低/中/高）
3. 影响程度（低/中/高）
4. 应对策略
5. 检测信号（如何识别风险已发生）

输出 JSON 数组：
```json
[
    {{
        "risk": "风险描述",
        "probability": "低/中/高",
        "impact": "低/中/高",
        "mitigation": "应对策略",
        "warning_signs": ["检测信号"]
    }}
]
```
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except:
            return []

# 使用示例
analyzer = RiskAnalyzer(llm)
risks = analyzer.analyze_risks(plan)

print("风险分析：")
for risk in risks:
    print(f"\n风险：{risk['risk']}")
    print(f"概率：{risk['probability']}")
    print(f"影响：{risk['impact']}")
    print(f"应对：{risk['mitigation']}")
    print(f"预警：{', '.join(risk['warning_signs'])}")
```

### 3.2 自我反思

#### 错误识别

```python
# 错误识别与诊断

from langchain_core.prompts import ChatPromptTemplate

error_diagnosis_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个错误诊断专家。请分析以下执行过程中的错误。

任务：{task}
执行步骤：{steps}
错误信息：{error}

请从以下维度分析：
1. **错误类型**：语法错误、逻辑错误、资源错误、超时等
2. **根本原因**：导致错误的根本原因
3. **影响范围**：错误影响了哪些部分
4. **修复建议**：如何修复错误
5. **预防措施**：如何避免类似错误

输出 JSON：
```json
{{
    "error_type": "错误类型",
    "root_cause": "根本原因",
    "impact": "影响范围",
    "fix_suggestion": "修复建议",
    "prevention": "预防措施",
    "severity": "low/medium/high/critical",
    "can_retry": true/false
}}
```"""),
    ("human", "请分析错误")
])

error_diagnoser = error_diagnosis_prompt | llm

class ErrorDiagnoser:
    """错误诊断器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.diagnoser = error_diagnosis_prompt | llm
    
    def diagnose(
        self,
        task: str,
        steps: List[dict],
        error: str
    ) -> dict:
        """诊断错误"""
        steps_text = json.dumps(steps, ensure_ascii=False, indent=2)
        
        response = self.diagnoser.invoke({
            "task": task,
            "steps": steps_text,
            "error": error
        })
        
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except:
            return {
                "error_type": "unknown",
                "root_cause": "诊断失败",
                "can_retry": False
            }
    
    def auto_fix(self, diagnosis: dict, original_step: dict) -> dict:
        """自动生成修复方案"""
        fix_prompt = f"""
基于以下错误诊断，生成修复后的步骤：

原始步骤：{json.dumps(original_step, ensure_ascii=False)}
诊断结果：{json.dumps(diagnosis, ensure_ascii=False)}

请输出修复后的步骤（JSON）：
"""
        
        response = self.llm.invoke([HumanMessage(content=fix_prompt)])
        
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except:
            return original_step

# 使用示例
diagnoser = ErrorDiagnoser(llm)

task = "查询数据库并生成报告"
steps = [
    {"action": "查询数据库", "status": "success"},
    {"action": "分析数据", "status": "failed"}
]
error = "数据库连接超时"

diagnosis = diagnoser.diagnose(task, steps, error)
print(f"错误类型：{diagnosis['error_type']}")
print(f"根本原因：{diagnosis['root_cause']}")
print(f"严重程度：{diagnosis['severity']}")
print(f"是否可重试：{diagnosis['can_retry']}")
print(f"修复建议：{diagnosis['fix_suggestion']}")
```

#### 策略调整

```python
# 策略调整与优化

class StrategyAdjuster:
    """策略调整器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.execution_history: List[dict] = []
    
    def record_execution(self, task: str, plan: dict, result: dict):
        """记录执行"""
        self.execution_history.append({
            "task": task,
            "plan": plan,
            "result": result,
            "timestamp": time.time()
        })
    
    def suggest_improvements(self, task: str) -> dict:
        """基于历史执行建议改进"""
        # 查找类似任务
        similar_tasks = [
            exec for exec in self.execution_history
            if self._similarity(exec["task"], task) > 0.7
        ]
        
        if not similar_tasks:
            return {"suggestions": ["无历史数据可供参考"]}
        
        # 分析失败案例
        failures = [
            exec for exec in similar_tasks
            if not exec["result"].get("success", True)
        ]
        
        if not failures:
            return {"suggestions": ["历史执行都很成功，保持当前策略"]}
        
        # LLM 分析改进建议
        analysis_prompt = f"""
分析以下失败案例，提出改进策略：

失败案例：
{json.dumps(failures[:3], ensure_ascii=False, indent=2)}

请提出：
1. 策略调整建议
2. 工具选择优化
3. 错误处理改进
4. 资源分配优化

输出 JSON：
"""
        
        response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
        
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except:
            return {"suggestions": ["分析失败"]}
    
    def _similarity(self, task1: str, task2: str) -> float:
        """计算任务相似度（简化版）"""
        words1 = set(task1.lower().split())
        words2 = set(task2.lower().split())
        
        if not words1 or not words2:
            return 0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)

# 使用示例
adjuster = StrategyAdjuster(llm)

# 记录执行历史
adjuster.record_execution(
    "查询数据",
    {"tools": ["database"]},
    {"success": False, "error": "超时"}
)

# 获取改进建议
suggestions = adjuster.suggest_improvements("查询销售数据")
print(f"改进建议：{suggestions}")
```

#### 经验积累

```python
# 经验积累与知识库

import json
from pathlib import Path

class ExperienceBase:
    """经验知识库"""
    
    def __init__(self, persist_file: str = "./experiences.json"):
        self.persist_file = persist_file
        self.experiences: List[dict] = []
        self._load()
    
    def add_experience(
        self,
        task: str,
        strategy: dict,
        outcome: dict,
        lessons: List[str]
    ):
        """添加经验"""
        experience = {
            "task": task,
            "strategy": strategy,
            "outcome": outcome,
            "lessons": lessons,
            "timestamp": time.time(),
            "success_count": 1 if outcome.get("success") else 0,
            "failure_count": 0 if outcome.get("success") else 1
        }
        
        self.experiences.append(experience)
        self._save()
    
    def query_experiences(self, task: str, limit: int = 5) -> List[dict]:
        """查询相关经验"""
        # 简化：关键词匹配
        keywords = set(task.lower().split())
        
        scored = []
        for exp in self.experiences:
            exp_keywords = set(exp["task"].lower().split())
            similarity = len(keywords & exp_keywords) / len(keywords | exp_keywords)
            scored.append((similarity, exp))
        
        # 按相似度排序
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [exp for _, exp in scored[:limit]]
    
    def get_best_practices(self) -> List[dict]:
        """获取最佳实践"""
        # 筛选成功率高的经验
        high_success = [
            exp for exp in self.experiences
            if exp["success_count"] > 0 and exp["failure_count"] == 0
        ]
        
        return high_success
    
    def _save(self):
        """保存"""
        with open(self.persist_file, "w", encoding="utf-8") as f:
            json.dump(self.experiences, f, ensure_ascii=False, indent=2)
    
    def _load(self):
        """加载"""
        if Path(self.persist_file).exists():
            with open(self.persist_file, "r", encoding="utf-8") as f:
                self.experiences = json.load(f)

# 使用示例
experience_db = ExperienceBase()

# 添加经验
experience_db.add_experience(
    task="数据库查询优化",
    strategy={"use_cache": True, "batch_size": 100},
    outcome={"success": True, "execution_time": 5.2},
    lessons=["使用缓存可以显著提升性能", "批量大小 100 是最佳平衡点"]
)

# 查询经验
experiences = experience_db.query_experiences("数据库查询")
for exp in experiences:
    print(f"任务：{exp['task']}")
    print(f"经验：{', '.join(exp['lessons'])}")
    print("---")
```

### 3.3 思维链应用

#### Zero-shot CoT

```python
# Zero-shot Chain-of-Thought

from langchain_core.prompts import ChatPromptTemplate

zero_shot_cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个逻辑推理专家。请一步步思考，展示你的推理过程。"),
    ("human", "{question}\n\n请一步步思考：")
])

zero_shot_cot = zero_shot_cot_prompt | llm

# 使用示例
question = "如果 3 个人 3 天可以完成 3 个项目，那么 9 个人 9 天可以完成多少个项目？"

response = zero_shot_cot.invoke({"question": question})
print(response.content)
# 输出会包含详细的推理步骤
```

#### Few-shot CoT

```python
# Few-shot Chain-of-Thought

few_shot_cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个逻辑推理专家。请参考示例，一步步推理。"),
    
    # 示例 1
    ("human", "问题：一个农场有 100 只鸡，每天增加 5 只，30 天后有多少只？"),
    ("ai", """让我一步步思考：

1. 初始数量：100 只
2. 每天增加：5 只
3. 天数：30 天
4. 总增加：5 × 30 = 150 只
5. 最终数量：100 + 150 = 250 只

答案：250 只"""),
    
    # 示例 2
    ("human", "问题：列车以 60km/h 速度行驶，2.5 小时行驶多远？"),
    ("ai", """让我一步步思考：

1. 速度：60 km/h
2. 时间：2.5 小时
3. 距离 = 速度 × 时间
4. 距离 = 60 × 2.5 = 150 km

答案：150 公里"""),
    
    # 用户问题
    ("human", "问题：{question}\n\n请一步步思考：")
])

few_shot_cot = few_shot_cot_prompt | llm

# 使用示例
question = "如果 5 台机器 5 分钟可以生产 5 个零件，100 台机器生产 100 个零件需要多长时间？"
response = few_shot_cot.invoke({"question": question})
print(response.content)
```

#### Self-Consistency

```python
# Self-Consistency：多次采样取一致

from typing import List
from collections import Counter

class SelfConsistencyCoT:
    """Self-Consistency CoT"""
    
    def __init__(self, llm, num_samples: int = 5):
        self.llm = llm
        self.num_samples = num_samples
        self.cot_chain = few_shot_cot_prompt | llm
    
    def reason(self, question: str) -> dict:
        """推理"""
        # 多次采样
        responses = []
        for i in range(self.num_samples):
            response = self.cot_chain.invoke({"question": question})
            responses.append(response.content)
        
        # 提取答案（简化：假设答案在最后）
        answers = []
        for resp in responses:
            lines = resp.strip().split("\n")
            for line in reversed(lines):
                if "答案" in line or "answer" in line.lower():
                    answers.append(line)
                    break
        
        # 投票
        if answers:
            counter = Counter(answers)
            most_common = counter.most_common(1)[0]
            
            return {
                "question": question,
                "all_responses": responses,
                "all_answers": answers,
                "final_answer": most_common[0],
                "confidence": most_common[1] / self.num_samples,
                "vote_distribution": dict(counter)
            }
        else:
            return {
                "question": question,
                "all_responses": responses,
                "final_answer": "无法确定答案",
                "confidence": 0
            }

# 使用示例
sc_cot = SelfConsistencyCoT(llm, num_samples=5)

question = "一个水箱有两个水管，A 管单独注水需要 6 小时，B 管单独注水需要 4 小时，两管同时注水需要多长时间？"
result = sc_cot.reason(question)

print(f"问题：{result['question']}")
print(f"最终答案：{result['final_answer']}")
print(f"置信度：{result['confidence']:.2%}")
print(f"投票分布：{result['vote_distribution']}")
```

#### Tree of Thoughts

```python
# Tree of Thoughts（思维树）

from typing import List, Optional
import heapq

class ThoughtNode:
    """思维节点"""
    def __init__(self, thought: str, score: float = 0, depth: int = 0):
        self.thought = thought
        self.score = score
        self.depth = depth
        self.children: List["ThoughtNode"] = []
        self.parent: Optional["ThoughtNode"] = None
    
    def add_child(self, child: "ThoughtNode"):
        child.parent = self
        self.children.append(child)
    
    def __lt__(self, other):
        return self.score < other.score

class TreeOfThoughts:
    """思维树"""
    
    def __init__(self, llm, max_depth: int = 3, branch_factor: int = 3):
        self.llm = llm
        self.max_depth = max_depth
        self.branch_factor = branch_factor
        self.root = None
    
    def solve(self, problem: str) -> dict:
        """解决问题"""
        # 创建根节点
        self.root = ThoughtNode(f"问题：{problem}", score=0, depth=0)
        
        # BFS 搜索
        best_solution = self._search()
        
        return {
            "problem": problem,
            "solution": best_solution.thought if best_solution else "未找到解决方案",
            "solution_score": best_solution.score if best_solution else 0
        }
    
    def _search(self) -> Optional[ThoughtNode]:
        """搜索最佳解决方案"""
        # 优先队列（按分数排序）
        queue = [-self.root]  # 负数实现最大堆
        best_node = None
        best_score = -float("inf")
        
        while queue:
            # 取出最高分节点
            current = -heapq.heappop(queue)
            
            # 更新最佳节点
            if current.score > best_score:
                best_score = current.score
                best_node = current
            
            # 如果达到最大深度，跳过
            if current.depth >= self.max_depth:
                continue
            
            # 生成子节点
            children = self._generate_thoughts(current)
            
            for child in children:
                current.add_child(child)
                heapq.heappush(queue, -child)
        
        return best_node
    
    def _generate_thoughts(self, node: ThoughtNode) -> List[ThoughtNode]:
        """生成下一步思维"""
        prompt = f"""
当前思维：{node.thought}

请生成 {self.branch_factor} 个可能的下一步思维，并为每个思维评分（0-1）。

输出 JSON：
```json
[
    {{
        "thought": "思维内容",
        "score": 0.8
    }}
]
```
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            json_str = response.content.split("```json")[1].split("```")[0]
            thoughts = json.loads(json_str.strip())
            
            return [
                ThoughtNode(t["thought"], score=t["score"], depth=node.depth + 1)
                for t in thoughts
            ]
        except:
            return []

# 使用示例
tot = TreeOfThoughts(llm, max_depth=3, branch_factor=3)

problem = "如何优化一个电商网站的加载速度？"
result = tot.solve(problem)

print(f"问题：{result['problem']}")
print(f"最佳解决方案：{result['solution']}")
print(f"评分：{result['solution_score']}")
```

---

