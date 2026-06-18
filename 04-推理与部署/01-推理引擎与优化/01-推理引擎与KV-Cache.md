# 推理引擎与KV Cache优化

> 📅 **更新时间**：2025-06-12  
> 🎯 **难度等级**：Level 4-5 (高级到专家级)  
> ⏱️ **预计阅读**：90-120 分钟

---

## 1. 前言

大语言模型(LLM)的推理优化是2024-2025年AI基础设施领域最活跃的研究方向之一。随着模型规模从7B增长到405B甚至更大，推理过程中的内存瓶颈和计算效率问题变得日益突出。KV Cache技术及其优化方案（如PagedAttention、RadixAttention等）成为了突破这些瓶颈的关键。

本笔记将深入探讨：
- 推理与训练的本质差异
- KV Cache的核心原理与内存优化
- 主流推理引擎架构对比（vLLM、TensorRT-LLM、llama.cpp、SGLang）
- 生产环境部署的最佳实践
- 2025年最新优化技术

---

## 2. 推理基础概念

### 1.1 推理 vs 训练

理解推理与训练的差异是优化的前提。两者在计算模式、资源需求和性能目标上存在根本性差异。

#### 1.1.1 计算特征差异

**训练阶段（Training）**：
```
计算模式：批量前向传播 + 反向传播
数据流：  大量训练数据 → 模型 → 损失计算 → 梯度更新
GPU利用： 计算密集型 (Compute-bound)
内存占用： 模型权重 + 梯度 + 优化器状态 + 激活值
          = 4 × 模型参数量 (FP16训练)
时间尺度： 小时/天级别
```

**推理阶段（Inference）**：
```
计算模式：自回归生成 (Autoregressive Generation)
数据流：  提示词 → 模型 → Token生成 → KV Cache更新 → 下一Token
GPU利用： 内存密集型 (Memory-bound)
内存占用： 模型权重 + KV Cache + 激活值
时间尺度： 毫秒/秒级别
```

#### 1.1.2 内存瓶颈分析

**示例：7B模型推理内存计算**

```python
# 模型权重内存 (FP16)
model_size = 7e9  # 7B参数
weight_memory = model_size * 2  # FP16: 2字节/参数
# = 14 GB

# KV Cache内存计算
batch_size = 32
seq_length = 2048
num_heads = 32
head_dim = 128
num_layers = 32

# 每个Token的KV Cache大小
kv_per_token = 2 * num_layers * num_heads * head_dim  # K和V
# = 2 * 32 * 32 * 128 = 262,144 字节 = 256 KB

# 总KV Cache内存
kv_cache_memory = batch_size * seq_length * kv_per_token
# = 32 * 2048 * 256 KB = 16 GB

# 总内存需求
total_memory = weight_memory + kv_cache_memory
# = 14 + 16 = 30 GB
```

**关键发现**：
- 对于长序列和大batch，KV Cache内存可能超过模型权重
- 内存带宽成为限制吞吐量的主要瓶颈
- GPU计算单元常常空闲等待数据加载

#### 1.1.3 延迟要求

| 场景 | 首Token延迟 (TTFT) | 生成速度 | 并发数 | 典型应用 |
|------|-------------------|---------|--------|---------|
| 在线对话 | < 200ms | 20-50 tokens/s | 1-10 | ChatGPT、客服 |
| 批量处理 | < 2s | 100-500 tokens/s | 100-1000 | 文档摘要 |
| 实时翻译 | < 100ms | 30-60 tokens/s | 1-5 | 同声传译 |
| 代码生成 | < 500ms | 50-100 tokens/s | 1-20 | Copilot |

**延迟组成**：
```
总延迟 = Prefill阶段延迟 + Decoding阶段延迟
       = (处理提示词)   + (逐Token生成)

Prefill: 计算密集，可并行
Decoding: 内存密集，串行依赖
```

### 1.2 推理性能指标

#### 1.2.1 核心指标定义

**1. 吞吐量 (Throughput)**
```
定义：单位时间内处理的Token数
单位：tokens/second 或 tokens/sec/GPU

测量方法：
- 离线吞吐：固定数据集，测量总处理时间
- 在线吞吐：持续请求流，测量稳态吞吐

优化方向：
- 增大batch size (受内存限制)
- 连续批处理 (Continuous batching)
- KV Cache优化
```

**2. 延迟 (Latency)**
```
定义：从请求到响应的时间
关键指标：
- TTFT (Time To First Token): 首Token延迟
- TPOT (Time Per Output Token): 每Token生成时间
- E2E (End-to-End): 完整响应时间

P99延迟：99%请求的延迟上限（服务等级协议SLA常用）
```

**3. 并发数 (Concurrency)**
```
定义：同时处理的请求数
影响因素：
- GPU内存容量
- KV Cache管理效率
- 请求序列长度分布

理想情况：高并发 + 低延迟（通常矛盾）
```

**4. 成本效率 (Cost Efficiency)**
```
定义：单位计算成本产生的价值
公式：tokens / (GPU小时 × 成本)

优化策略：
- 模型量化 (减少内存)
- 投机解码 (减少计算)
- 动态批处理 (提高利用率)
```

#### 1.2.2 性能测试基准

**标准测试工具**：
```bash
# 1. vLLM性能测试
python benchmarks/benchmark_throughput.py \
    --model meta-llama/Llama-3.1-8B \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000

# 2. TensorRT-LLM性能测试
python benchmarks/python/benchmark.py \
    --model llm \
    --batch_size 32 \
    --input_output_len 2048,2048

# 3. Ollama性能测试
ollama run llama3.1:8b
# 在另一个终端
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Hello",
  "stream": false
}'
```

**性能对比表格（Llama-3.1-8B，A100 80GB）**：

| 引擎 | 吞吐量 (tok/s) | TTFT (ms) | TPOT (ms) | 最大并发 | 显存占用 |
|------|---------------|-----------|-----------|---------|---------|
| vLLM | 8,500 | 45 | 12 | 128 | 22 GB |
| TensorRT-LLM | 10,200 | 38 | 10 | 150 | 20 GB |
| llama.cpp (GGUF Q4) | 3,200 | 120 | 35 | 32 | 8 GB |
| Ollama (默认) | 2,800 | 150 | 40 | 24 | 8 GB |
| SGLang | 9,100 | 42 | 11 | 140 | 21 GB |
| HuggingFace TGI | 6,500 | 55 | 15 | 96 | 25 GB |

> ⚠️ **注意**：以上数据仅供参考，实际性能取决于具体配置和负载特征。

---

## 3. KV Cache 技术

### 3.1 KV Cache 原理

#### 3.1.1 注意力机制回顾

要理解KV Cache，必须先理解Transformer的自注意力机制。

**自注意力计算过程**：

```python
import torch
import torch.nn.functional as F

class SelfAttention:
    def __init__(self, hidden_size, num_heads):
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 可学习参数
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, past_kv=None):
        batch_size, seq_len, hidden_size = x.shape
        
        # 1. 投影得到Q、K、V
        Q = self.q_proj(x)  # [batch, seq, hidden]
        K = self.k_proj(x)  # [batch, seq, hidden]
        V = self.v_proj(x)  # [batch, seq, hidden]
        
        # 2. 多头reshape
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 3. 如果有历史KV，则拼接（这是KV Cache的核心！）
        if past_kv is not None:
            past_K, past_V = past_kv
            K = torch.cat([past_K, K], dim=1)  # 在序列维度拼接
            V = torch.cat([past_V, V], dim=1)
            # 拼接后K、V的seq_len会增长
        
        # 4. 计算注意力分数
        # Q: [batch, seq_q, head, dim]
        # K: [batch, seq_k, head, dim]  (seq_k >= seq_q)
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
        scores = scores / math.sqrt(self.head_dim)
        
        # 5. 因果掩码（防止看到未来）
        mask = torch.triu(
            torch.ones(seq_q, seq_k), 
            diagonal=seq_k - seq_q + 1
        ).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        # 6. Softmax + 加权求和
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bhqk,bkhd->bqhd', attn_weights, V)
        
        # 7. 输出投影
        output = output.reshape(batch_size, seq_len, hidden_size)
        output = self.o_proj(output)
        
        # 8. 返回输出和当前KV（供后续使用）
        return output, (K, V)
```

**关键理解**：
```
生成第t个Token时：
- 需要所有1到t-1位置的K、V
- 每次生成新Token，都会重新计算Q
- 但K、V可以复用历史计算结果

这就是KV Cache的核心思想！
```

#### 3.1.2 KV Cache 作用

**没有KV Cache的情况**：
```
生成过程（低效）：
Step 1: 输入 [x1] → 计算 Q1,K1,V1 → 输出 x2
Step 2: 输入 [x1,x2] → 重新计算 Q1,K1,V1,Q2,K2,V2 → 输出 x3
Step 3: 输入 [x1,x2,x3] → 重新计算所有Q,K,V → 输出 x4

计算量：O(n²) 次注意力计算
```

**有KV Cache的情况**：
```
生成过程（高效）：
Step 1: 输入 [x1] → 计算 Q1,K1,V1 → 缓存(K1,V1) → 输出 x2
Step 2: 输入 [x2] → 计算 Q2,K2,V2 → 拼接缓存 → 缓存(K1,K2,V1,V2) → 输出 x3
Step 3: 输入 [x3] → 计算 Q3,K3,V3 → 拼接缓存 → 缓存(K1..K3,V1..V3) → 输出 x4

计算量：O(n) 次注意力计算（每步只计算新Token的QKV）
```

**性能提升**：
- 对于1000 Token的生成：从 ~500,000 次计算降到 ~1,000 次
- 加速比：约 500x（仅注意力部分）
- 实际端到端加速：2-5x（因为还有其他计算）

#### 3.1.3 内存占用计算

**详细计算公式**：

```python
def calculate_kv_cache_memory(
    batch_size: int,
    seq_length: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2  # FP16=2, FP32=4, INT8=1
) -> int:
    """
    计算KV Cache总内存占用
    
    参数说明：
    - batch_size: 并发请求数
    - seq_length: 每个请求的序列长度
    - num_layers: Transformer层数
    - num_heads: 注意力头数
    - head_dim: 每个头的维度
    - dtype_bytes: 数据类型字节数
    """
    
    # 每个位置的KV Cache大小
    # 2 表示 K 和 V 两个矩阵
    kv_per_position = 2 * num_layers * num_heads * head_dim * dtype_bytes
    
    # 总内存
    total_memory = batch_size * seq_length * kv_per_position
    
    return total_memory

# 示例计算
print("=== KV Cache内存计算示例 ===")

# Llama-3.1-8B 配置
config = {
    "num_layers": 32,
    "num_heads": 32,
    "head_dim": 128,
}

# 场景1：单请求，短序列
mem1 = calculate_kv_cache_memory(
    batch_size=1,
    seq_length=512,
    **config
)
print(f"场景1 (1×512): {mem1 / 1e9:.2f} GB")

# 场景2：多请求，长序列
mem2 = calculate_kv_cache_memory(
    batch_size=32,
    seq_length=4096,
    **config
)
print(f"场景2 (32×4096): {mem2 / 1e9:.2f} GB")

# 场景3：极端情况
mem3 = calculate_kv_cache_memory(
    batch_size=128,
    seq_length=8192,
    **config
)
print(f"场景3 (128×8192): {mem3 / 1e9:.2f} GB")

# 输出：
# 场景1 (1×512): 0.01 GB
# 场景2 (32×4096): 2.15 GB
# 场景3 (128×8192): 17.18 GB
```

**常见模型KV Cache对比**：

| 模型 | 参数量 | 层数 | 头数 | 头维度 | 单Token KV (FP16) | 1024 Tokens | 4096 Tokens |
|------|--------|------|------|--------|------------------|-------------|-------------|
| Llama-3.1-8B | 8B | 32 | 32 | 128 | 256 KB | 256 MB | 1 GB |
| Llama-3.1-70B | 70B | 80 | 64 | 128 | 1 MB | 1 GB | 4 GB |
| Qwen2.5-72B | 72B | 80 | 64 | 128 | 1 MB | 1 GB | 4 GB |
| Mixtral-8x7B | 47B | 32 | 32 | 128 | 256 KB | 256 MB | 1 GB |

> 💡 **洞察**：70B模型的KV Cache是8B模型的4倍，这解释了为什么大模型推理需要更多显存。

#### 3.1.4 缓存命中率

KV Cache的复用效率直接影响性能。

**缓存命中率场景**：

```python
class KVCacheHitRate:
    def analyze_cache_patterns(self):
        """分析不同场景的缓存命中率"""
        
        patterns = {
            "多轮对话": {
                "描述": "用户连续提问，共享历史",
                "命中率": "60-80%",
                "原因": "每轮复用之前的KV",
                "示例": [
                    "用户：介绍Python",
                    "AI：Python是一种...",
                    "用户：它有哪些特性？",  # 复用前面的KV
                    "AI：主要特性包括...",
                ]
            },
            "批量独立请求": {
                "描述": "多个不相关请求",
                "命中率": "0-5%",
                "原因": "每个请求独立，无共享",
                "示例": [
                    "请求1：翻译'Hello'",
                    "请求2：总结这篇文章",
                    "请求3：写一首诗",
                ]
            },
            "系统提示词": {
                "描述": "共享相同的system prompt",
                "命中率": "20-40%",
                "原因": "前缀共享",
                "示例": [
                    "System: 你是一个助手...",  # 所有请求共享
                    "User: 问题1",
                    "User: 问题2",
                ]
            },
            "函数调用": {
                "描述": "结构化输出，固定格式",
                "命中率": "70-90%",
                "原因": "前缀和格式高度可预测",
                "示例": [
                    '{"name": "get_weather",',  # 固定JSON前缀
                    ' "parameters": {"location":',
                ]
            }
        }
        
        return patterns
```

**提升缓存命中率的策略**：

1. **前缀缓存（Prefix Caching）**：
```python
# RadixAttention的核心思想
class PrefixCache:
    def __init__(self):
        self.trie = {}  # 前缀树存储KV Cache
    
    def insert(self, tokens, kv_cache):
        """将KV Cache插入前缀树"""
        node = self.trie
        for i, token in enumerate(tokens):
            if token not in node:
                node[token] = {"kv": None, "children": {}}
            node = node[token]["children"]
        node["kv"] = kv_cache
    
    def find_longest_prefix(self, tokens):
        """找到最长匹配前缀"""
        node = self.trie
        matched_length = 0
        matched_kv = None
        
        for token in tokens:
            if token in node:
                node = node[token]
                matched_length += 1
                if node["kv"] is not None:
                    matched_kv = node["kv"]
            else:
                break
        
        return matched_length, matched_kv
```

2. **请求调度优化**：
```python
def schedule_with_prefix_awareness(requests):
    """优先调度共享前缀的请求"""
    
    # 按前缀分组
    prefix_groups = {}
    for req in requests:
        prefix = req.prompt[:100]  # 取前100个字符
        if prefix not in prefix_groups:
            prefix_groups[prefix] = []
        prefix_groups[prefix].append(req)
    
    # 优先处理大组（缓存命中率高）
    sorted_groups = sorted(
        prefix_groups.values(),
        key=lambda x: len(x),
        reverse=True
    )
    
    return [req for group in sorted_groups for req in group]
```

### 3.2 KV Cache 优化

#### 3.2.1 PagedAttention (vLLM)

**问题背景**：

传统KV Cache管理存在严重的内存碎片问题：

```python
# 传统方式：连续内存分配
class TraditionalKVCache:
    def __init__(self, max_seq_length):
        # 预分配最大长度的连续内存
        self.cache = torch.zeros(
            max_batch_size,
            max_seq_length,  # 必须预留最大值
            num_layers,
            num_heads,
            head_dim
        )
    
    def allocate(self, seq_length):
        # 即使实际长度只有100，也要占用max_seq_length的空间
        # 内存浪费率通常达到60-80%
        pass

# 问题示例：
# 请求1：实际50 tokens，占用2048 tokens空间 → 浪费97.5%
# 请求2：实际200 tokens，占用2048 tokens空间 → 浪费90.2%
# 请求3：实际1000 tokens，占用2048 tokens空间 → 浪费51.2%
```

**PagedAttention解决方案**：

灵感来自操作系统的虚拟内存和分页技术。

```python
class PagedAttention:
    """
    PagedAttention核心思想：
    1. 将KV Cache分成固定大小的块（Block）
    2. 块可以非连续存储在物理内存中
    3. 通过块表（Block Table）映射逻辑序列到物理块
    """
    
    def __init__(self, block_size=16):
        """
        block_size: 每个块包含的token数
        - 太小：管理开销大
        - 太大：内部碎片多
        - 经验值：16-64
        """
        self.block_size = block_size
        self.physical_blocks = {}  # 物理块存储
        self.block_tables = {}  # 逻辑序列 → 物理块映射
        self.free_blocks = []  # 空闲块列表
    
    def allocate_sequence(self, seq_id, seq_length):
        """为序列分配物理块"""
        num_blocks = (seq_length + self.block_size - 1) // self.block_size
        
        # 分配物理块（可以是非连续的）
        allocated_blocks = []
        for _ in range(num_blocks):
            block_id = self.free_blocks.pop()
            allocated_blocks.append(block_id)
        
        # 记录映射关系
        self.block_tables[seq_id] = allocated_blocks
    
    def generate(self, seq_id):
        """生成新token时，只需分配一个块"""
        if seq_id not in self.block_tables:
            return
        
        blocks = self.block_tables[seq_id]
        current_length = len(blocks) * self.block_size
        
        # 需要新块？
        if current_length % self.block_size == 0:
            new_block = self.free_blocks.pop()
            blocks.append(new_block)
    
    def attention(self, query, block_table):
        """
        PagedAttention核心算子
        从非连续的物理块中读取KV，计算注意力
        """
        # CUDA kernel实现（伪代码）
        # 1. 根据block_table读取K、V
        # 2. 计算 Q·K^T
        # 3. Softmax
        # 4. 乘以V
        # 关键：GPU内核直接处理非连续内存
        pass
```

**内存优化效果对比**：

```
场景：100个并发请求，序列长度均匀分布[50, 2000]

传统方式：
- 预分配：100 × 2048 tokens
- 实际使用：100 × 1025 tokens (平均)
- 浪费：50%

PagedAttention：
- 按需分配：100 × (平均块数) 块
- 内部碎片：< 1 block_size tokens
- 浪费：< 5%

内存节省：10倍！
```

**块大小选择策略**：

| 块大小 | 内部碎片 | 管理开销 | 适用场景 |
|--------|---------|---------|---------|
| 16 | 极小 (<2%) | 较高 | 短序列为主 (<512) |
| 32 | 小 (<3%) | 中等 | 混合场景 |
| 64 | 中等 (<5%) | 较低 | 长序列为主 (>1024) |
| 128 | 较大 (<8%) | 低 | 超长序列 (>4096) |

#### 3.2.2 RadixAttention

**核心创新**：

RadixAttention由SGLang团队提出，通过前缀树（Radix Tree）实现跨请求的KV Cache共享。

```python
class RadixAttention:
    """
    Radix Tree结构：
    
         root
        / | \
       A  B  C
      / \    \
     B   D    D
    /         \
   C           E
    
    路径 "ABC"、"ABD"、"CDE" 共享前缀节点
    """
    
    def __init__(self):
        self.tree = RadixNode()
        self.ref_count = {}  # 引用计数
    
    def match_prefix(self, tokens):
        """
        在树中匹配最长前缀
        返回：(匹配长度, KV Cache)
        """
        node = self.tree
        matched_len = 0
        cached_kv = []
        
        for i, token in enumerate(tokens):
            if token in node.children:
                node = node.children[token]
                matched_len += 1
                if node.kv_cache is not None:
                    cached_kv = node.kv_cache
            else:
                break
        
        return matched_len, cached_kv
    
    def insert(self, tokens, kv_cache):
        """插入新的序列，复用已有前缀"""
        node = self.tree
        
        for token in tokens:
            if token not in node.children:
                node.children[token] = RadixNode()
            node = node.children[token]
        
        node.kv_cache = kv_cache
        self.ref_count[node] = self.ref_count.get(node, 0) + 1
    
    def evict(self):
        """LRU淘汰策略"""
        # 找到引用计数为0的节点
        # 按最近访问时间排序
        # 释放内存
        pass
```

**RadixAttention vs PagedAttention**：

| 特性 | PagedAttention (vLLM) | RadixAttention (SGLang) |
|------|----------------------|------------------------|
| 共享粒度 | 请求内 | 跨请求 |
| 数据结构 | 块表 | 前缀树 |
| 内存复用 | 低 | 高 |
| 适用场景 | 独立请求 | 多轮对话、系统提示 |
| 管理开销 | 低 | 中 |
| 缓存命中率 | 0-10% | 30-80% |

#### 3.2.3 KV Cache 量化

**量化原理**：

```python
class KVCacheQuantization:
    """
    KV Cache量化：降低精度以节省内存
    
    精度对比：
    - FP16: 16位，范围±65504，精度3位小数
    - INT8: 8位，范围-128~127，需要缩放因子
    - FP8: 8位，范围可配置，动态精度
    - INT4: 4位，范围-8~7，损失较大
    """
    
    def quantize_int8(self, kv_cache):
        """
        INT8量化（每Token量化）
        """
        # 1. 计算缩放因子
        scale = torch.max(torch.abs(kv_cache)) / 127.0
        
        # 2. 量化
        kv_int8 = torch.round(kv_cache / scale * 127).to(torch.int8)
        
        # 3. 存储 (kv_int8, scale)
        return kv_int8, scale
    
    def dequantize_int8(self, kv_int8, scale):
        """反量化"""
        return kv_int8.float() * scale / 127.0
    
    def quantize_fp8(self, kv_cache):
        """
        FP8量化（NVIDIA Hopper架构支持）
        
        FP8格式：
        - E4M3: 4位指数，3位尾数（适合激活值）
        - E5M2: 5位指数，2位尾数（适合梯度）
        """
        # 硬件原生支持，无需软件转换
        return kv_cache.to(torch.float8_e4m3fn)
    
    def quantize_groupwise(self, kv_cache, group_size=128):
        """
        分组量化（每128个token一组）
        平衡精度和压缩率
        """
        seq_len = kv_cache.shape[1]
        num_groups = (seq_len + group_size - 1) // group_size
        
        quantized_groups = []
        scales = []
        
        for i in range(num_groups):
            start = i * group_size
            end = min((i + 1) * group_size, seq_len)
            group = kv_cache[:, start:end]
            
            scale = torch.max(torch.abs(group)) / 127.0
            q_group = torch.round(group / scale * 127).to(torch.int8)
            
            quantized_groups.append(q_group)
            scales.append(scale)
        
        return quantized_groups, scales
```

**量化效果对比**：

| 量化方法 | 内存压缩 | 精度损失 | 速度影响 | 硬件要求 |
|---------|---------|---------|---------|---------|
| FP16 (基线) | 1x | 无 | 基线 | 通用 |
| INT8 | 2x | <1% | +5-10% | 通用 |
| FP8 | 2x | <2% | +10-20% | H100+ |
| INT4 | 4x | 3-5% | +15-25% | 通用 |
| INT4+SmoothQuant | 4x | 2-3% | +20-30% | 通用 |

**实战配置**：

```bash
# vLLM启用KV Cache量化
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B \
    --kv-cache-dtype int8 \
    --quantization-parameter "group_size=128"

# 或使用FP8（H100）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B \
    --kv-cache-dtype fp8
```

#### 3.2.4 动态 KV Cache 管理

**问题**：
- 固定分配导致浪费
- 突发流量导致OOM
- 长请求阻塞短请求

**解决方案**：

```python
class DynamicKVCacheManager:
    """
    动态KV Cache管理策略
    """
    
    def __init__(self, total_memory):
        self.total_memory = total_memory
        self.used_memory = 0
        self.requests = {}
        self.priority_queue = PriorityQueue()
    
    def admit_request(self, request):
        """
        请求准入控制
        """
        estimated_memory = self.estimate_kv_memory(
            request.batch_size,
            request.max_length
        )
        
        if self.used_memory + estimated_memory > self.total_memory:
            # 内存不足，触发淘汰
            self.evict_low_priority_requests(
                estimated_memory - (self.total_memory - self.used_memory)
            )
        
        # 分配内存
        self.allocate(request, estimated_memory)
        return True
    
    def evict_low_priority_requests(self, needed_memory):
        """
        淘汰低优先级请求
        
        优先级策略：
        1. 请求等待时间（长的优先）
        2. 已处理进度（多的优先）
        3. 用户等级（付费用户优先）
        4. 序列长度（短的优先，释放快）
        """
        candidates = sorted(
            self.requests.values(),
            key=lambda r: (
                r.wait_time,
                r.generated_tokens,
                r.user_priority,
                -r.current_length  # 负号：短序列优先
            )
        )
        
        freed_memory = 0
        for req in candidates:
            if freed_memory >= needed_memory:
                break
            
            self.free_request(req)
            freed_memory += req.memory_usage
    
    def adaptive_batch_size(self):
        """
        动态调整batch size
        
        根据当前内存使用情况自动调整
        """
        memory_utilization = self.used_memory / self.total_memory
        
        if memory_utilization < 0.5:
            return self.max_batch_size * 1.5
        elif memory_utilization < 0.8:
            return self.max_batch_size
        else:
            return self.max_batch_size * 0.5
```

### 3.3 实战示例

#### 3.3.1 vLLM KV Cache 配置

**完整配置示例**：

```python
from vllm import LLM, SamplingParams

# 1. 基础配置
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    tensor_parallel_size=1,
    # KV Cache配置
    gpu_memory_utilization=0.9,  # 使用90% GPU内存
    max_model_len=4096,          # 最大序列长度
    block_size=16,               # PagedAttention块大小
)

# 2. 高级配置
llm_advanced = LLM(
    model="meta-llama/Llama-3.1-8B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95,
    max_model_len=8192,
    block_size=32,
    # KV Cache量化
    kv_cache_dtype="auto",  # 自动选择（FP16/INT8/FP8）
    # 调度策略
    scheduler_policy="priority",
    max_num_seqs=256,       # 最大并发序列数
    max_num_batched_tokens=8192,
)

# 3. 推理
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)

outputs = llm.generate(prompts, sampling_params)

# 4. 监控KV Cache使用
from vllm.engine.metrics import Stats
stats = llm.llm_engine.get_stats()
print(f"KV Cache使用率: {stats.gpu_cache_usage:.2%}")
```

**生产环境配置模板**：

```yaml
# vllm_config.yaml
model: meta-llama/Llama-3.1-8B-Instruct
tensor_parallel_size: 1
gpu_memory_utilization: 0.90
max_model_len: 4096
block_size: 16
kv_cache_dtype: auto
max_num_seqs: 128
max_num_batched_tokens: 4096
enable_prefix_caching: true  # 启用前缀缓存
swap_space: 4  # CPU交换空间(GB)
disable_log_stats: false
```

#### 3.3.2 内存优化

**内存优化检查清单**：

```python
class MemoryOptimization:
    """
    KV Cache内存优化策略
    """
    
    def strategy_1_quantization(self):
        """
        策略1：KV Cache量化
        效果：减少50-75%内存
        """
        # INT8量化
        config = {"kv_cache_dtype": "int8"}
        # 或使用FP8（H100）
        config = {"kv_cache_dtype": "fp8"}
        
    def strategy_2_prefix_caching(self):
        """
        策略2：前缀缓存
        效果：共享请求减少30-60%KV Cache
        """
        config = {
            "enable_prefix_caching": True,
            "prefix_caching_block_size": 16
        }
        
    def strategy_3_dynamic_batching(self):
        """
        策略3：动态批处理
        效果：提高吞吐量2-5x
        """
        # vLLM默认启用连续批处理
        # 调整参数优化
        config = {
            "max_num_seqs": 256,
            "max_num_batched_tokens": 8192
        }
        
    def strategy_4_swap_space(self):
        """
        策略4：CPU交换空间
        效果：支持更大batch，但增加延迟
        """
        config = {"swap_space": 4}  # GB
        
    def strategy_5_tensor_parallel(self):
        """
        策略5：张量并行
        效果：多GPU分担内存
        """
        config = {"tensor_parallel_size": 2}  # 2 GPUs
```

**内存诊断工具**：

```bash
# 1. 实时监控
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'

# 2. vLLM内置监控
curl http://localhost:8000/metrics | grep vllm:gpu_cache_usage

# 3. 详细分析
python -c "
from vllm import LLM
import torch

llm = LLM(model='meta-llama/Llama-3.1-8B')
engine = llm.llm_engine

# 获取内存信息
cache_config = engine.cache_config
print(f'Block size: {cache_config.block_size}')
print(f'Num GPU blocks: {cache_config.num_gpu_blocks}')
print(f'Num CPU blocks: {cache_config.num_cpu_blocks}')
print(f'Memory per block: {cache_config.gpu_memory_utilization * 80 / cache_config.num_gpu_blocks:.2f} GB')
"
```

#### 3.3.3 性能调优

**性能调优决策树**：

```
性能问题诊断：
│
├─ 延迟过高？
│  ├─ TTFT高 → 减小batch size，启用前缀缓存
│  ├─ TPOT高 → 检查KV Cache命中率，考虑量化
│  └─ P99高 → 启用请求优先级，淘汰长尾请求
│
├─ 吞吐量低？
│  ├─ 增大gpu_memory_utilization (0.9→0.95)
│  ├─ 增大max_num_seqs
│  ├─ 启用连续批处理
│  └─ 使用KV Cache量化
│
├─ OOM错误？
│  ├─ 减小max_model_len
│  ├─ 减小block_size
│  ├─ 启用KV Cache量化
│  └─ 增加tensor_parallel_size
│
└─ 缓存命中率低？
   ├─ 启用前缀缓存
   ├─ 使用RadixAttention
   └─ 优化请求调度（前缀感知）
```

**基准测试脚本**：

```python
import time
import asyncio
from openai import AsyncOpenAI

class PerformanceBenchmark:
    def __init__(self, base_url="http://localhost:8000/v1"):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="empty"
        )
    
    async def benchmark_single(self, prompt, max_tokens=100):
        """单请求基准测试"""
        start = time.time()
        
        response = await self.client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        end = time.time()
        total_time = end - start
        num_tokens = len(response.choices[0].message.content.split())
        
        return {
            "latency": total_time,
            "tokens": num_tokens,
            "throughput": num_tokens / total_time
        }
    
    async def benchmark_concurrent(self, prompts, concurrency=10):
        """并发基准测试"""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_generate(prompt):
            async with semaphore:
                return await self.benchmark_single(prompt)
        
        tasks = [limited_generate(p) for p in prompts]
        results = await asyncio.gather(*tasks)
        
        # 统计
        latencies = [r["latency"] for r in results]
        throughputs = [r["throughput"] for r in results]
        
        return {
            "avg_latency": sum(latencies) / len(latencies),
            "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)],
            "avg_throughput": sum(throughputs) / len(throughputs),
            "total_throughput": sum(r["tokens"] for r in results) / max(latencies)
        }

# 运行测试
async def main():
    benchmark = PerformanceBenchmark()
    
    prompts = ["Write a story about AI"] * 100
    
    # 单请求测试
    result = await benchmark.benchmark_single(prompts[0])
    print(f"单请求: {result['throughput']:.2f} tokens/s")
    
    # 并发测试
    result = await benchmark.benchmark_concurrent(prompts, concurrency=32)
    print(f"并发32: {result['total_throughput']:.2f} tokens/s")
    print(f"P99延迟: {result['p99_latency']:.2f}s")

asyncio.run(main())
```

---

## 4. vLLM 高吞吐推理

### 4.1 vLLM 架构

#### 4.1.1 PagedAttention 原理

**操作系统分页的启发**：

```
传统内存管理 vs 虚拟内存：

[传统] 连续分配：
进程A: [████████████] 2048 tokens (实际只需200)
进程B: [████████████] 2048 tokens (实际只需500)
浪费：>70%

[虚拟内存] 分页：
进程A: [██░░░░░░░░░░] → 页表 → 物理页 [1, 5, 8]
进程B: [█████░░░░░░░] → 页表 → 物理页 [2, 3, 9, 10]
浪费：<10%
```

**vLLM的三层抽象**：

```python
class VLLMArchitecture:
    """
    vLLM三层架构
    """
    
    # 层1：逻辑层 - 序列（Sequence）
    class Sequence:
        def __init__(self, seq_id, tokens):
            self.seq_id = seq_id
            self.tokens = tokens  # 逻辑token序列
            self.logical_blocks = []  # 逻辑块列表
    
    # 层2：映射层 - 块表（Block Table）
    class BlockTable:
        def __init__(self):
            self.mapping = {}  # seq_id → [physical_block_ids]
        
        def get_physical_blocks(self, seq_id):
            """将逻辑块映射到物理块"""
            return self.mapping[seq_id]
    
    # 层3：物理层 - 块池（Block Pool）
    class BlockPool:
        def __init__(self, num_blocks, block_size):
            self.blocks = [None] * num_blocks  # 物理块数组
            self.free_list = list(range(num_blocks))
            self.block_size = block_size
        
        def allocate(self):
            """分配一个物理块"""
            return self.free_list.pop()
        
        def free(self, block_id):
            """释放一个物理块"""
            self.free_list.append(block_id)
```

**PagedAttention CUDA Kernel核心逻辑**：

```cuda
// 伪代码：PagedAttention CUDA Kernel
__global__ void paged_attention_kernel(
    float* query,           // [batch, num_heads, head_dim]
    int* block_tables,      // [batch, max_num_blocks]
    float* kv_cache,        // [num_blocks, block_size, 2, num_heads, head_dim]
    int* context_lens,      // [batch]
    float* output,          // [batch, num_heads, head_dim]
    int batch_size,
    int num_heads,
    int head_dim,
    int block_size
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    
    // 1. 读取当前batch的query
    float q = query[batch_idx * num_heads * head_dim + head_idx * head_dim];
    
    // 2. 根据block_table读取KV
    int num_blocks = context_lens[batch_idx] / block_size;
    float attention_sum = 0;
    
    for (int b = 0; b < num_blocks; b++) {
        int physical_block_id = block_tables[batch_idx * max_num_blocks + b];
        
        // 从物理块读取K、V（可能非连续）
        float* k_block = kv_cache + physical_block_id * block_size * 2 * num_heads * head_dim;
        float* v_block = k_block + block_size * num_heads * head_dim;
        
        // 3. 计算注意力分数
        for (int i = 0; i < block_size; i++) {
            float score = dot_product(q, k_block[i * num_heads * head_dim + head_idx * head_dim]);
            float weight = exp(score);
            attention_sum += weight * v_block[i * num_heads * head_dim + head_idx * head_dim];
        }
    }
    
    // 4. 写入输出
    output[batch_idx * num_heads * head_dim + head_idx * head_dim] = attention_sum;
}
```

**关键优化点**：

1. **非连续内存访问优化**：
   - 使用shared memory缓存块元数据
   - warp-level primitives减少同步
   - 预取下一个块的数据

2. **批处理优化**：
   - 动态处理不同长度的序列
   - 避免padding浪费
   - warp-level负载均衡

#### 4.1.2 连续批处理（Continuous Batching）

**传统批处理问题**：

```python
# 静态批处理（低效）
class StaticBatching:
    """
    问题：必须等待batch中所有请求完成
    
    时间线：
    Batch 1: [R1████████] [R2████████████] [R3████]
             ←---------- 必须等R2完成 ----------→
    
    浪费：R1和R3完成后，GPU空闲等待
    """
    
    def process_batch(self, requests):
        # 所有请求同时开始
        # 等待最慢的完成
        # 才能开始下一个batch
        results = []
        for req in requests:
            result = self.generate(req)
            results.append(result)
        return results  # 必须全部完成才返回
```

**连续批处理解决方案**：

```python
class ContinuousBatching:
    """
    vLLM连续批处理：请求完成立即替换
    
    时间线：
    Slot 1: [R1████]→[R4████████]
    Slot 2: [R2████████████]
    Slot 3: [R3████]→[R5██]
    
    优势：
    - 无等待时间
    - GPU持续满载
    - 吞吐量提升2-4x
    """
    
    def __init__(self, max_batch_size):
        self.max_batch_size = max_batch_size
        self.active_requests = {}
        self.waiting_queue = Queue()
    
    def scheduling_loop(self):
        """调度循环"""
        while True:
            # 1. 检查完成的请求
            completed = []
            for req_id, req in self.active_requests.items():
                if req.is_done():
                    completed.append(req_id)
            
            # 2. 移除完成的请求
            for req_id in completed:
                del self.active_requests[req_id]
            
            # 3. 从等待队列填充新请求
            while (len(self.active_requests) < self.max_batch_size 
                   and not self.waiting_queue.empty()):
                new_req = self.waiting_queue.get()
                self.active_requests[new_req.id] = new_req
            
            # 4. 执行一步推理（所有active requests）
            self.step()
    
    def step(self):
        """执行一步生成（一个token）"""
        # 构建batch
        batch = self.build_batch()
        
        # 前向传播
        outputs = self.model.forward(batch)
        
        # 更新所有请求的状态
        for req_id, output in outputs.items():
            req = self.active_requests[req_id]
            req.append_token(output)
```

**性能对比**：

| 批处理策略 | 吞吐量 | 平均延迟 | P99延迟 | GPU利用率 |
|-----------|--------|---------|---------|----------|
| 静态批处理 | 100% | 基准 | 基准 | 40-60% |
| 连续批处理 | 200-400% | -20% | -30% | 80-95% |

#### 4.1.3 内存管理

**内存分配策略**：

```python
class MemoryManager:
    """
    vLLM内存管理器
    """
    
    def __init__(self, gpu_memory_utilization=0.9):
        # 1. 计算可用内存
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        self.available_memory = total_gpu_memory * gpu_memory_utilization
        
        # 2. 预留模型权重内存
        model_size = self.calculate_model_size()
        self.available_memory -= model_size
        
        # 3. 计算可分配的块数
        self.block_size = 16  # tokens
        memory_per_block = self.calculate_block_memory()
        self.num_gpu_blocks = int(self.available_memory / memory_per_block)
        
        # 4. 初始化块池
        self.gpu_block_pool = BlockPool(self.num_gpu_blocks)
        
        # 5. CPU交换空间（可选）
        self.swap_space = 4 * 1024**3  # 4GB
        self.cpu_block_pool = BlockPool(
            int(self.swap_space / memory_per_block)
        )
    
    def allocate_sequence(self, seq):
        """为序列分配内存"""
        # 1. 计算需要的块数
        num_blocks = (seq.length + self.block_size - 1) // self.block_size
        
        # 2. 尝试从GPU分配
        if self.gpu_block_pool.available() >= num_blocks:
            blocks = self.gpu_block_pool.allocate(num_blocks)
            return blocks, "gpu"
        
        # 3. GPU不足，使用CPU交换
        if self.cpu_block_pool.available() >= num_blocks:
            blocks = self.cpu_block_pool.allocate(num_blocks)
            return blocks, "cpu"
        
        # 4. 都不足，触发淘汰
        self.evict_sequences()
        return self.allocate_sequence(seq)  # 递归重试
    
    def evict_sequences(self):
        """淘汰序列释放内存"""
        # LRU策略
        sorted_seqs = sorted(
            self.active_sequences,
            key=lambda s: s.last_access_time
        )
        
        # 淘汰最久未访问的序列
        for seq in sorted_seqs:
            if seq.is_swappable():
                self.swap_to_cpu(seq)
                break
```

**内存监控**：

```bash
# 1. 查看GPU内存使用
nvidia-smi

# 2. vLLM详细内存信息
curl http://localhost:8000/metrics | grep -E "vllm:(gpu|cpu)_cache_usage"

# 3. 自定义监控脚本
python -c "
import requests
import time

while True:
    resp = requests.get('http://localhost:8000/metrics')
    for line in resp.text.split('\n'):
        if 'gpu_cache_usage' in line:
            usage = float(line.split()[-1])
            print(f'GPU Cache: {usage:.2%}')
    time.sleep(1)
"
```

### 4.2 vLLM 实战

#### 4.2.1 安装配置

**安装方式**：

```bash
# 方式1：pip安装（推荐）
pip install vllm

# 方式2：从源码安装（最新特性）
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .

# 方式3：Docker（生产环境）
docker run --gpus all \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```

**依赖检查**：

```bash
# 检查CUDA版本
nvcc --version

# 检查PyTorch CUDA支持
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# 检查vLLM安装
python -c "
import vllm
print(f'vLLM: {vllm.__version__}')
"
```

**完整启动脚本**：

```bash
#!/bin/bash
# start_vllm.sh

# 配置参数
MODEL="meta-llama/Llama-3.1-8B-Instruct"
TP_SIZE=1
GPU_UTIL=0.9
MAX_LEN=4096
PORT=8000

# 启动vLLM
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL} \
    --tensor-parallel-size ${TP_SIZE} \
    --gpu-memory-utilization ${GPU_UTIL} \
    --max-model-len ${MAX_LEN} \
    --port ${PORT} \
    --enable-prefix-caching \
    --kv-cache-dtype auto \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --disable-log-requests false \
    --log-requests-interval 10

# 后台运行
# nohup ./start_vllm.sh > vllm.log 2>&1 &
```

#### 4.2.2 API 部署

**OpenAI兼容API**：

```bash
# 1. Chat Completions
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "system", "content": "你是一个助手"},
      {"role": "user", "content": "介绍下你自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'

# 2. Completions (Legacy)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Hello, world!",
    "max_tokens": 100
  }'

# 3. 流式输出
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "讲个故事"}],
    "stream": true
  }'

# 4. Batch API（离线处理）
curl http://localhost:8000/v1/batch \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-abc123",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }'
```

**Python客户端**：

```python
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"  # vLLM不需要验证
)

# 同步调用
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "解释量子计算"}
    ],
    temperature=0.7,
    max_tokens=1000,
    stream=False
)
print(response.choices[0].message.content)

# 流式调用
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "写一首诗"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# 异步调用
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"
)

async def async_generate():
    response = await async_client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "user", "content": "总结这篇文章"}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

result = asyncio.run(async_generate())
```

#### 4.2.3 性能调优

**调优参数详解**：

```python
# 关键参数及其影响
optimization_params = {
    # 内存相关
    "gpu_memory_utilization": {
        "默认": 0.9,
        "范围": [0.5, 0.95],
        "影响": "越高，可缓存的KV越多，但可能OOM",
        "建议": "0.9-0.95（留一些给系统）"
    },
    
    "max_model_len": {
        "默认": 模型最大长度,
        "影响": "限制单个请求的最大长度",
        "建议": "根据实际需求设置，不要过大"
    },
    
    "block_size": {
        "默认": 16,
        "选项": [16, 32, 64],
        "影响": "小块：碎片少但管理开销大",
        "建议": "短序列用16，长序列用32-64"
    },
    
    # 批处理相关
    "max_num_seqs": {
        "默认": 256,
        "影响": "最大并发序列数",
        "建议": "延迟敏感：64-128，吞吐敏感：256-512"
    },
    
    "max_num_batched_tokens": {
        "默认": max_model_len,
        "影响": "单个batch的最大token数",
        "建议": "通常等于max_model_len"
    },
    
    # 缓存相关
    "enable_prefix_caching": {
        "默认": False,
        "影响": "启用前缀缓存，提高缓存命中率",
        "建议": "多轮对话场景必开"
    },
    
    "kv_cache_dtype": {
        "默认": "auto",
        "选项": ["auto", "fp16", "int8", "fp8"],
        "影响": "量化降低内存，但可能有精度损失",
        "建议": "H100用fp8，其他用int8或auto"
    },
    
    # 交换空间
    "swap_space": {
        "默认": 4,
        "单位": "GB",
        "影响": "CPU交换空间，防止OOM",
        "建议": "4-8GB，但会增加延迟"
    }
}
```

**调优场景配置**：

```python
# 场景1：低延迟在线服务
low_latency_config = {
    "gpu_memory_utilization": 0.85,  # 降低，减少竞争
    "max_num_seqs": 64,              # 小batch，快速响应
    "enable_prefix_caching": True,   # 提高命中率
    "kv_cache_dtype": "fp16",        # 不量化，保证精度
    "swap_space": 0,                 # 不用交换空间
}

# 场景2：高吞吐离线处理
high_throughput_config = {
    "gpu_memory_utilization": 0.95,  # 最大化利用
    "max_num_seqs": 512,             # 大batch
    "enable_prefix_caching": False,  # 独立请求，不需要
    "kv_cache_dtype": "int8",        # 量化，支持更大batch
    "swap_space": 8,                 # 充足交换空间
}

# 场景3：长文档处理
long_context_config = {
    "gpu_memory_utilization": 0.9,
    "max_model_len": 32768,          # 支持长序列
    "block_size": 64,                # 大块，减少管理开销
    "max_num_seqs": 32,              # 小batch，因为每个请求很大
    "kv_cache_dtype": "int8",        # 必须量化，否则OOM
}
```

#### 4.2.4 监控指标

**Prometheus指标**：

```python
# vLLM暴露的关键指标
vllm_metrics = {
    # 吞吐量
    "vllm:generation_tokens_total": "生成的总token数",
    "vllm:num_requests_running": "正在处理的请求数",
    "vllm:num_requests_swapped": "被交换到CPU的请求数",
    
    # 延迟
    "vllm:e2e_request_latency_seconds": "端到端请求延迟",
    "vllm:time_to_first_token_seconds": "首Token延迟",
    "vllm:time_per_output_token_seconds": "每Token生成时间",
    
    # 内存
    "vllm:gpu_cache_usage": "GPU KV Cache使用率",
    "vllm:cpu_cache_usage": "CPU KV Cache使用率",
    "vllm:gpu_prefix_cache_hit_rate": "GPU前缀缓存命中率",
    
    # 批处理
    "vllm:batch_size": "当前batch大小",
    "vllm:num_tokens_total": "总token数",
}

# Grafana仪表板配置
grafana_dashboard = {
    "panels": [
        {
            "title": "吞吐量",
            "metrics": ["rate(vllm:generation_tokens_total[1m])"],
            "type": "timeseries"
        },
        {
            "title": "P99延迟",
            "metrics": ["histogram_quantile(0.99, vllm:e2e_request_latency_seconds)"],
            "type": "timeseries"
        },
        {
            "title": "KV Cache使用率",
            "metrics": ["vllm:gpu_cache_usage"],
            "type": "gauge"
        }
    ]
}
```

**实时监控脚本**：

```python
import requests
import time
from rich.console import Console
from rich.table import Table

console = Console()

def monitor_vllm():
    """实时监控vLLM性能"""
    
    while True:
        try:
            # 获取指标
            response = requests.get("http://localhost:8000/metrics")
            metrics = parse_prometheus_metrics(response.text)
            
            # 构建表格
            table = Table(title="vLLM 实时监控")
            table.add_column("指标")
            table.add_column("值")
            table.add_column("状态")
            
            # GPU Cache
            gpu_usage = metrics.get("vllm:gpu_cache_usage", 0)
            status = "✅" if gpu_usage < 0.8 else "⚠️" if gpu_usage < 0.95 else "🔴"
            table.add_row("GPU Cache", f"{gpu_usage:.2%}", status)
            
            # 运行中请求
            running = metrics.get("vllm:num_requests_running", 0)
            table.add_row("Running Requests", str(running), "✅")
            
            # 吞吐量
            throughput = metrics.get("vllm:generation_tokens_total", 0)
            table.add_row("Total Tokens", f"{throughput:,}", "📊")
            
            # 打印
            console.clear()
            console.print(table)
            
            time.sleep(2)
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            time.sleep(5)

monitor_vllm()
```

---

## 5. TensorRT-LLM 优化

### 5.1 TensorRT-LLM 特性

#### 5.1.1 内核融合（Kernel Fusion）

**优化原理**：

```python
# 未融合：多个独立kernel launch
class UnfusedImplementation:
    def forward(self, x):
        # 每个操作一个kernel，多次GPU启动开销
        x = layer_norm(x)           # Kernel 1
        x = linear_proj1(x)         # Kernel 2
        x = gelu(x)                 # Kernel 3
        x = linear_proj2(x)         # Kernel 4
        x = dropout(x)              # Kernel 5
        return x
    
    # 问题：
    # - 5次kernel launch，每次有延迟
    # - 中间结果写回全局内存
    # - 内存带宽浪费

# 融合：单个kernel完成所有操作
class FusedImplementation:
    """
    TensorRT-LLM内核融合
    
    CUDA Kernel伪代码：
    """
    def fused_kernel(self, x, weights):
        # 单个kernel完成所有操作
        # 数据保持在寄存器/shared memory
        # 减少全局内存访问
        
        # 伪代码
        for i in range(block_size):
            temp = layer_norm(x[i])
            temp = linear(temp, weights[0])
            temp = gelu(temp)
            temp = linear(temp, weights[1])
            output[i] = dropout(temp)
        
        return output
    
    # 优势：
    # - 1次kernel launch
    # - 中间结果在寄存器中
    # - 内存访问减少80%+
```

**融合策略**：

| 融合类型 | 示例 | 加速比 | 适用场景 |
|---------|------|--------|---------|
| 激活融合 | GELU + Linear | 1.5-2x | 所有MLP层 |
| LayerNorm融合 | LayerNorm + Linear | 1.3-1.5x | Transformer块 |
| 注意力融合 | QKV投影 + 注意力 | 2-3x | 注意力层 |
| 残差融合 | Add + Norm | 1.2-1.3x | 残差连接 |

**TensorRT-LLM构建过程**：

```python
import tensorrt_llm
from tensorrt_llm import strllm

# 1. 定义模型
model_config = {
    "model_type": "llama",
    "num_layers": 32,
    "num_heads": 32,
    "hidden_size": 4096,
    "vocab_size": 128256,
}

# 2. 构建TRT引擎
builder = tensorrt_llm.Builder()
network = builder.create_network()

# 应用优化
builder.config.max_batch_size = 32
builder.config.max_input_len = 2048
builder.config.max_output_len = 2048
builder.config.enable_kernel_fusion = True  # 启用内核融合
builder.config.use_paged_kv_cache = True    # 使用Paged KV Cache

# 3. 编译
engine = builder.build(network)
engine.save("llama_trt.engine")
```

#### 5.1.2 量化支持

**量化方法对比**：

```python
class QuantizationMethods:
    """
    TensorRT-LLM支持的量化方法
    """
    
    def weight_only_quantization(self):
        """
        仅权重量化（W8A16, W4A16）
        
        特点：
        - 权重：INT8或INT4
        - 激活：FP16
        - 精度损失小
        - 适合内存受限场景
        """
        config = {
            "quant_mode": "weight_only",
            "weight_only_precision": "int8",  # 或 "int4"
            "per_group_scaling": True,
            "group_size": 128
        }
        return config
    
    def smooth_quant(self):
        """
        SmoothQuant（W8A8）
        
        特点：
        - 权重和激活都是INT8
        - 通过平滑因子平衡难度
        - 加速比更高
        """
        config = {
            "quant_mode": "smooth_quant",
            "alpha": 0.5,  # 平滑因子 [0, 1]
            "per_channel": True
        }
        return config
    
    def fp8_quantization(self):
        """
        FP8量化（Hopper架构）
        
        特点：
        - 原生FP8支持（E4M3）
        - 2x带宽提升
        - 最小精度损失
        """
        config = {
            "quant_mode": "fp8",
            "fp8_format": "e4m3",
            "kv_cache_fp8": True
        }
        return config
```

**量化流程**：

```bash
# 1. 收集校准数据（PTQ需要）
python collect_calibration_data.py \
    --model meta-llama/Llama-3.1-8B \
    --dataset ShareGPT \
    --num_samples 512 \
    --output calibration_data.json

# 2. 执行量化
python build.py \
    --model meta-llama/Llama-3.1-8B \
    --quant_mode smooth_quant \
    --smooth_quant_alpha 0.5 \
    --calib_dataset calibration_data.json \
    --output_dir trt_llm_quantized

# 3. 测试量化模型
python run.py \
    --engine_dir trt_llm_quantized \
    --max_output_len 100 \
    --input_text "Hello, world!"

# 4. 评估精度
python evaluate.py \
    --model trt_llm_quantized \
    --benchmark hellaswag \
    --compare_with_fp16
```

#### 5.1.3 多 GPU 推理

**张量并行（Tensor Parallelism）**：

```python
"""
张量并行原理：将大矩阵拆分到多个GPU

原始矩阵乘法：Y = X × W

拆分后（2 GPU）：
GPU 0: Y_0 = X × W_0  (W的前半部分)
GPU 1: Y_1 = X × W_1  (W的后半部分)

然后：Y = concat(Y_0, Y_1) 或 Y = Y_0 + Y_1（取决于拆分方式）
"""

# TensorRT-LLM张量并行配置
tp_config = {
    "tp_size": 2,  # GPU数量
    "pp_size": 1,  # 流水线并行（通常不需要）
    
    # 通信优化
    "use_custom_all_reduce": True,  # 自定义AllReduce
    "enable_xla": False,  # 不使用XLA
}

# 启动多GPU推理
mpirun -np 2 \
    python run.py \
    --engine_dir trt_llm_engine \
    --tp_size 2
```

**流水线并行（Pipeline Parallelism）**：

```python
"""
流水线并行：将模型层拆分到不同GPU

GPU 0: Layers 0-15
GPU 1: Layers 16-31

数据流：
GPU 0: Layer 0 → Layer 1 → ... → Layer 15 → 发送给GPU 1
GPU 1:                                      → Layer 16 → ... → Layer 31

优势：支持超大模型（>100B）
劣势：流水线气泡（bubble）降低效率
"""

pp_config = {
    "pp_size": 2,
    "tp_size": 1,
    "num_layers_per_stage": 16,
}

# 混合并行（TP + PP）
# 70B模型，4 GPU
hybrid_config = {
    "tp_size": 2,  # 每个副本2 GPU
    "pp_size": 2,  # 2个副本
    "total_gpus": 4
}
```

**多GPU性能**：

| 配置 | 模型 | 吞吐量 | 扩展效率 | 通信开销 |
|------|------|--------|---------|---------|
| 1 GPU | 8B | 100% | 基线 | 0% |
| 2 GPU (TP) | 70B | 180% | 90% | 10% |
| 4 GPU (TP) | 405B | 340% | 85% | 15% |
| 4 GPU (TP+PP) | 405B | 320% | 80% | 20% |

### 5.2 实战部署

#### 5.2.1 模型转换

**完整转换流程**：

```bash
#!/bin/bash
# convert_to_trtllm.sh

# 1. 下载HuggingFace模型
huggingface-cli download meta-llama/Llama-3.1-8B \
    --local-dir ./hf_model

# 2. 转换为TensorRT-LLM格式
python convert_checkpoint.py \
    --model_dir ./hf_model \
    --output_dir ./trtllm_checkpoint \
    --tp_size 1 \
    --dtype float16

# 3. （可选）量化
python convert_checkpoint.py \
    --model_dir ./hf_model \
    --output_dir ./trtllm_checkpoint_int8 \
    --tp_size 1 \
    --quant_mode smooth_quant \
    --smooth_quant_alpha 0.5

# 4. 构建TRT引擎
trtllm-build \
    --checkpoint_dir ./trtllm_checkpoint \
    --output_dir ./trtllm_engine \
    --max_batch_size 32 \
    --max_input_len 2048 \
    --max_output_len 2048 \
    --max_beam_width 1 \
    --enable_paged_kv_cache

# 5. 验证引擎
python run.py \
    --engine_dir ./trtllm_engine \
    --max_output_len 100 \
    --input_text "Explain quantum computing"
```

**转换选项详解**：

```python
conversion_options = {
    # 数据类型
    "dtype": {
        "options": ["float16", "float32", "bfloat16"],
        "建议": "float16（平衡精度和性能）"
    },
    
    # 批处理
    "max_batch_size": {
        "默认": 32,
        "建议": "根据GPU内存调整"
    },
    
    # 序列长度
    "max_input_len": {
        "默认": 1024,
        "建议": "根据实际需求"
    },
    
    "max_output_len": {
        "默认": 1024,
        "建议": "通常与input相同"
    },
    
    # KV Cache
    "enable_paged_kv_cache": {
        "默认": True,
        "建议": "始终启用"
    },
    
    # GPTQ量化
    "use_gptq": {
        "需要": "预量化的GPTQ模型",
        "优势": "精度损失极小"
    }
}
```

#### 5.2.2 性能优化

**优化检查清单**：

```python
class TRTLLMOptimization:
    """
    TensorRT-LLM性能优化清单
    """
    
    def check_1_kernel_fusion(self):
        """确保内核融合启用"""
        # 默认启用，检查日志
        # 应看到 "Fused XXX kernel"
        pass
    
    def check_2_memory_pools(self):
        """优化内存池"""
        config = {
            "gpu_memory_fraction": 0.9,
            "paged_kv_cache": True,
            "max_tokens_in_paged_kv_cache": 8192
        }
        return config
    
    def check_3_batching(self):
        """优化批处理"""
        config = {
            "max_batch_size": 32,
            "max_beam_width": 1,  # beam search会增加内存
            "enable_chunked_context": True  # 分块处理长上下文
        }
        return config
    
    def check_4_quantization(self):
        """应用量化"""
        # 优先级：FP8 > INT8 > W8A16 > W4A16
        configs = [
            {"method": "fp8", "hardware": "H100"},
            {"method": "smooth_quant", "hardware": "A100"},
            {"method": "weight_only_int8", "hardware": "any"},
        ]
        return configs
    
    def check_5_profiling(self):
        """性能分析"""
        # 使用Nsight Systems
        commands = [
            "nsys profile --stats=true python run.py ...",
            "ncu --metrics ... python run.py ..."
        ]
        return commands
```

**性能分析工具**：

```bash
# 1. Nsight Systems（整体性能）
nsys profile --stats=true \
    --output trtllm_profile \
    python run.py \
    --engine_dir ./trtllm_engine

# 2. Nsight Compute（Kernel级别）
ncu --metrics sm__throughput.avg.pct \
    python run.py \
    --engine_dir ./trtllm_engine

# 3. 内置性能分析
python run.py \
    --engine_dir ./trtllm_engine \
    --profiling
```

#### 5.2.3 生产部署

**Triton Inference Server集成**：

```bash
# 1. 启动Triton + TensorRT-LLM
docker run --gpus all \
    --shm-size=2g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    nvcr.io/nvidia/tritonserver:24.01-trtllm-python-py3 \
    tritonserver \
    --model-repository=/models \
    --allow-http=true \
    --allow-grpc=true

# 2. 模型仓库结构
/models
└── tensorrt_llm
    ├── 1
    │   └── model.engine  # TRT引擎
    └── config.pbtxt      # 配置文件

# 3. 配置文件 (config.pbtxt)
cat > /models/tensorrt_llm/config.pbtxt << EOF
name: "tensorrt_llm"
backend: "tensorrtllm"
max_batch_size: 32

input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [-1, -1]
  },
  {
    name: "request_output_len"
    data_type: TYPE_UINT32
    dims: [-1]
  }
]

output [
  {
    name: "output_ids"
    data_type: TYPE_INT32
    dims: [-1, -1]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]

parameters [
  {
    key: "gpt_model_type"
    value: { string_value: "inflight_fused_batching" }
  }
]
EOF
```

**生产环境最佳实践**：

```python
class ProductionDeployment:
    """
    生产部署检查清单
    """
    
    checklist = [
        # 性能
        "✅ 启用内核融合",
        "✅ 启用Paged KV Cache",
        "✅ 应用量化（FP8/INT8）",
        "✅ 优化batch size",
        "✅ 配置内存池",
        
        # 可靠性
        "✅ 健康检查端点",
        "✅ 优雅关闭",
        "✅ 请求超时处理",
        "✅ 错误重试机制",
        
        # 监控
        "✅ Prometheus指标",
        "✅ 分布式追踪",
        "✅ 日志聚合",
        "✅ 告警规则",
        
        # 安全
        "✅ 请求限流",
        "✅ 输入验证",
        "✅ API认证",
        "✅ 速率限制",
        
        # 扩展
        "✅ 负载均衡",
        "✅ 自动扩缩容",
        "✅ 多区域部署",
        "✅ 版本管理",
    ]
```

---

## 6. Ollama 与 llama.cpp

### 6.1 llama.cpp 优化

#### 6.1.1 GGUF 格式

**GGUF设计原理**：

```python
"""
GGUF (GGML Universal File) 格式特点：

1. 自包含：
   - 模型权重
   - 元数据（词汇表、配置）
   - 量化参数
   全部在一个文件中

2. 内存映射（mmap）：
   - 不需要全部加载到内存
   - 按需加载权重
   - 支持大于内存的模型

3. 量化原生支持：
   - 多种量化类型
   - 每层可不同
   - 元数据完整

文件结构：
[GGUF Header]
[Tensor Info 1]
[Tensor Info 2]
[...]
[Tensor Data 1]  (可能量化)
[Tensor Data 2]
[...]
[Metadata]
"""

# GGUF文件读取示例
import gguf

# 读取元数据
reader = gguf.GGUFReader("llama-3.1-8b.Q4_K_M.gguf")

print(f"架构: {reader.metadata['general.architecture']}")
print(f"参数量: {reader.metadata['llama.embedding_length']}")
print(f"量化类型: {reader.metadata['general.quantization_version']}")

# 查看张量
for tensor in reader.tensors:
    print(f"{tensor.name}: {tensor.shape} {tensor.tensor_type}")
```

**GGUF vs 其他格式**：

| 格式 | 文件大小 | 加载速度 | 量化支持 | 跨平台 | 适用场景 |
|------|---------|---------|---------|--------|---------|
| GGUF | 小 | 快（mmap） | 原生 | 是 | 边缘设备、CPU |
| PyTorch (.pt) | 大 | 慢 | 手动 | 否 | 训练、微调 |
| Safetensors | 中 | 中 | 手动 | 部分 | HuggingFace生态 |
| ONNX | 中 | 中 | 有限 | 是 | 跨框架部署 |
| TensorRT | 中 | 快 | 原生 | 否（NVIDIA） | NVIDIA GPU |

#### 6.1.2 量化方法

**量化类型详解**：

```python
class GGUFQuantization:
    """
    llama.cpp量化类型
    """
    
    quantization_types = {
        # 无损量化
        "Q8_0": {
            "描述": "8-bit整数，每32个值一组",
            "大小": "约等于FP16的50%",
            "精度": "几乎无损（<0.1%）",
            "速度": "快",
            "适用": "对精度要求高的场景"
        },
        
        # 高质量量化
        "Q6_K": {
            "描述": "6-bit，K-means聚类",
            "大小": "FP16的37.5%",
            "精度": "极小损失（~0.2%）",
            "速度": "较快",
            "适用": "平衡质量和大小"
        },
        
        "Q5_K_M": {
            "描述": "5-bit，中等质量",
            "大小": "FP16的31.25%",
            "精度": "小损失（~0.5%）",
            "速度": "快",
            "适用": "推荐默认选项"
        },
        
        # 中等量化
        "Q4_K_M": {
            "描述": "4-bit，K-means中等",
            "大小": "FP16的25%",
            "精度": "可接受损失（~1%）",
            "速度": "很快",
            "适用": "资源受限场景"
        },
        
        "Q4_0": {
            "描述": "4-bit，基础版本",
            "大小": "FP16的25%",
            "精度": "较大损失（~2%）",
            "速度": "很快",
            "适用": "旧硬件"
        },
        
        # 低质量量化
        "Q3_K_M": {
            "描述": "3-bit",
            "大小": "FP16的18.75%",
            "精度": "明显损失（~3-5%）",
            "速度": "极快",
            "适用": "极端受限场景"
        },
        
        "Q2_K": {
            "描述": "2-bit",
            "大小": "FP16的12.5%",
            "精度": "严重损失（>5%）",
            "速度": "极快",
            "适用": "仅测试用途"
        },
        
        # 特殊量化
        "IQ4_XS": {
            "描述": "Importance Matrix 4-bit",
            "大小": "FP16的25%",
            "精度": "优于Q4_K_M",
            "速度": "快",
            "适用": "最新推荐（2024+）"
        }
    }
```

**量化实战**：

```bash
# 1. 克隆llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
make -j

# 2. 转换HF模型为GGUF
python convert_hf_to_gguf.py \
    meta-llama/Llama-3.1-8B \
    --outfile llama-3.1-8b.F16.gguf

# 3. 量化（推荐Q4_K_M）
./llama-quantize \
    llama-3.1-8b.F16.gguf \
    llama-3.1-8b.Q4_K_M.gguf \
    Q4_K_M

# 4. 测试量化模型
./llama \
    --model llama-3.1-8b.Q4_K_M.gguf \
    --prompt "Hello, world!" \
    --n_predict 100 \
    --threads 8 \
    --n_gpu_layers 32  # GPU offload层数

# 5. 不同量化对比
for qtype in Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M; do
    echo "Testing $qtype..."
    ./llama-quantize \
        llama-3.1-8b.F16.gguf \
        llama-3.1-8b.${qtype}.gguf \
        ${qtype}
    
    ./llama \
        --model llama-3.1-8b.${qtype}.gguf \
        --prompt "Test prompt" \
        --n_predict 50 \
        --no-display-prompt
    
    ls -lh llama-3.1-8b.${qtype}.gguf
done
```

**量化选择决策树**：

```
选择量化类型：
│
├─ 有A100/H100 GPU？
│  ├─ 是 → 使用FP16或Q8_0（不需要量化）
│  └─ 否 ↓
│
├─ 内存 >= 16GB？
│  ├─ 是 → Q5_K_M 或 Q6_K（高质量）
│  └─ 否 ↓
│
├─ 内存 >= 8GB？
│  ├─ 是 → Q4_K_M（推荐平衡点）
│  └─ 否 ↓
│
└─ 内存 < 8GB？
   ├─ 4-8GB → Q3_K_M
   └─ <4GB → Q2_K（质量会差）
```

#### 6.1.3 CPU/GPU 推理

**GPU Offload**：

```bash
# 完全CPU推理
./llama \
    --model model.Q4_K_M.gguf \
    --threads 8 \
    --n_gpu_layers 0

# 部分GPU offload（推荐）
./llama \
    --model model.Q4_K_M.gguf \
    --threads 8 \
    --n_gpu_layers 20  # 将20层放到GPU

# 完全GPU offload
./llama \
    --model model.Q4_K_M.gguf \
    --n_gpu_layers 999  # 所有层都放GPU

# 多GPU
./llama \
    --model model.Q4_K_M.gguf \
    --n_gpu_layers 999 \
    --tensor-split 50,50  # 两个GPU各50%
```

**性能对比**：

| 配置 | 模型 | 设备 | 速度 (tok/s) | 延迟 |
|------|------|------|-------------|------|
| Q4_K_M, CPU only | 8B | M3 Max | 15-20 | 高 |
| Q4_K_M, GPU offload | 8B | M3 Max + 20层 | 35-45 | 中 |
| Q4_K_M, full GPU | 8B | RTX 4090 | 80-120 | 低 |
| Q8_0, full GPU | 8B | A100 | 60-90 | 低 |
| FP16, full GPU | 8B | A100 | 50-70 | 低 |

### 6.2 Ollama 实战

#### 6.2.1 安装使用

**安装**：

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Docker
docker run -d \
    --gpus all \
    -v ollama:/root/.ollama \
    -p 11434:11434 \
    --name ollama \
    ollama/ollama
```

**基本使用**：

```bash
# 1. 运行模型（自动下载）
ollama run llama3.1:8b

# 2. 交互式对话
>>> 你好，请介绍下自己
>>> 写一首关于AI的诗
>>> /bye  # 退出

# 3. 单行推理
ollama run llama3.1:8b "解释量子计算"

# 4. 从文件输入
ollama run llama3.1:8b < prompt.txt

# 5. 管道输入
echo "翻译：Hello World" | ollama run llama3.1:8b
```

#### 6.2.2 模型管理

```bash
# 列出已下载模型
ollama list

# 下载模型
ollama pull llama3.1:8b
ollama pull llama3.1:70b
ollama pull qwen2.5:72b

# 删除模型
ollama rm llama3.1:8b

# 查看模型信息
ollama show llama3.1:8b

# 创建自定义模型（Modelfile）
cat > Modelfile << EOF
FROM llama3.1:8b
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM """
你是一个专业的编程助手。
- 使用中文回答
- 提供代码示例
- 解释关键概念
"""
EOF

ollama create my-coding-assistant -f Modelfile
ollama run my-coding-assistant
```

**可用模型**：

| 模型 | 大小 | Q4大小 | 最小内存 | 推荐场景 |
|------|------|--------|---------|---------|
| llama3.1:8b | 15.7B | 4.7GB | 6GB | 日常对话、代码 |
| llama3.1:70b | 141B | 39GB | 48GB | 复杂推理、分析 |
| qwen2.5:7b | 15B | 4.4GB | 6GB | 中文任务 |
| qwen2.5:72b | 146B | 41GB | 48GB | 中文复杂任务 |
| mistral:7b | 7.6B | 4.1GB | 6GB | 通用任务 |
| codellama:7b | 7.6B | 3.8GB | 6GB | 代码生成 |

#### 6.2.3 API 服务

**REST API**：

```bash
# 1. 生成（非流式）
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Why is the sky blue?",
  "stream": false
}'

# 2. 生成（流式）
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "写一个Python排序函数",
  "stream": true
}'

# 3. 聊天API
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.1:8b",
  "messages": [
    {
      "role": "system",
      "content": "你是一个助手"
    },
    {
      "role": "user",
      "content": "解释递归"
    }
  ],
  "stream": false
}'

# 4. 带参数
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "创作一个故事",
  "options": {
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 50,
    "num_predict": 500,
    "repeat_penalty": 1.1
  }
}'
```

**Python客户端**：

```python
import ollama

# 1. 简单调用
response = ollama.chat(
    model='llama3.1:8b',
    messages=[
        {'role': 'user', 'content': '为什么天空是蓝色的？'}
    ]
)
print(response['message']['content'])

# 2. 流式调用
stream = ollama.chat(
    model='llama3.1:8b',
    messages=[
        {'role': 'user', 'content': '讲个故事'}
    ],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)

# 3. 异步调用
import asyncio
from ollama import AsyncClient

async def main():
    client = AsyncClient()
    response = await client.chat(
        model='llama3.1:8b',
        messages=[
            {'role': 'user', 'content': '解释机器学习'}
        ]
    )
    print(response['message']['content'])

asyncio.run(main())

# 4. 嵌入（Embeddings）
embedding = ollama.embeddings(
    model='llama3.1:8b',
    prompt='这是一个测试句子'
)
print(f"嵌入维度: {len(embedding['embedding'])}")
```

**性能调优**：

```python
# Ollama配置（~/.ollama/config.json）
{
  "num_parallel": 4,        # 并行请求数
  "max_vram": 0,           # 最大显存（0=自动）
  "gpu_layers": 999,       # GPU offload层数
  "threads": 8,            # CPU线程数
  "context_length": 4096,  # 上下文长度
}

# 环境变量
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_VRAM=16GB
export OLLAMA_HOST=0.0.0.0:11434
```

---

## 7. SGLang 结构化生成

### 7.1 SGLang 特性

#### 7.1.1 结构化输出

**问题**：
```python
# 传统方式：生成后解析（容易失败）
response = llm.generate("输出JSON：{'name': '', 'age': ''}")
try:
    data = json.loads(response)  # 可能失败！
except:
    # 重试？修正？
    pass
```

**SGLang解决方案**：

```python
import sglang as sgl
from sglang import function, gen

# 1. 定义结构化生成程序
@function
def extract_info(s, text):
    s += text + "\n"
    s += "姓名: " + gen("name", stop="\n") + "\n"
    s += "年龄: " + gen("age", stop="\n", regex=r"\d+") + "\n"
    s += "邮箱: " + gen("email", stop="\n", regex=r".+@.+\..+") + "\n"
    return s

# 2. 执行（保证符合格式）
result = extract_info.run(
    text="张三，25岁，邮箱zhangsan@example.com"
)

print(result["name"])   # 保证有值
print(result["age"])    # 保证是数字
print(result["email"])  # 保证是邮箱
```

**JSON结构化生成**：

```python
from pydantic import BaseModel
import sglang as sgl

# 1. 定义Pydantic模型
class Person(BaseModel):
    name: str
    age: int
    email: str

# 2. SGLang自动生成
@function
def extract_person(s, text):
    s += f"从文本中提取信息：{text}\n"
    s += "JSON: " + gen("json", max_tokens=200, regex=Person.schema_json())
    return s

# 3. 解析（保证成功）
result = extract_person.run(text="...")
person = Person.parse_raw(result["json"])
```

#### 7.1.2 并行执行

**并发推理**：

```python
import sglang as sgl
from sglang import function, gen

@function
def answer_question(s, question):
    s += "Q: " + question + "\n"
    s += "A: " + gen("answer", max_tokens=200)
    return s

# 批量并行执行
questions = [
    "什么是量子计算？",
    "解释机器学习",
    "Python和Java的区别？",
    # ... 100个问题
]

# 自动并行
states = answer_question.run_batch(
    [{"question": q} for q in questions],
    num_threads=32  # 并发数
)

# 收集结果
for q, state in zip(questions, states):
    print(f"Q: {q}")
    print(f"A: {state['answer']}\n")
```

#### 7.1.3 与 vLLM 对比

**详细对比**：

| 特性 | vLLM | SGLang |
|------|------|--------|
| **核心优势** | 高吞吐、生产就绪 | 结构化生成、灵活性 |
| **KV Cache** | PagedAttention | RadixAttention（前缀树） |
| **缓存共享** | 请求内 | 跨请求 |
| **结构化输出** | 需要外部库（Guidance） | 原生支持 |
| **编程模型** | API调用 | 命令式编程 |
| **并发控制** | 连续批处理 | run_batch |
| **适用场景** | 在线服务、API | 复杂推理、Agent |
| **性能** | 吞吐最优 | 灵活性和吞吐平衡 |
| **生态** | OpenAI兼容 | Pythonic API |
| **成熟度** | 生产级 | 快速发展中 |

**选择建议**：

```
选择推理引擎：
│
├─ 需要OpenAI兼容API？
│  ├─ 是 → vLLM 或 TensorRT-LLM
│  └─ 否 ↓
│
├─ 需要结构化输出（JSON、正则）？
│  ├─ 是 → SGLang
│  └─ 否 ↓
│
├─ 追求极致吞吐？
│  ├─ 是 → TensorRT-LLM 或 vLLM
│  └─ 否 ↓
│
├─ 资源受限（CPU/边缘）？
│  ├─ 是 → Ollama / llama.cpp
│  └─ 否 ↓
│
└─ 构建AI Agent？
   ├─ 是 → SGLang（灵活性）
   └─ 否 → vLLM（默认选择）
```

---

## 8. 实战场景与最佳实践

### 8.1 场景分析

#### 8.1.1 场景1：多轮对话服务

```python
"""
需求：
- 低延迟（TTFT < 200ms）
- 支持100并发用户
- 多轮对话（缓存命中率高）
"""

# 推荐方案：vLLM + 前缀缓存
config = {
    "engine": "vLLM",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "gpu_memory_utilization": 0.9,
    "enable_prefix_caching": True,  # 关键！
    "kv_cache_dtype": "fp16",
    "max_num_seqs": 128,
    "block_size": 16
}

# 对话历史管理
class ConversationManager:
    def __init__(self):
        self.sessions = {}
    
    def add_message(self, session_id, role, content):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({
            "role": role,
            "content": content
        })
    
    def get_messages(self, session_id):
        return self.sessions[session_id]
    
    def format_prompt(self, session_id):
        """格式化为模型输入"""
        messages = self.sessions[session_id]
        # 复用历史，前缀缓存自动生效
        return messages
```

#### 8.1.2 场景2：批量文档处理

```python
"""
需求：
- 高吞吐（>10000 tokens/s）
- 处理1000个文档
- 延迟不敏感
"""

# 推荐方案：TensorRT-LLM + 大批次
config = {
    "engine": "TensorRT-LLM",
    "model": "meta-llama/Llama-3.1-8B",
    "quantization": "fp8",  # H100
    "max_batch_size": 128,
    "enable_paged_kv_cache": True
}

# 批量处理脚本
def batch_process_documents(documents):
    """批量处理文档"""
    # 分组（每批128个）
    batches = [documents[i:i+128] for i in range(0, len(documents), 128)]
    
    results = []
    for batch in batches:
        # 并行处理
        batch_results = trtllm_engine.generate(batch)
        results.extend(batch_results)
    
    return results
```

#### 8.1.3 场景3：边缘设备部署

```python
"""
需求：
- 内存<8GB
- 无GPU或弱GPU
- 可接受较低速度
"""

# 推荐方案：Ollama + Q4_K_M量化
config = {
    "engine": "Ollama",
    "model": "llama3.1:8b",  # 自动使用Q4_K_M
    "quantization": "Q4_K_M",
    "context_length": 2048,
    "gpu_layers": 20  # 部分GPU offload
}

# 部署脚本
#!/bin/bash
# 启动Ollama
ollama serve &

# 等待就绪
sleep 5

# 加载模型
ollama run llama3.1:8b "测试"

# API访问
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "你好"
}'
```

### 8.2 性能优化检查清单

```markdown
## 9. LLM推理优化检查清单

### 模型层面
- [ ] 选择合适的模型大小（8B vs 70B）
- [ ] 应用量化（FP8/INT8/Q4）
- [ ] 使用Instruct版本（对话优化）
- [ ] 考虑蒸馏模型

### 引擎层面
- [ ] 选择正确的推理引擎
- [ ] 启用PagedAttention/RadixAttention
- [ ] 启用连续批处理
- [ ] 配置合适的block size

### 内存层面
- [ ] 优化gpu_memory_utilization
- [ ] 启用前缀缓存
- [ ] 应用KV Cache量化
- [ ] 配置CPU交换空间

### 批处理层面
- [ ] 调整max_num_seqs
- [ ] 优化max_batch_size
- [ ] 实现请求优先级
- [ ] 避免长尾阻塞

### 监控层面
- [ ] 部署Prometheus监控
- [ ] 设置告警规则
- [ ] 跟踪P99延迟
- [ ] 监控KV Cache命中率

### 生产层面
- [ ] 实现优雅降级
- [ ] 配置请求限流
- [ ] 准备回滚方案
- [ ] 压力测试
```

---

## 10. 2025年最新进展

### 9.1 新兴技术

#### 9.1.1 Speculative Decoding（投机解码）

```python
"""
投机解码原理：
1. 小模型（草稿模型）快速生成多个候选token
2. 大模型（目标模型）并行验证
3. 接受的token直接输出，拒绝的重新生成

效果：
- 加速比：2-3x
- 质量：无损（与大模型相同）
"""

# vLLM实现
llm = LLM(
    model="meta-llama/Llama-3.1-70B",  # 大模型
    speculative_model="meta-llama/Llama-3.1-8B",  # 小模型
    num_speculative_tokens=5,  # 每次猜测5个token
)

# 加速效果
# 原始：70B生成 50 tokens/s
# 投机：70B生成 120 tokens/s (2.4x)
```

#### 9.1.2 Chunked Prefill

```python
"""
分块Prefill：
将长提示词分块处理，减少TTFT

场景：
- 100K上下文
- 传统：TTFT > 10s
- 分块：TTFT < 2s（首块快速返回）
"""

# SGLang实现
sglang.launch_server(
    model_path="meta-llama/Llama-3.1-8B",
    chunked_prefill_size=2048,  # 每块2048 tokens
)
```

#### 9.1.3 MLASched（多级调度）

```python
"""
多级调度：
- Level 1: 请求优先级
- Level 2: 内存分配
- Level 3: 计算调度

目标：
- 提高P99延迟
- 公平性
- 吞吐量
"""
```

### 9.2 基准测试更新

**2025年Q2基准测试（A100 80GB，Llama-3.1-8B）**：

| 引擎 | 版本 | 吞吐 (tok/s) | TTFT (ms) | 特性 |
|------|------|-------------|-----------|------|
| vLLM | 0.8.0 | 9,200 | 42 | 投机解码GA |
| TensorRT-LLM | 0.12 | 11,500 | 35 | FP8原生支持 |
| SGLang | 0.4.0 | 9,800 | 40 | RadixAttention v2 |
| Ollama | 0.5 | 3,100 | 110 | 易用性最佳 |
| TGI | 2.2 | 7,200 | 50 | 企业特性 |

---

## 11. 总结与展望

### 10.1 核心要点

1. **KV Cache是推理优化的关键**：
   - 占总内存50%+
   - PagedAttention解决碎片问题
   - RadixAttention实现跨请求共享

2. **引擎选择取决于场景**：
   - 生产API：vLLM / TensorRT-LLM
   - 结构化生成：SGLang
   - 边缘/本地：Ollama / llama.cpp

3. **量化是必选项**：
   - INT8：通用选择
   - FP8：H100最佳
   - Q4_K_M：CPU/边缘

4. **持续优化永无止境**：
   - 投机解码
   - 分块prefill
   - 多级调度

### 10.2 未来趋势

- **硬件协同设计**：专用推理芯片
- **自动优化**：AutoML for Inference
- **Serverless推理**：按需扩展
- **端侧部署**：手机/浏览器推理
- **绿色AI**：能效优化

---

## 12. 参考资料

### 官方文档
1. vLLM Documentation - https://docs.vllm.ai
2. TensorRT-LLM Documentation - https://nvidia.github.io/TensorRT-LLM
3. llama.cpp GitHub - https://github.com/ggml-org/llama.cpp
4. SGLang Documentation - https://sgl-project.github.io
5. Ollama Documentation - https://ollama.com/docs

### 学术论文
1. **PagedAttention**: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)
2. **RadixAttention**: "SGLang: Efficient Execution of Structured Language Model Programs" (NeurIPS 2024)
3. **Speculative Decoding**: "Speculative Decoding: Lossless Acceleration of Large Language Model Generation" (2024)
4. **SmoothQuant**: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (ICML 2023)
5. **FP8**: "FP8 Formats for Deep Learning" (2024)

### 实践资源
1. vLLM Performance Benchmarks - https://github.com/vllm-project/vllm/benchmarks
2. LLM Performance Optimization Guide - https://huggingface.co/docs/transformers/llm_perf
3. NVIDIA TensorRT-LLM Examples - https://github.com/NVIDIA/TensorRT-LLM/examples
4. llama.cpp Quantization Guide - https://github.com/ggml-org/llama.cpp#quantization

### 工具链
1. Prometheus + Grafana 监控
2. Nsight Systems/Compute 性能分析
3. MLflow 实验追踪
4. OpenAI SDK 统一API

---

> 📝 **笔记维护**：本笔记将随技术发展持续更新  
> 🔗 **相关笔记**：参见 `02-模型量化与压缩.md`、`03-分布式推理.md`  
> 💬 **反馈**：如有错误或补充，请提交Issue

---

*最后更新：2025-06-12 | 版本：1.0.0*
