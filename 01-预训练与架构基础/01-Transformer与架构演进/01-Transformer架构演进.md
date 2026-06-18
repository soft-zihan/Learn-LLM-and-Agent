# LLM 预训练基础

> 📅 **更新时间**: 2026-06-17

---

## 目录

- [1. Transformer 架构演进（2025-2026）](#1-transformer-架构演进2025-2026)
- [2. 大规模数据准备与清洗](#2-大规模数据准备与清洗)
- [3. 分布式训练策略](#3-分布式训练策略)
- [4. 训练优化技术](#4-训练优化技术)
- [5. 实战示例：从头训练 7B 模型](#5-实战示例从头训练-7b-模型)
- [6. 最佳实践](#6-最佳实践)
- [7. 参考资源](#7-参考资源)
- [8. 词元化技术详解](#8-词元化技术详解)
- [9. 混合精度训练](#9-混合精度训练)
- [10. 训练稳定性技巧](#10-训练稳定性技巧)

---

## 1. Transformer 架构演进（2025-2026）

### 1.1 传统 Transformer 的局限

经典的 Transformer 架构自 2017 年提出以来，经历了多年的优化，但在 2025-2026 年面临新的挑战：

- **二次方复杂度**：自注意力机制的计算复杂度为 O(n²)，限制了上下文长度
- **内存瓶颈**：KV Cache 在长上下文场景下占用大量显存
- **推理效率**：自回归生成速度慢，难以满足实时应用需求

### 1.2 混合架构崛起（2025-2026 核心趋势）

2025-2026 年最显著的架构演进是**混合架构（Hybrid Architecture）**的兴起：

#### Nemotron 3 架构（NVIDIA，2026 年 4 月）

Nemotron 3 Super 采用了 Attention + Mamba-2 的交替层设计：

```
Layer 1: Multi-Head Attention
Layer 2: Mamba-2 (State Space Model)
Layer 3: Multi-Head Attention
Layer 4: Mamba-2 (State Space Model)
...
```

**核心优势**：
- 长上下文效率提升 50%
- 参数量减少但性能更强（120B-A12B MoE 配置）
- 适合 Agent 场景的长上下文处理

#### Qwen3.6 架构

Qwen3.6 采用了 Attention + Gated DeltaNet 的混合设计：

```python
# 混合架构伪代码示例
class HybridTransformerBlock(nn.Module):
    def __init__(self, use_attention: bool):
        if use_attention:
            self.layer = MultiHeadAttention(
                num_heads=32,
                head_dim=128,
                attention_type="flash_attention_3"
            )
        else:
            self.layer = GatedDeltaNet(
                state_dim=256,
                feature_map="elu"
            )
    
    def forward(self, x, state=None):
        return self.layer(x, state)
```

### 1.3 状态空间模型（SSM）进展

#### Mamba-3（2026 年 3 月）

Mamba-3 在序列建模方面取得重大突破：

**改进点**：
- 改进的状态空间原理
- 更好的长程依赖捕获
- 线性复杂度 O(n) 而非二次方

**核心公式**：
```
h_t = A * h_{t-1} + B * x_t
y_t = C * h_t

# 其中 A, B, C 是可学习的状态转移矩阵
```

#### Gated DeltaNet-2（2026 年 5 月）

关键创新是**解耦擦除和写入操作**：

```python
class GatedDeltaNet2:
    def __init__(self, state_dim):
        self.erase_gate = nn.Linear(state_dim, state_dim)
        self.write_gate = nn.Linear(state_dim, state_dim)
        self.state = None
    
    def update_state(self, x_t):
        # 解耦擦除和写入
        erase_signal = torch.sigmoid(self.erase_gate(x_t))
        write_signal = torch.sigmoid(self.write_gate(x_t))
        
        # 先擦除，后写入
        self.state = self.state * (1 - erase_signal) + write_signal * x_t
        return self.state
```

### 1.4 注意力机制优化

#### Flash Attention 3（2025）

Flash Attention 3 进一步优化了内存访问模式：

**性能提升**：
- 比 Flash Attention 2 快 1.5-2 倍
- 支持 FP8 精度训练
- 更好的 GPU 利用率

```python
# 使用 Flash Attention 3
from flash_attn import flash_attn_func

# 标准调用
output = flash_attn_func(
    q, k, v,
    dropout_p=0.0,
    causal=True,  # 自回归模型
    window_size=(-1, -1),  # 全局注意力
    alibi_slopes=None
)
```

#### 稀疏注意力模式

2025-2026 年流行的稀疏注意力策略：

1. **滑动窗口注意力**：只关注局部上下文
2. **全局 Token 注意力**：特殊 token（如 [CLS]）关注所有位置
3. **随机注意力**：随机选择部分位置进行注意力计算

```python
class SparseAttention(nn.Module):
    def __init__(self, window_size=4096, global_tokens=64):
        self.window_size = window_size
        self.global_tokens = global_tokens
    
    def forward(self, q, k, v):
        # 滑动窗口注意力
        local_attn = sliding_window_attention(q, k, v, self.window_size)
        
        # 全局 token 注意力
        global_attn = global_attention(q[:self.global_tokens], k, v)
        
        return combine_attention(local_attn, global_attn)
```

### 1.5 激活行为分析（2026 年新发现）

2026 年 3 月的研究《The Spike, the Sparse and the Sink》揭示了：

- **Spike（尖峰）**：少数 token 产生异常大的激活值
- **Sparse（稀疏）**：大多数激活值接近零
- **Sink（汇点）**：注意力倾向于集中在特定位置

**实践意义**：
```python
# 激活值裁剪策略
def clip_activations(x, threshold=10.0):
    """防止激活值爆炸"""
    return torch.clamp(x, -threshold, threshold)

# 注意力温度调节
def attention_with_temperature(q, k, temperature=1.0):
    """调节注意力分布的尖锐程度"""
    scores = torch.matmul(q, k.transpose(-2, -1)) / temperature
    return torch.softmax(scores, dim=-1)
```

## 2. 大规模数据准备与清洗

### 2.1 数据规模与来源

2025-2026 年主流预训练数据集规模：

| 数据集 | 规模 | 语言 | 特点 |
|--------|------|------|------|
| FineWeb | 15T tokens | 多语言 | 高质量网页数据 |
| DCLM-Baseline | 3.7T tokens | 英文 | 严格过滤的 Common Crawl |
| RedPajama V2 | 30T tokens | 多语言 | 最大开源数据集 |
| Dolma V2 | 3T tokens | 英文 | AI2 出品，质量优先 |

### 2.2 数据清洗流水线

现代数据清洗包含多个阶段：

```python
class DataCleaningPipeline:
    """2025 年标准数据清洗流程"""
    
    def __init__(self):
        self.filters = [
            DeduplicationFilter(),      # 去重
            QualityFilter(),            # 质量过滤
            LanguageFilter(),           # 语言识别
            ToxicityFilter(),           # 毒性内容过滤
            PIIRemovalFilter(),         # 个人信息移除
            HeuristicFilter()           # 启发式规则过滤
        ]
    
    def process(self, raw_data):
        data = raw_data
        for filter in self.filters:
            data = filter.apply(data)
            print(f"After {filter.name}: {len(data)} samples")
        return data

class QualityFilter:
    """基于模型的质量过滤"""
    
    def __init__(self, model_name="quality-scorer-v2"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.threshold = 0.5
    
    def apply(self, documents):
        scores = self.score_documents(documents)
        return [doc for doc, score in zip(documents, scores) if score > self.threshold]
    
    def score_documents(self, documents):
        # 批量评分
        inputs = self.tokenizer(documents, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[:, 1]
        return scores.tolist()
```

### 2.3 去重技术

**MinHash + LSH 去重**（2025 年标准方法）：

```python
from datasketch import MinHash, MinHashLSH

class DeduplicationFilter:
    name = "Deduplication"
    
    def __init__(self, threshold=0.8, num_perm=128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.seen = set()
    
    def apply(self, documents):
        unique_docs = []
        for doc in documents:
            minhash = self.compute_minhash(doc)
            doc_id = hash(doc)
            
            # 检查是否重复
            similar_docs = self.lsh.query(minhash)
            if not similar_docs:
                unique_docs.append(doc)
                self.lsh.insert(doc_id, minhash)
        
        return unique_docs
    
    def compute_minhash(self, text):
        m = MinHash(num_perm=self.num_perm)
        # 使用 n-gram 特征
        for d in self.ngrams(text.split(), n=5):
            m.update(" ".join(d).encode('utf8'))
        return m
    
    def ngrams(self, tokens, n):
        return zip(*[tokens[i:] for i in range(n)])
```

### 2.4 数据混合策略

**多源数据混合**（2025-2026 最佳实践）：

```python
class DataMixer:
    """智能数据混合策略"""
    
    def __init__(self, data_sources):
        """
        data_sources: dict
            {
                "web_text": {"path": "...", "ratio": 0.6},
                "books": {"path": "...", "ratio": 0.15},
                "code": {"path": "...", "ratio": 0.15},
                "scientific": {"path": "...", "ratio": 0.1}
            }
        """
        self.sources = data_sources
    
    def create_mixed_dataset(self, target_size):
        mixed_data = []
        
        for name, config in self.sources.items():
            # 按比例采样
            source_size = int(target_size * config["ratio"])
            data = self.load_and_sample(name, source_size)
            mixed_data.extend(data)
        
        # 随机打乱
        random.shuffle(mixed_data)
        return mixed_data
    
    def load_and_sample(self, name, size):
        """从数据源加载并采样"""
        # 实现数据加载逻辑
        pass
```

### 2.5 合成数据生成（2025-2026 趋势）

使用高质量 LLM 生成训练数据：

```python
class SyntheticDataGenerator:
    """使用 LLM 生成合成数据"""
    
    def __init__(self, teacher_model, prompts):
        self.teacher = teacher_model
        self.prompts = prompts
    
    def generate(self, num_samples):
        synthetic_data = []
        
        for prompt in self.prompts:
            # 生成多个变体
            responses = self.teacher.generate(
                prompt,
                num_return_sequences=num_samples // len(self.prompts),
                temperature=0.7,
                top_p=0.9,
                max_length=2048
            )
            
            for response in responses:
                synthetic_data.append({
                    "input": prompt,
                    "output": response,
                    "source": "synthetic"
                })
        
        return synthetic_data

# 2025 年合成数据最佳实践
def create_synthetic_dataset():
    # 1. 使用多个教师模型
    teachers = [
        load_model("gpt-4"),
        load_model("claude-3"),
        load_model("llama-3-70b")
    ]
    
    # 2. 多样化的提示模板
    prompts = load_prompt_templates("diverse_prompts.json")
    
    # 3. 质量过滤
    generator = EnsembleSyntheticGenerator(teachers, prompts)
    raw_data = generator.generate(num_samples=100000)
    
    # 4. 去重和验证
    cleaned_data = validate_and_deduplicate(raw_data)
    
    return cleaned_data
```

## 3. 分布式训练策略

### 3.1 数据并行（Data Parallelism）

**原理**：每个 GPU 持有完整模型副本，处理不同的数据批次

```python
# PyTorch DDP 示例
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def train_ddp(model, train_loader):
    setup_ddp()
    
    # 包装模型
    ddp_model = DDP(model, device_ids=[local_rank])
    
    for batch in train_loader:
        # 每个 GPU 处理不同的 batch
        loss = ddp_model(batch)
        loss.backward()
        
        # DDP 自动梯度同步
        optimizer.step()
        optimizer.zero_grad()
```

**2025 年优化：FSDP（Fully Sharded Data Parallel）**

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

def train_fsdp(model):
    # FSDP 将模型参数、梯度、优化器状态分片
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=size_based_auto_wrap_policy(min_num_params=1e8),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32
        )
    )
    
    # 训练循环与 DDP 相同
    for batch in train_loader:
        loss = fsdp_model(batch)
        loss.backward()
        optimizer.step()
```

**FSDP vs DDP 对比**：

| 特性 | DDP | FSDP |
|------|-----|------|
| 模型副本 | 每个 GPU 完整模型 | 模型分片存储 |
| 显存效率 | 低（完整模型×N） | 高（模型/N） |
| 适用场景 | 小模型（<10B） | 大模型（>10B） |
| 通信开销 | 中 | 低（优化后） |

### 3.2 模型并行（Model Parallelism / Tensor Parallelism）

**原理**：将模型的不同部分分配到不同 GPU

```python
# Megatron-LM 张量并行示例
class TensorParallelLinear(nn.Module):
    """张量并行的线性层"""
    
    def __init__(self, in_features, out_features, tp_size):
        super().__init__()
        self.tp_size = tp_size
        
        # 将权重分片
        self.out_features_per_gpu = out_features // tp_size
        
        # 每个 GPU 只持有部分权重
        self.weight = nn.Parameter(
            torch.randn(in_features, self.out_features_per_gpu)
        )
    
    def forward(self, x):
        # 局部计算
        out = x @ self.weight
        
        # 跨 GPU 聚合（All-Reduce）
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

# 使用 Megatron-Core
from megatron.core import tensor_parallel

class TensorParallelMLP(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super().__init__()
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            hidden_size, ffn_size, gather_output=False
        )
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            ffn_size, hidden_size, input_is_parallel=True
        )
    
    def forward(self, x):
        x = self.dense_h_to_4h(x)
        x = F.gelu(x)
        x = self.dense_4h_to_h(x)
        return x
```

### 3.3 流水线并行（Pipeline Parallelism）

**原理**：将模型的不同层分配到不同 GPU

```python
# PyTorch Pipeline Parallelism
from torch.distributed.pipelining import pipeline, SplitPoint

def create_pipeline_parallel_model(model, num_stages):
    # 定义分割点
    split_policy = {
        'layer.0': SplitPoint.START,
        'layer.12': SplitPoint.START,
        'layer.24': SplitPoint.START,
        'layer.36': SplitPoint.START,
    }
    
    # 创建流水线
    pipe = pipeline(
        model,
        num_stages=num_stages,
        split_points=split_policy
    )
    
    return pipe

# 训练循环
def train_pipeline(pipe_model, data_loader):
    for micro_batch in data_loader:
        # 前向传播（流水线执行）
        output = pipe_model(micro_batch)
        loss = loss_fn(output, target)
        
        # 反向传播
        loss.backward()
```

**流水线调度策略**：

1. **GPipe**：简单的微批次调度
2. **PipeDream**：优化的内存和调度
3. **1F1B（One-Forward-One-Backward）**：2025 年主流

```python
# 1F1B 调度伪代码
def schedule_1f1b(num_micro_batches, num_stages):
    """
    1F1B 调度：每个 stage 执行一次前向传播后立即执行一次反向传播
    最小化显存占用
    """
    schedule = []
    
    # Warmup 阶段
    for i in range(num_stages - 1):
        schedule.append(('forward', i))
    
    #  steady state 阶段
    for i in range(num_micro_batches - num_stages + 1):
        schedule.append(('forward', num_stages - 1 + i))
        schedule.append(('backward', i))
    
    # Cooldown 阶段
    for i in range(num_stages - 1):
        schedule.append(('backward', num_micro_batches - num_stages + 1 + i))
    
    return schedule
```

### 3.4 混合并行策略（5D 并行）

2025-2026 年最先进的训练系统支持 **5D 并行**：

```
5D Parallelism = Data Parallelism 
               + Tensor Parallelism 
               + Pipeline Parallelism 
               + Context Parallelism 
               + Expert Parallelism (MoE)
```

**配置示例**（Megatron-LM + DeepSpeed）：

```yaml
# 训练配置（1024 GPU 集群）
parallelism:
  data_parallel: 32          # DP = 32
  tensor_parallel: 8         # TP = 8
  pipeline_parallel: 4       # PP = 4
  context_parallel: 1        # CP = 1（可选）
  expert_parallel: 1         # EP = 1（MoE 模型）
  
  # 总 GPU 数 = 32 × 8 × 4 = 1024

model:
  hidden_size: 8192
  num_layers: 80
  num_heads: 64
  ffn_hidden_size: 32768
  
training:
  micro_batch_size: 1
  global_batch_size: 2048
  gradient_accumulation_steps: 64
```

**实现代码**：

```python
from megatron.core import mpu
from deepspeed import DeepSpeedEngine

def setup_5d_parallel(config):
    # 初始化进程组
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=config.tensor_parallel,
        pipeline_model_parallel_size=config.pipeline_parallel,
        context_parallel_size=config.context_parallel,
        expert_model_parallel_size=config.expert_parallel
    )
    
    # 构建模型
    model = build_gpt_model(config)
    
    # 使用 DeepSpeed 进行数据并行和优化
    engine = DeepSpeedEngine(
        model=model,
        optimizer=optimizer,
        config=deepspeed_config
    )
    
    return engine
```

## 4. 训练优化技术

### 4.1 混合精度训练（Mixed Precision Training）

**BF16 vs FP16**（2025-2026 标准）：

```python
# BF16 训练（推荐）
from torch.cuda.amp import autocast, GradScaler

def train_mixed_precision(model, data_loader):
    # BF16 比 FP16 更稳定
    scaler = GradScaler(enabled=True)
    
    for batch in data_loader:
        optimizer.zero_grad()
        
        with autocast(dtype=torch.bfloat16):  # 使用 BF16
            loss = model(batch)
        
        # 梯度缩放防止下溢
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**FP8 训练**（2025 年新趋势）：

```python
# NVIDIA Hopper 架构支持 FP8
import transformer_engine as te
from transformer_engine.pytorch import fp8_autocast

def train_fp8(model, data_loader):
    """使用 Transformer Engine 进行 FP8 训练"""
    
    # FP8 训练需要特殊处理
    with fp8_autocast(enabled=True):
        for batch in data_loader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
```

**精度对比**：

| 精度 | 显存占用 | 速度 | 稳定性 | 适用场景 |
|------|----------|------|--------|----------|
| FP32 | 1x | 1x | 最高 | 基线测试 |
| BF16 | 0.5x | 2x | 高 | **推荐** |
| FP16 | 0.5x | 2x | 中 | 需要 Loss Scaling |
| FP8 | 0.25x | 3-4x | 中低 | Hopper GPU |

### 4.2 梯度累积（Gradient Accumulation）

```python
def train_with_gradient_accumulation(model, data_loader, accumulation_steps):
    """
    当 batch size 受限于显存时，使用梯度累积
    """
    optimizer.zero_grad()
    
    for i, batch in enumerate(data_loader):
        # 前向传播
        loss = model(batch) / accumulation_steps
        
        # 反向传播（不更新参数）
        loss.backward()
        
        # 每 accumulation_steps 步更新一次
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

# 计算有效 batch size
# global_batch_size = micro_batch_size × num_gpus × accumulation_steps
# 例：1 × 1024 × 64 = 65,536
```

### 4.3 ZeRO 优化器（DeepSpeed）

**ZeRO 三个阶段**：

```python
import deepspeed

# ZeRO 配置
deepspeed_config = {
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 64,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,  # ZeRO-3
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "gradient_clipping": 1.0
}

# 初始化
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=deepspeed_config
)

# 训练
for batch in data_loader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

**ZeRO 阶段对比**：

| 阶段 | 优化器状态 | 梯度 | 参数 | 适用场景 |
|------|------------|------|------|----------|
| ZeRO-1 | 分片 | 不分片 | 不分片 | 中等模型 |
| ZeRO-2 | 分片 | 分片 | 不分片 | 较大模型 |
| ZeRO-3 | 分片 | 分片 | 分片 | **超大模型** |

### 4.4 激活检查点（Activation Checkpointing）

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerBlock(nn.Module):
    """使用激活检查点减少显存"""
    
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffn = FeedForwardNetwork(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
    
    def forward(self, x):
        # 使用 checkpoint 包装前向传播
        # 不保存中间激活值，反向传播时重新计算
        x = x + checkpoint(self._attention_block, self.norm1(x))
        x = x + checkpoint(self._ffn_block, self.norm2(x))
        return x
    
    def _attention_block(self, x):
        return self.attention(x)
    
    def _ffn_block(self, x):
        return self.ffn(x)

# Megatron-LM 风格的激活检查点
from megatron.core import mpu

def apply_activation_checkpointing(model):
    """自动应用激活检查点"""
    checkpoint_activations = True
    checkpoint_num_layers = 1  # 每层都 checkpoint
    
    for layer in model.layers:
        layer.checkpoint_activations = checkpoint_activations
        layer.checkpoint_num_layers = checkpoint_num_layers
```

### 4.5 学习率调度

**Cosine Warmup + Decay**（2025-2026 标准）：

```python
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

class CosineWarmupScheduler:
    """余弦退火学习率调度"""
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, step_num):
        if step_num < self.warmup_steps:
            # Warmup 阶段：线性增长
            lr = self.base_lr * (step_num / self.warmup_steps)
        else:
            # Cosine 衰减阶段
            progress = (step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr_ratio + (1 - self.min_lr_ratio) * \
                 0.5 * (1 + math.cos(math.pi * progress))
            lr *= self.base_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# 使用示例
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = CosineWarmupScheduler(
    optimizer,
    warmup_steps=2000,
    total_steps=100000,
    min_lr_ratio=0.1
)

for step in range(100000):
    train_step()
    scheduler.step(step)
```

**学习率调度可视化**：

```
学习率
  ↑
3e-4 |     /\
     |    /  \
     |   /    \
     |  /      \
     | /        \
     |/          \___________
     +----------------------→ step
     0    2000         100000
     |<--warmup-->|
```

### 4.6 梯度裁剪（Gradient Clipping）

```python
def train_with_gradient_clipping(model, data_loader, max_norm=1.0):
    for batch in data_loader:
        loss = model(batch)
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        optimizer.step()
        optimizer.zero_grad()

# 自适应梯度裁剪（2025 年新技术）
class AdaptiveGradientClipping:
    """根据参数范数自适应裁剪"""
    
    def __init__(self, clip_factor=0.01, eps=1e-8):
        self.clip_factor = clip_factor
        self.eps = eps
    
    def __call__(self, parameters):
        for param in parameters:
            if param.grad is None:
                continue
            
            param_norm = torch.linalg.norm(param.detach())
            grad_norm = torch.linalg.norm(param.grad)
            
            # 最大允许梯度 = 参数范数 × 裁剪因子
            max_norm = max(self.eps, param_norm * self.clip_factor)
            
            if grad_norm > max_norm:
                param.grad.mul_(max_norm / (grad_norm + self.eps))
```

## 5. 实战示例：从头训练 7B 模型

### 5.1 完整训练脚本

```python
"""
LLM 预训练完整示例（7B 参数）
基于 2025-2026 年最佳实践
"""

import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from megatron.core import mpu
import deepspeed

class LLMPretraining:
    def __init__(self, config):
        self.config = config
        self.setup_distributed()
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
    
    def setup_distributed(self):
        """初始化分布式环境"""
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        
        # 初始化 Megatron 并行组
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.config.tp_size,
            pipeline_model_parallel_size=self.config.pp_size
        )
    
    def setup_model(self):
        """构建模型"""
        config = AutoConfig.from_pretrained(self.config.model_name)
        
        # 修改配置
        config.hidden_size = 4096
        config.num_hidden_layers = 32
        config.num_attention_heads = 32
        config.intermediate_size = 16384
        
        self.model = self.build_model(config)
        
        # 应用激活检查点
        self.apply_checkpointing()
    
    def build_model(self, config):
        """构建 Transformer 模型"""
        from transformers import LlamaForCausalLM
        
        model = LlamaForCausalLM(config)
        
        # 初始化权重
        model.apply(self._init_weights)
        
        return model
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def apply_checkpointing(self):
        """应用激活检查点"""
        for layer in self.model.model.layers:
            # 每层都使用 checkpoint
            layer._forward_pre_hooks.clear()
    
    def setup_data(self):
        """准备训练数据"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # 加载数据集
        dataset = self.load_pretraining_dataset(self.config.data_path)
        
        # 分布式采样器
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank()
        )
        
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.config.micro_batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
    
    def setup_optimizer(self):
        """设置优化器和 DeepSpeed"""
        # DeepSpeed 配置
        ds_config = {
            "train_micro_batch_size_per_gpu": self.config.micro_batch_size,
            "gradient_accumulation_steps": self.config.accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.config.learning_rate,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.1
                }
            },
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "gradient_clipping": 1.0
        }
        
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config=ds_config
        )
    
    def train(self):
        """训练循环"""
        global_step = 0
        scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.config.total_steps
        )
        
        for epoch in range(self.config.num_epochs):
            for batch in self.data_loader:
                # 前向传播
                loss = self.model_engine(batch["input_ids"])
                
                # 反向传播
                self.model_engine.backward(loss.loss)
                
                # 梯度裁剪和参数更新
                self.model_engine.step()
                
                # 学习率调度
                scheduler.step(global_step)
                
                # 日志记录
                if global_step % self.config.log_interval == 0:
                    self.log_metrics(global_step, loss.item())
                
                # 保存检查点
                if global_step % self.config.save_interval == 0:
                    self.save_checkpoint(global_step)
                
                global_step += 1
                
                if global_step >= self.config.total_steps:
                    return
    
    def log_metrics(self, step, loss):
        """记录训练指标"""
        if dist.get_rank() == 0:  # 只在 rank 0 记录
            print(f"Step {step}: Loss = {loss:.4f}, LR = {self.get_lr():.6f}")
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def save_checkpoint(self, step):
        """保存检查点"""
        checkpoint_dir = f"checkpoints/step_{step}"
        self.model_engine.save_checkpoint(checkpoint_dir)

# 运行训练
if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass
    class TrainingConfig:
        model_name: str = "meta-llama/Llama-2-7b-hf"
        data_path: str = "data/pretraining"
        tp_size: int = 8
        pp_size: int = 4
        micro_batch_size: int = 1
        accumulation_steps: int = 64
        learning_rate: float = 3e-4
        warmup_steps: int = 2000
        total_steps: int = 100000
        num_epochs: int = 1
        log_interval: int = 100
        save_interval: int = 5000
    
    config = TrainingConfig()
    trainer = LLMPretraining(config)
    trainer.train()
```

### 5.2 训练监控

```python
import wandb

class TrainingMonitor:
    """训练监控工具"""
    
    def __init__(self, config):
        wandb.init(
            project="llm-pretraining",
            config=config,
            name=config.run_name
        )
    
    def log(self, step, metrics):
        wandb.log(metrics, step=step)
    
    def log_training(self, step, loss, lr, gpu_memory):
        self.log(step, {
            "training_loss": loss,
            "learning_rate": lr,
            "gpu_memory_gb": gpu_memory / 1024**3,
            "step": step
        })
    
    def log_validation(self, step, val_loss, perplexity):
        self.log(step, {
            "validation_loss": val_loss,
            "perplexity": perplexity,
            "step": step
        })

# 使用示例
monitor = TrainingMonitor(config)

for step in range(total_steps):
    loss = train_step()
    monitor.log_training(step, loss, lr, gpu_memory)
    
    if step % eval_interval == 0:
        val_loss = evaluate()
        perplexity = math.exp(val_loss)
        monitor.log_validation(step, val_loss, perplexity)
```

## 6. 最佳实践

### 6.1 训练稳定性

1. **Warmup 至关重要**：至少 2000 步 warmup
2. **使用 BF16 而非 FP16**：更稳定，不需要 loss scaling
3. **梯度裁剪**：防止梯度爆炸，推荐 max_norm=1.0
4. **权重初始化**：使用小方差正态分布（std=0.02）
5. **LayerNorm 位置**：Pre-Norm 比 Post-Norm 更稳定

### 6.2 超参数推荐（7B 模型）

```yaml
# 推荐配置
model:
  hidden_size: 4096
  num_layers: 32
  num_heads: 32
  ffn_hidden_size: 16384

training:
  learning_rate: 3.0e-4
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_eps: 1.0e-8
  
  warmup_steps: 2000
  total_steps: 100000
  min_lr_ratio: 0.1
  
  micro_batch_size: 1
  global_batch_size: 2048
  gradient_accumulation: 64
  
  max_seq_length: 4096
  gradient_clipping: 1.0
```

### 6.3 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| Loss 突然爆炸 | 学习率过高 | 降低学习率，增加 warmup |
| Loss 不下降 | 数据质量问题 | 检查数据清洗流程 |
| 显存溢出 | batch size 过大 | 减小 batch size，启用 ZeRO-3 |
| 训练不稳定 | 精度问题 | 使用 BF16，关闭 FP16 |
| 通信瓶颈 | 并行策略不当 | 调整 TP/PP/DP 比例 |

## 7. 参考资源

- [Nemotron 3 Super Technical Report (2026)](https://arxiv.org/abs/2604.12374)
- [Mamba-3: Improved Sequence Modeling (2026)](https://arxiv.org/abs/2603.15569)
- [Gated DeltaNet-2 (2026)](https://arxiv.org/abs/2605.22791)
- [LLM Research Papers 2026 - Sebastian Raschka](https://magazine.sebastianraschka.com/p/llm-research-papers-2026-part1)
- [Rethinking the Primitives: Next Generation LLM Architecture](https://web.stanford.edu/~jksun/blog/llm-architecture.html)
- [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [The Spike, the Sparse and the Sink (2026)](https://arxiv.org/abs/2603.05498)
- [Flash Attention 3](https://github.com/Dao-AILab/flash-attention)
- [Transformer Engine - FP8 Training](https://github.com/NVIDIA/TransformerEngine)

---

## 8. 词元化技术详解

### 8.1 什么是词元化

词元化（Tokenization）是将原始文本转换为模型可处理的离散单元的过程，是 LLM 预处理的第一步。

**核心概念**：

- **Token（词元）**：文本的最小处理单元，可以是字符、子词或单词
- **Vocabulary（词汇表）**：模型能够识别的所有 token 的集合
- **Tokenizer（词元器）**：执行文本到 token ID 转换的工具

**词汇表大小的影响**：

| 词汇表大小 | 优势 | 劣势 | 典型应用 |
|-----------|------|------|----------|
| 小 (10K-30K) | 模型参数少 | 序列长，推理慢 | 早期模型 |
| 中 (50K-100K) | 平衡性能和效率 | 需要仔细调优 | GPT-2, BERT |
| 大 (100K-200K) | 序列短，效率高 | 嵌入层参数多 | GPT-3, LLaMA |
| 超大 (250K+) | 更好的多语言支持 | 显存占用大 | Qwen, Yi |

**词元化效率**：

```python
# 词元化效率计算
def calculate_tokenization_efficiency(text, tokenizer):
    """计算词元化效率指标"""
    tokens = tokenizer.encode(text)
    
    # 压缩率：字符数 / token 数
    compression_ratio = len(text) / len(tokens)
    
    # 每个 token 平均字符数
    chars_per_token = len(text) / len(tokens)
    
    return {
        "num_chars": len(text),
        "num_tokens": len(tokens),
        "compression_ratio": compression_ratio,
        "chars_per_token": chars_per_token
    }

# 不同语言的典型效率
# 英文：~3-4 字符/token
# 中文：~1-2 字符/token（因为汉字通常是独立 token）
# 日文：~2-3 字符/token
```

**多语言支持策略**：

1. **共享词汇表**：所有语言使用同一词汇表
   - 优点：简单，跨语言迁移好
   - 缺点：词汇表膨胀，低资源语言覆盖差

2. **语言特定词汇表**：每种语言独立词汇表
   - 优点：针对性强，效率高
   - 缺点：模型复杂度增加

3. **混合策略**（2025-2026 主流）：
   - 基础共享词汇表 + 语言特定扩展
   - 动态词汇表加载

```python
class MultiLanguageTokenizer:
    """多语言词元器示例"""
    
    def __init__(self, base_vocab, language_extensions):
        """
        base_vocab: 共享基础词汇表
        language_extensions: dict
            {
                "zh": chinese_extension,
                "en": english_extension,
                "ja": japanese_extension
            }
        """
        self.base_vocab = base_vocab
        self.extensions = language_extensions
        self.current_lang = None
    
    def set_language(self, lang):
        """切换到特定语言词汇表"""
        self.current_lang = lang
    
    def encode(self, text, lang=None):
        """编码文本"""
        if lang:
            self.set_language(lang)
        
        # 使用基础词汇表 + 当前语言扩展
        vocab = self.base_vocab.copy()
        if self.current_lang in self.extensions:
            vocab.update(self.extensions[self.current_lang])
        
        return self._encode_with_vocab(text, vocab)
    
    def _encode_with_vocab(self, text, vocab):
        """使用指定词汇表编码"""
        # 实现编码逻辑
        pass
```

### 8.2 BPE (Byte-Pair Encoding)

**BPE 算法原理**：

BPE 是一种数据驱动的子词分割算法，通过迭代合并最频繁的字符对来构建词汇表。

**算法流程**：

```
1. 初始化词汇表为所有独立字符
2. 统计训练数据中字符对的频率
3. 合并频率最高的字符对
4. 重复步骤 2-3，直到达到目标词汇表大小
```

**Python 实现示例**：

```python
class BPETokenizer:
    """Byte-Pair Encoding 词元器实现"""
    
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.merges = {}  # 合并规则
        self.vocab = {}   # 词汇表
    
    def train(self, texts):
        """训练 BPE 词元器"""
        # 1. 初始化：将文本分割为字符
        word_freqs = self.get_word_frequencies(texts)
        splits = {word: [c for c in word] for word in word_freqs.keys()}
        
        # 2. 迭代合并
        merges_count = 0
        while merges_count < self.vocab_size - len(self.vocab):
            # 统计字符对频率
            pair_freqs = self.compute_pair_frequencies(splits, word_freqs)
            
            if not pair_freqs:
                break
            
            # 找到最频繁的字符对
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # 添加合并规则
            new_token = "".join(best_pair)
            self.merges[best_pair] = new_token
            self.vocab[new_token] = len(self.vocab)
            
            # 应用合并
            splits = self.apply_merges(splits, best_pair)
            merges_count += 1
            
            if merges_count % 1000 == 0:
                print(f"Trained {merges_count} merges...")
    
    def get_word_frequencies(self, texts):
        """统计词频"""
        freqs = {}
        for text in texts:
            for word in text.split():
                word = word + "</w>"  # 添加结束标记
                freqs[word] = freqs.get(word, 0) + 1
        return freqs
    
    def compute_pair_frequencies(self, splits, word_freqs):
        """统计字符对频率"""
        pair_freqs = {}
        for word, freq in word_freqs.items():
            split = splits[word]
            for i in range(len(split) - 1):
                pair = (split[i], split[i+1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
        return pair_freqs
    
    def apply_merges(self, splits, pair):
        """应用合并规则"""
        new_splits = {}
        for word, split in splits.items():
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i+1]) == pair:
                    new_split.append("".join(pair))
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        return new_splits
    
    def encode(self, text):
        """编码文本"""
        # 1. 分割为字符
        tokens = [c for c in text]
        
        # 2. 应用合并规则
        for merge in self.merges.values():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] + tokens[i+1] == merge:
                    new_tokens.append(merge)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # 3. 转换为 ID
        token_ids = [self.vocab[t] for t in tokens if t in self.vocab]
        return token_ids
```

**GPT 系列使用 BPE 的特点**：

- **GPT-2**：50,257 词汇表，使用 BPE
- **GPT-3**：50,257 词汇表，与 GPT-2 兼容
- **GPT-4**：推测使用更大的词汇表（100K+）

**BPE 优缺点分析**：

| 优点 | 缺点 |
|------|------|
| 有效处理未知词 | 对形态丰富的语言效果差 |
| 词汇表大小可控 | 可能产生无意义的子词 |
| 实现简单 | 训练速度慢 |
| 跨语言迁移性好 | 不支持无空格语言（如中文） |

### 8.3 SentencePiece

SentencePiece 是 Google 开发的无监督文本词元器，支持 BPE 和 Unigram 两种模式。

**核心优势**：

- **无监督训练**：直接从原始文本训练，无需预分词
- **语言无关**：支持所有语言，包括无空格语言（中文、日文、泰文等）
- ** reversible**：可以从 token ID 无损还原原始文本

**Unigram 语言模型**：

```python
class UnigramTokenizer:
    """Unigram 语言模型词元器"""
    
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.vocab = {}  # token -> probability
    
    def train(self, texts):
        """训练 Unigram 模型"""
        # 1. 构建初始词汇表（所有子串）
        initial_vocab = self.build_initial_vocab(texts)
        
        # 2. 迭代剪枝
        current_vocab = initial_vocab
        while len(current_vocab) > self.vocab_size:
            # 计算每个 token 的损失
            losses = self.compute_losses(current_vocab, texts)
            
            # 移除损失最大的 10-20% token
            num_to_remove = int(len(current_vocab) * 0.1)
            worst_tokens = sorted(losses, key=losses.get)[:num_to_remove]
            
            for token in worst_tokens:
                del current_vocab[token]
        
        self.vocab = current_vocab
    
    def compute_losses(self, vocab, texts):
        """计算每个 token 的损失"""
        losses = {}
        for token in vocab:
            # 计算移除该 token 后的损失增加
            loss_with = self.compute_loss(vocab, texts)
            loss_without = self.compute_loss(
                {t: p for t, p in vocab.items() if t != token},
                texts
            )
            losses[token] = loss_without - loss_with
        return losses
    
    def encode(self, text):
        """使用 Viterbi 算法编码"""
        # 构建 lattice
        lattice = self.build_lattice(text)
        
        # Viterbi 解码
        best_sequence = self.viterbi_decode(lattice)
        
        return best_sequence
```

**SentencePiece BPE 模式**：

```python
import sentencepiece as spm

# 训练 SentencePiece 模型
spm.SentencePieceTrainer.train(
    input='training_data.txt',
    model_prefix='mymodel',
    vocab_size=32000,
    model_type='bpe',  # 或 'unigram'
    character_coverage=0.9995,  # 字符覆盖率
    max_sentence_length=10000,
    shuffle_input_sentence=True,
    bos_id=0,      # Beginning of Sequence
    eos_id=1,      # End of Sequence
    pad_id=2,      # Padding
    unk_id=3       # Unknown
)

# 加载模型
sp = spm.SentencePieceProcessor(model_file='mymodel.model')

# 编码
tokens = sp.encode("Hello world!", out_type=int)
print(tokens)  # [1234, 567, 89]

# 解码
text = sp.decode(tokens)
print(text)  # "Hello world!"

# 获取 token 字符串
tokens_str = sp.encode("Hello world!", out_type=str)
print(tokens_str)  # ['▁Hello', '▁world', '!']
```

**多语言处理最佳实践**：

```python
# 多语言 SentencePiece 训练
def train_multilingual_sp(texts_dict, vocab_size=50000):
    """
    训练多语言 SentencePiece 模型
    
    texts_dict: {
        "en": [english_texts],
        "zh": [chinese_texts],
        "ja": [japanese_texts]
    }
    """
    # 1. 合并所有文本，按比例采样
    all_texts = []
    for lang, texts in texts_dict.items():
        # 按语言重要性调整权重
        weight = {"en": 0.4, "zh": 0.3, "ja": 0.2, "other": 0.1}.get(lang, 0.05)
        num_samples = int(len(texts) * weight)
        all_texts.extend(texts[:num_samples])
    
    # 2. 写入临时文件
    with open('multilingual_train.txt', 'w', encoding='utf-8') as f:
        for text in all_texts:
            f.write(text + '\n')
    
    # 3. 训练模型
    spm.SentencePieceTrainer.train(
        input='multilingual_train.txt',
        model_prefix='multilingual',
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        # 多语言特定设置
        byte_fallback=True,  # 对未覆盖字符使用字节回退
        vocabulary_output_piece_score=True,
        hard_vocab_limit=False
    )
    
    return spm.SentencePieceProcessor(model_file='multilingual.model')
```

**Google 生态集成**：

- **T5**：使用 SentencePiece (32K vocab)
- **BERT**：后续版本支持 SentencePiece
- **PaLM**：使用 SentencePiece (256K vocab)
- **Gemini**：使用优化的 SentencePiece 变体

### 8.4 TikToken

TikToken 是 OpenAI 开发的快速 BPE 词元器，专为 GPT 模型优化。

**性能对比**：

```python
import tiktoken
import time

# 性能测试
def benchmark_tokenizers(text):
    """对比不同词元器的性能"""
    results = {}
    
    # TikToken
    enc = tiktoken.get_encoding("cl100k_base")
    start = time.time()
    for _ in range(100):
        enc.encode(text)
    results['tiktoken'] = time.time() - start
    
    # SentencePiece
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file='model.model')
    start = time.time()
    for _ in range(100):
        sp.encode(text)
    results['sentencepiece'] = time.time() - start
    
    # HuggingFace Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    start = time.time()
    for _ in range(100):
        tokenizer.encode(text)
    results['huggingface'] = time.time() - start
    
    return results

# 典型结果（处理 1MB 文本）：
# tiktoken:     0.5s  (最快)
# sentencepiece: 2.1s
# huggingface:  3.8s
```

**编码计数与成本控制**：

```python
import tiktoken

class TokenCostCalculator:
    """API 调用成本计算器"""
    
    def __init__(self, model="gpt-4"):
        self.model = model
        self.enc = tiktoken.encoding_for_model(model)
        
        # 2025-2026 年价格（每 1M tokens）
        self.pricing = {
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "claude-3-opus": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet": {"input": 3.0, "output": 15.0}
        }
    
    def count_tokens(self, text):
        """计算 token 数量"""
        return len(self.enc.encode(text))
    
    def estimate_cost(self, prompt, max_tokens):
        """估算 API 调用成本"""
        prompt_tokens = self.count_tokens(prompt)
        
        pricing = self.pricing.get(self.model, {})
        input_cost = (prompt_tokens / 1_000_000) * pricing.get("input", 0)
        output_cost = (max_tokens / 1_000_000) * pricing.get("output", 0)
        
        return {
            "prompt_tokens": prompt_tokens,
            "max_output_tokens": max_tokens,
            "estimated_input_cost": input_cost,
            "estimated_output_cost": output_cost,
            "total_cost": input_cost + output_cost
        }
    
    def optimize_prompt(self, text, max_tokens):
        """优化 prompt 以符合 token 限制"""
        tokens = self.enc.encode(text)
        
        if len(tokens) <= max_tokens:
            return text, len(tokens)
        
        # 截断到最大 token 数
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.enc.decode(truncated_tokens)
        
        return truncated_text, len(truncated_tokens)

# 使用示例
calculator = TokenCostCalculator("gpt-4-turbo")

cost = calculator.estimate_cost(
    prompt="Translate this document to Chinese...",
    max_tokens=2000
)

print(f"Prompt tokens: {cost['prompt_tokens']}")
print(f"Estimated cost: ${cost['total_cost']:.4f}")
```

### 8.5 词元化最佳实践

**词汇表选择指南**：

```python
def choose_vocabulary_size(task, languages, target_seq_length):
    """
    根据任务需求选择词汇表大小
    """
    base_size = 50000  # 基础大小
    
    # 语言调整
    language_factors = {
        "en": 1.0,
        "zh": 1.5,  # 中文需要更大词汇表
        "ja": 1.5,
        "multi": 2.0  # 多语言
    }
    
    lang_factor = max(language_factors.get(lang, 1.0) for lang in languages)
    
    # 序列长度调整
    seq_factor = 1.0
    if target_seq_length > 8192:
        seq_factor = 1.5  # 长序列需要更大词汇表
    
    vocab_size = int(base_size * lang_factor * seq_factor)
    
    # 对齐到 2 的幂（硬件优化）
    vocab_size = 2 ** (vocab_size - 1).bit_length()
    
    return vocab_size

# 示例
print(choose_vocabulary_size("pretraining", ["en"], 4096))     # 65536
print(choose_vocabulary_size("pretraining", ["en", "zh"], 8192))  # 131072
```

**特殊 Token 处理**：

```python
# 常见特殊 Token
SPECIAL_TOKENS = {
    "bos": "<|begin_of_text|>",      # 序列开始
    "eos": "<|end_of_text|>",        # 序列结束
    "pad": "<|padding|>",           # 填充
    "unk": "<|unknown|>",           # 未知词
    "mask": "<|mask|>",             # 掩码（预训练用）
    "sep": "<|separator|>",         # 分隔符
    "cls": "<|classification|>",    # 分类标记
}

# 指令微调特殊 Token
INSTRUCTION_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool|>",
    "tool_response": "<|tool_response|>",
}

# 代码特殊 Token
CODE_TOKENS = {
    "python": "<|python|>",
    "javascript": "<|javascript|>",
    "code_block": "<|code|>",
    "code_end": "<|/code|>",
}

class SpecialTokenManager:
    """特殊 Token 管理器"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_tokens = {}
    
    def add_special_tokens(self, tokens_dict):
        """添加特殊 token 到词元器"""
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(tokens_dict.values())}
        )
        self.special_tokens.update(tokens_dict)
        
        print(f"Added {num_added} special tokens")
        return num_added
    
    def format_conversation(self, messages):
        """格式化对话"""
        formatted = self.special_tokens.get("bos", "")
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            role_token = self.special_tokens.get(role, f"<|{role}|>")
            formatted += f"{role_token}\n{content}\n"
        
        formatted += self.special_tokens.get("assistant", "")
        
        return formatted
```

**多语言词元化策略**：

```python
class MultiLanguageTokenizationStrategy:
    """多语言词元化策略"""
    
    def __init__(self):
        self.strategies = {
            "latin_languages": self.use_bpe,      # 英文、法文等
            "cjk_languages": self.use_character,   # 中文、日文、韩文
            "agglutinative": self.use_morpheme,    # 土耳其文、芬兰文等
        }
    
    def use_bpe(self, text, tokenizer):
        """对拉丁语言使用 BPE"""
        return tokenizer.encode(text)
    
    def use_character(self, text, tokenizer):
        """对 CJK 语言使用字符级词元化"""
        # CJK 字符通常是独立的 token
        return tokenizer.encode(text)
    
    def use_morpheme(self, text, tokenizer):
        """对黏着语使用形态学词元化"""
        # 先进行形态学分析
        morphemes = self.morphological_analysis(text)
        return tokenizer.encode(" ".join(morphemes))
    
    def morphological_analysis(self, text):
        """形态学分析（简化版）"""
        # 实际使用需要专业的形态学分析工具
        # 如 MeCab（日文）、KoNLPy（韩文）
        return text.split()
```

**性能优化技巧**：

```python
# 1. 批量编码
def batch_encode(texts, tokenizer, batch_size=1000):
    """批量编码提高性能"""
    all_tokens = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer.encode_batch(batch)
        all_tokens.extend(tokens)
    
    return all_tokens

# 2. 预分词缓存
from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_encode(text, tokenizer_name):
    """缓存常见文本的编码结果"""
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    return tokenizer.encode(text)

# 3. 异步编码（用于数据加载）
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_encode(texts, tokenizer):
    """异步编码"""
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=4)
    
    tasks = []
    for text in texts:
        task = loop.run_in_executor(executor, tokenizer.encode, text)
        tasks.append(task)
    
    return await asyncio.gather(*tasks)

# 4. 内存映射大文件
def encode_large_file(file_path, tokenizer, chunk_size=10000):
    """编码大文件（使用内存映射）"""
    import mmap
    
    with open(file_path, 'r', encoding='utf-8') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # 分块读取和编码
            all_tokens = []
            chunk = ""
            
            for line in iter(mm.readline, b""):
                chunk += line.decode('utf-8')
                
                if len(chunk) >= chunk_size:
                    tokens = tokenizer.encode(chunk)
                    all_tokens.extend(tokens)
                    chunk = ""
            
            # 处理剩余部分
            if chunk:
                tokens = tokenizer.encode(chunk)
                all_tokens.extend(tokens)
            
            return all_tokens
```

## 9. 混合精度训练

### 9.1 精度格式对比

**浮点数格式对比**：

| 格式 | 总位数 | 符号位 | 指数位 | 尾数位 | 范围 | 精度 |
|------|--------|--------|--------|--------|------|------|
| FP32 | 32 | 1 | 8 | 23 | ±3.4×10³⁸ | ~7 位十进制 |
| FP16 | 16 | 1 | 5 | 10 | ±65504 | ~3 位十进制 |
| BF16 | 16 | 1 | 8 | 7 | ±3.4×10³⁸ | ~2 位十进制 |
| TF32 | 19 | 1 | 8 | 10 | ±3.4×10³⁸ | ~3 位十进制 |
| FP8 (E4M3) | 8 | 1 | 4 | 3 | ±16 | ~1 位十进制 |
| FP8 (E5M2) | 8 | 1 | 5 | 2 | ±57344 | ~0.5 位十进制 |

**TF32 (NVIDIA Ampere 架构，2020+)**：

```python
# TF32 自动启用（Ampere 及更新架构）
import torch

# 默认启用 TF32（矩阵乘法）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# TF32 特点：
# - 使用 FP32 的指数范围（8 位）
# - 使用 FP16 的精度（10 位尾数）
# - 无需代码修改，自动加速
# - 性能提升：最高 2x（A100）

def benchmark_tf32():
    """对比 TF32 性能"""
    a = torch.randn(10000, 10000, device='cuda', dtype=torch.float32)
    b = torch.randn(10000, 10000, device='cuda', dtype=torch.float32)
    
    # 启用 TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    tf32_time = time.time() - start
    
    # 禁用 TF32
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    fp32_time = time.time() - start
    
    print(f"TF32: {tf32_time:.3f}s")
    print(f"FP32: {fp32_time:.3f}s")
    print(f"Speedup: {fp32_time/tf32_time:.2f}x")
```

**FP8 (NVIDIA Hopper 架构，2022+)**：

```python
# FP8 需要 Transformer Engine
import transformer_engine as te
from transformer_engine.common import recipe

# FP8 格式
# E4M3：适合前向传播（更高精度）
# E5M2：适合反向传播（更大范围）

fp8_format = recipe.DelayedScaling(
    fp8_format=recipe.Format.HYBRID,  # E4M3 for FWD, E5M2 for BWD
    amax_history_len=16,
    amax_compute_algo="max",
)

# 使用 FP8
def train_with_fp8(model, data_loader):
    """FP8 训练"""
    for batch in data_loader:
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_format):
            output = model(batch)
            loss = loss_fn(output, target)
        
        loss.backward()
        optimizer.step()
```

**精度 vs 速度权衡**：

```python
def analyze_precision_tradeoff():
    """分析不同精度的性能"""
    results = {}
    
    dtypes = [torch.float32, torch.bfloat16, torch.float16]
    
    for dtype in dtypes:
        model = LargeModel().cuda().to(dtype)
        
        # 测量显存
        mem_before = torch.cuda.memory_allocated()
        input_tensor = torch.randn(64, 512, device='cuda', dtype=dtype)
        
        # 测量速度
        start = time.time()
        for _ in range(100):
            output = model(input_tensor)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        mem_after = torch.cuda.memory_allocated()
        
        results[str(dtype)] = {
            "memory_gb": (mem_after - mem_before) / 1e9,
            "time_s": elapsed,
            "throughput": 64 * 100 / elapsed  # samples/s
        }
    
    return results
```

### 9.2 混合精度训练原理

**核心思想**：
- **前向传播**：使用低精度（FP16/BF16）加速计算
- **反向传播**：使用低精度计算梯度
- **梯度缩放**：防止小梯度下溢
- **Master Weights**：使用 FP32 保存权重副本

**混合精度训练流程**：

```python
class MixedPrecisionTrainer:
    """混合精度训练器"""
    
    def __init__(self, model, optimizer, dtype=torch.bfloat16):
        self.model = model
        self.optimizer = optimizer
        self.dtype = dtype
        
        # Master Weights（FP32 副本）
        self.master_weights = {
            name: param.data.clone().float()
            for name, param in model.named_parameters()
        }
        
        # GradScaler（仅 FP16 需要）
        self.scaler = GradScaler() if dtype == torch.float16 else None
    
    def train_step(self, batch):
        """单步训练"""
        self.optimizer.zero_grad()
        
        if self.dtype == torch.float16:
            # FP16 混合精度（需要 GradScaler）
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = self.model(batch)
            
            # 缩放梯度防止下溢
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新权重
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        else:  # BF16
            # BF16 混合精度（不需要 GradScaler）
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = self.model(batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # 同步 Master Weights
        self.sync_master_weights()
    
    def sync_master_weights(self):
        """同步 Master Weights 到模型"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(self.master_weights[name].to(param.dtype))
```

**梯度缩放（GradScaler）原理**：

```python
class CustomGradScaler:
    """自定义 GradScaler 实现"""
    
    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = 1000
        self.steps_since_last_growth = 0
    
    def scale_loss(self, loss):
        """缩放损失"""
        return loss * self.scale
    
    def unscale_gradients(self, optimizer):
        """反缩放梯度"""
        for param in optimizer.param_groups[0]['params']:
            if param.grad is not None:
                param.grad.data.div_(self.scale)
    
    def update(self, found_inf=False):
        """更新缩放因子"""
        self.steps_since_last_growth += 1
        
        if found_inf:
            # 发现 Inf/NaN，减小缩放因子
            self.scale *= self.backoff_factor
            self.steps_since_last_growth = 0
        elif self.steps_since_last_growth >= self.growth_interval:
            # 稳定增长，增加缩放因子
            self.scale *= self.growth_factor
            self.steps_since_last_growth = 0
```

### 9.3 PyTorch AMP 实现

**自动混合精度（AMP）**：

```python
from torch.cuda.amp import autocast, GradScaler

# 基本使用
scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in data_loader:
        optimizer.zero_grad()
        
        # 前向传播（自动使用 FP16）
        with autocast():
            output = model(batch)
            loss = loss_fn(output, target)
        
        # 反向传播（缩放梯度）
        scaler.scale(loss).backward()
        
        # 反缩放梯度
        scaler.unscale_(optimizer)
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新权重
        scaler.step(optimizer)
        scaler.update()
```

**手动精度控制**：

```python
class ManualPrecisionControl:
    """手动控制不同层的精度"""
    
    def __init__(self, model):
        self.model = model
    
    def set_precision_by_layer(self):
        """为不同层设置不同精度"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 线性层使用 BF16
                module.to(torch.bfloat16)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm 保持 FP32（数值稳定性）
                module.to(torch.float32)
            elif isinstance(module, nn.Embedding):
                # Embedding 保持 FP32
                module.to(torch.float32)
    
    def forward_mixed_precision(self, x):
        """前向传播使用混合精度"""
        # Embedding: FP32
        x = self.model.embedding(x.to(torch.float32))
        
        # Transformer Layers: BF16
        with autocast(dtype=torch.bfloat16):
            for layer in self.model.layers:
                x = layer(x)
        
        # Output Layer: FP32
        x = self.model.output_layer(x.to(torch.float32))
        
        return x
```

### 9.4 BF16 训练

**BF16 优势**：

```python
def compare_bf16_vs_fp16():
    """对比 BF16 和 FP16"""
    
    # BF16 优势：
    # 1. 与 FP32 相同的指数范围（8 位）
    # 2. 不需要 GradScaler
    # 3. 更稳定，不易溢出
    # 4. 适合大模型训练
    
    # FP16 劣势：
    # 1. 指数范围小（5 位）
    # 2. 需要 GradScaler 防止下溢
    # 3. 容易溢出（>65504）
    # 4. 需要仔细调参
    
    # 数值范围对比
    print(f"FP16 range: [-65504, 65504]")
    print(f"BF16 range: [-3.4e38, 3.4e38]")  # 与 FP32 相同
    
    # 测试溢出
    fp16_tensor = torch.tensor([70000], dtype=torch.float16)
    bf16_tensor = torch.tensor([70000], dtype=torch.bfloat16)
    
    print(f"FP16(70000): {fp16_tensor}")  # inf（溢出）
    print(f"BF16(70000): {bf16_tensor}")  # 70000（正常）
```

**梯度溢出预防**：

```python
class BF16TrainingGuard:
    """BF16 训练保护机制"""
    
    def __init__(self, model):
        self.model = model
        self.nan_detected = False
    
    def check_gradients(self):
        """检查梯度是否包含 NaN/Inf"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"NaN/Inf detected in {name}")
                    self.nan_detected = True
                    return False
        return True
    
    def check_activations(self, tensor, name=""):
        """检查激活值"""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"NaN/Inf in {name}")
            return False
        return True
    
    def safe_backward(self, loss):
        """安全的反向传播"""
        loss.backward()
        
        if not self.check_gradients():
            # 发现 NaN/Inf，跳过此步
            self.model.zero_grad()
            return False
        
        return True
```

**硬件支持**：

```python
def check_hardware_bf16_support():
    """检查硬件 BF16 支持"""
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    # BF16 支持：Ampere (8.0) 及更新架构
    major, minor = props.major, props.minor
    
    if major >= 8:
        print(f"✓ {props.name} supports BF16 natively")
        return True
    else:
        print(f"✗ {props.name} does not support BF16 natively")
        print("  Use FP16 with GradScaler instead")
        return False

# 支持 BF16 的 GPU：
# - A100 (Ampere, 8.0)
# - A6000 (Ampere, 8.6)
# - RTX 3090 (Ampere, 8.6)
# - H100 (Hopper, 9.0)
# - H200 (Hopper, 9.0)
```

### 9.5 FP8 训练前沿

**FP8 格式详解**：

```python
# FP8 有两种格式：

# E4M3 (4 位指数，3 位尾数)
# - 范围：[-16, 16]
# - 精度：较高
# - 用途：前向传播

# E5M2 (5 位指数，2 位尾数)
# - 范围：[-57344, 57344]
# - 精度：较低
# - 用途：反向传播（梯度）

def fp8_format_comparison():
    """FP8 格式对比"""
    
    formats = {
        "FP8 E4M3": {"exp": 4, "mantissa": 3, "range": "[-16, 16]"},
        "FP8 E5M2": {"exp": 5, "mantissa": 2, "range": "[-57344, 57344]"},
        "FP16": {"exp": 5, "mantissa": 10, "range": "[-65504, 65504]"},
        "BF16": {"exp": 8, "mantissa": 7, "range": "[-3.4e38, 3.4e38]"},
    }
    
    for fmt, props in formats.items():
        print(f"{fmt}:")
        print(f"  Exponent: {props['exp']} bits")
        print(f"  Mantissa: {props['mantissa']} bits")
        print(f"  Range: {props['range']}")
```

**NVIDIA H100 FP8 支持**：

```python
# H100 硬件加速 FP8
import transformer_engine as te

# Transformer Engine 自动处理 FP8
def train_h100_fp8(model, data_loader):
    """在 H100 上使用 FP8 训练"""
    
    # 配置 FP8
    fp8_recipe = recipe.DelayedScaling(
        fp8_format=recipe.Format.HYBRID,  # 自动选择 E4M3/E5M2
        amax_history_len=16,  # 历史窗口大小
        amax_compute_algo="max",  # 使用最大值
    )
    
    # 训练循环
    for batch in data_loader:
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            output = model(batch)
            loss = loss_fn(output, target)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**训练稳定性技巧**：

```python
class FP8TrainingStabilizer:
    """FP8 训练稳定性工具"""
    
    def __init__(self, model):
        self.model = model
        self.amax_history = []  # 记录最大绝对值
    
    def monitor_activations(self):
        """监控激活值范围"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                amax = param.grad.abs().max().item()
                self.amax_history.append(amax)
                
                # 如果 amax 过大，调整缩放因子
                if amax > 10.0:
                    print(f"Warning: Large amax in {name}: {amax}")
    
    def adjust_scaling(self):
        """动态调整缩放因子"""
        if len(self.amax_history) < 16:
            return
        
        recent_max = max(self.amax_history[-16:])
        
        # 如果激活值范围合适，保持
        # 如果太大，减小缩放因子
        # 如果太小，增加缩放因子
```

**性能提升预期**：

```python
def estimate_fp8_speedup():
    """估算 FP8 性能提升"""
    
    improvements = {
        "A100 (FP16)": {"memory": "1x", "speed": "1x", "baseline": True},
        "H100 (FP16)": {"memory": "1x", "speed": "2-3x", "baseline": False},
        "H100 (BF16)": {"memory": "0.5x", "speed": "2-3x", "baseline": False},
        "H100 (FP8)": {"memory": "0.25x", "speed": "3-4x", "baseline": False},
    }
    
    print("Performance Comparison (vs A100 FP16):")
    print("-" * 60)
    
    for platform, metrics in improvements.items():
        print(f"{platform}:")
        print(f"  Memory: {metrics['memory']}")
        print(f"  Speed: {metrics['speed']}")
        if metrics['baseline']:
            print(f"  ** Baseline **")
```

## 10. 训练稳定性技巧

### 10.1 梯度裁剪

**梯度爆炸问题**：

```python
def demonstrate_gradient_explosion():
    """演示梯度爆炸"""
    
    # 简单 RNN 示例
    class SimpleRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(64, 64)
            self.output = nn.Linear(64, 10)
        
        def forward(self, x, h):
            h = torch.tanh(self.hidden(x + h))
            return h, self.output(h)
    
    model = SimpleRNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 长序列训练
    sequence_length = 1000
    h = torch.zeros(64)
    
    for t in range(sequence_length):
        x = torch.randn(64)
        h, output = model(x, h)
        
        # 随着序列增长，梯度可能爆炸
        loss = loss_fn(output, target)
        loss.backward()
        
        # 检查梯度范数
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        print(f"Step {t}: Grad Norm = {grad_norm:.4f}")
```

**裁剪策略**：

```python
class GradientClippingStrategies:
    """梯度裁剪策略"""
    
    @staticmethod
    def clip_by_norm(parameters, max_norm=1.0):
        """按范数裁剪（最常用）"""
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    
    @staticmethod
    def clip_by_value(parameters, min_value=-1.0, max_value=1.0):
        """按值裁剪"""
        for param in parameters:
            if param.grad is not None:
                param.grad.data.clamp_(min_value, max_value)
    
    @staticmethod
    def adaptive_clipping(parameters, clip_factor=0.01):
        """自适应裁剪"""
        for param in parameters:
            if param.grad is None:
                continue
            
            param_norm = torch.linalg.norm(param.detach())
            grad_norm = torch.linalg.norm(param.grad)
            
            max_norm = max(1e-6, param_norm * clip_factor)
            
            if grad_norm > max_norm:
                param.grad.mul_(max_norm / (grad_norm + 1e-8))
    
    @staticmethod
    def global_clipping(parameters, max_norm=1.0):
        """全局裁剪（所有参数共享）"""
        total_norm = 0
        for param in parameters:
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in parameters:
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return total_norm
```

**阈值选择指南**：

| 模型类型 | 推荐阈值 | 说明 |
|---------|---------|------|
| Transformer | 1.0 | 标准值 |
| RNN/LSTM | 5.0 | 更容易爆炸 |
| 超大模型 (>100B) | 0.5-1.0 | 更保守 |
| FP16 训练 | 1.0 | 需要 GradScaler |
| BF16 训练 | 1.0-2.0 | 更稳定 |

### 10.2 Z-Loss

**概念与作用**：

Z-Loss 是 Google 在 PaLM 中提出的训练稳定性技巧，通过惩罚 logits 的范数来防止训练崩溃。

**数学公式**：

```
Z-Loss = (1/B) * Σ (log Σ exp(logits_i))²

其中：
- B: batch size
- logits_i: 第 i 个样本的 logits
- 目的：防止 logits 过大
```

**实现方式**：

```python
class ZLoss(nn.Module):
    """Z-Loss 实现"""
    
    def __init__(self, weight=1e-4):
        super().__init__()
        self.weight = weight
    
    def forward(self, logits):
        """
        计算 Z-Loss
        
        logits: [batch_size, seq_len, vocab_size]
        """
        # 计算 log(sum(exp(logits)))
        # 使用 logsumexp 防止数值不稳定
        log_sum_exp = torch.logsumexp(logits, dim=-1)  # [batch_size, seq_len]
        
        # Z-Loss = mean(log_sum_exp²)
        z_loss = torch.mean(log_sum_exp ** 2)
        
        return self.weight * z_loss

# 结合交叉熵损失
def train_with_z_loss(model, data_loader, z_loss_weight=1e-4):
    """使用 Z-Loss 训练"""
    ce_loss_fn = nn.CrossEntropyLoss()
    z_loss_fn = ZLoss(weight=z_loss_weight)
    
    for batch in data_loader:
        logits = model(batch["input_ids"])
        
        # 交叉熵损失
        ce_loss = ce_loss_fn(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1)
        )
        
        # Z-Loss
        z_loss = z_loss_fn(logits)
        
        # 总损失
        total_loss = ce_loss + z_loss
        
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 监控
        print(f"CE Loss: {ce_loss:.4f}, Z-Loss: {z_loss:.6f}")
```

**PaLM 实践**：

```python
# PaLM 的 Z-Loss 配置
PALM_Z_LOSS_CONFIG = {
    "weight": 1e-4,          # Z-Loss 权重
    "logit_clipping": 50.0,  # logits 裁剪阈值
    "use_with_gradient_clipping": True,
    "gradient_clipping_max_norm": 1.0,
}

class PaLMTrainingStabilizer:
    """PaLM 风格的训练稳定器"""
    
    def __init__(self):
        self.z_loss = ZLoss(weight=PALM_Z_LOSS_CONFIG["weight"])
        self.logit_clipping = PALM_Z_LOSS_CONFIG["logit_clipping"]
    
    def stabilize_training(self, logits, labels):
        """稳定训练"""
        # 1. 裁剪 logits
        logits = torch.clamp(logits, -self.logit_clipping, self.logit_clipping)
        
        # 2. 计算损失
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        z_loss = self.z_loss(logits)
        
        total_loss = ce_loss + z_loss
        
        # 3. 反向传播
        total_loss.backward()
        
        # 4. 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=PALM_Z_LOSS_CONFIG["gradient_clipping_max_norm"]
        )
        
        return total_loss, ce_loss, z_loss
```

### 10.3 学习率预热

**Warmup 策略**：

```python
class WarmupStrategies:
    """学习率预热策略"""
    
    @staticmethod
    def linear_warmup(step, warmup_steps, base_lr):
        """线性预热"""
        if step < warmup_steps:
            return base_lr * (step / warmup_steps)
        return base_lr
    
    @staticmethod
    def constant_warmup(step, warmup_steps, base_lr):
        """常数预热（保持低学习率）"""
        if step < warmup_steps:
            return base_lr * 0.1  # 10% 基础学习率
        return base_lr
    
    @staticmethod
    def inverse_sqrt_warmup(step, warmup_steps, base_lr):
        """逆平方根预热"""
        if step < warmup_steps:
            return base_lr * (step / warmup_steps)
        return base_lr * (warmup_steps / step) ** 0.5
    
    @staticmethod
    def cosine_warmup(step, warmup_steps, total_steps, base_lr, min_lr_ratio=0.1):
        """余弦预热 + 衰减"""
        if step < warmup_steps:
            # 线性预热
            return base_lr * (step / warmup_steps)
        else:
            # 余弦衰减
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
```

**预热步数选择**：

| 模型规模 | 推荐预热步数 | 说明 |
|---------|-------------|------|
| <1B | 500-1000 | 小型模型 |
| 1B-10B | 1000-2000 | 中型模型 |
| 10B-100B | 2000-5000 | 大型模型 |
| >100B | 5000-10000 | 超大模型 |

**最佳实践**：

```python
class OptimalWarmupScheduler:
    """最优预热调度器"""
    
    def __init__(self, model_size, total_steps):
        self.model_size = model_size  # 参数量（B）
        self.total_steps = total_steps
        
        # 根据模型大小自动选择预热步数
        if model_size < 1:
            self.warmup_steps = 500
        elif model_size < 10:
            self.warmup_steps = 1000
        elif model_size < 100:
            self.warmup_steps = 2000
        else:
            self.warmup_steps = 5000
        
        self.base_lr = self.calculate_optimal_lr(model_size)
    
    def calculate_optimal_lr(self, model_size):
        """Chinchilla 最优学习率"""
        # 经验公式：lr ∝ 1 / sqrt(model_size)
        return 3e-4 * (7 / model_size) ** 0.5
    
    def get_lr(self, step):
        """获取当前学习率"""
        return WarmupStrategies.cosine_warmup(
            step, self.warmup_steps, self.total_steps, self.base_lr
        )
```

### 10.4 训练崩溃预防

**损失突增检测**：

```python
class LossSpikeDetector:
    """损失突增检测器"""
    
    def __init__(self, window_size=100, threshold=3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.loss_history = []
        self.spike_detected = False
    
    def update(self, current_loss):
        """更新损失历史并检测突增"""
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) < self.window_size:
            return False
        
        # 保持窗口大小
        if len(self.loss_history) > self.window_size:
            self.loss_history = self.loss_history[-self.window_size:]
        
        # 计算统计量
        mean_loss = np.mean(self.loss_history[:-1])
        std_loss = np.std(self.loss_history[:-1])
        
        # 检测突增（当前损失 > mean + threshold * std）
        if current_loss > mean_loss + self.threshold * std_loss:
            self.spike_detected = True
            print(f"⚠️ Loss spike detected: {current_loss:.4f} > {mean_loss + self.threshold * std_loss:.4f}")
            return True
        
        return False
    
    def get_recovery_action(self):
        """获取恢复动作"""
        return {
            "skip_step": True,
            "reduce_lr": True,
            "lr_reduction_factor": 0.5,
            "reload_checkpoint": False,
        }
```

**自动恢复机制**：

```python
class AutoRecoverySystem:
    """自动恢复系统"""
    
    def __init__(self, model, optimizer, checkpoint_dir):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.loss_detector = LossSpikeDetector()
        self.consecutive_spikes = 0
        self.max_spikes = 3
    
    def on_training_step(self, step, loss):
        """处理训练步骤"""
        spike_detected = self.loss_detector.update(loss)
        
        if spike_detected:
            self.consecutive_spikes += 1
            
            if self.consecutive_spikes >= self.max_spikes:
                print("🚨 Multiple loss spikes detected, initiating recovery...")
                self.recover(step)
                self.consecutive_spikes = 0
        else:
            self.consecutive_spikes = 0
    
    def recover(self, current_step):
        """执行恢复"""
        # 1. 降低学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.5
        print(f"Reduced learning rate to {param_group['lr']}")
        
        # 2. 加载最近检查点
        latest_checkpoint = self.find_latest_checkpoint()
        if latest_checkpoint:
            self.load_checkpoint(latest_checkpoint)
            print(f"Loaded checkpoint: {latest_checkpoint}")
        
        # 3. 重置优化器状态
        self.optimizer.state = defaultdict(dict)
    
    def find_latest_checkpoint(self):
        """找到最新的检查点"""
        import glob
        checkpoints = glob.glob(f"{self.checkpoint_dir}/step_*.pt")
        if not checkpoints:
            return None
        return max(checkpoints)
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
```

**检查点策略**：

```python
class CheckpointStrategy:
    """检查点策略"""
    
    def __init__(self, save_interval=1000, keep_last_n=5):
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        self.checkpoints = []
    
    def should_save(self, step):
        """判断是否应该保存"""
        return step % self.save_interval == 0
    
    def save(self, step, model, optimizer, scheduler):
        """保存检查点"""
        checkpoint = {
            'step': step,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'rng_state': torch.get_rng_state(),
        }
        
        path = f"checkpoints/step_{step}.pt"
        torch.save(checkpoint, path)
        
        self.checkpoints.append(path)
        
        # 清理旧检查点
        if len(self.checkpoints) > self.keep_last_n:
            old_checkpoint = self.checkpoints.pop(0)
            os.remove(old_checkpoint)
            print(f"Removed old checkpoint: {old_checkpoint}")
        
        print(f"Saved checkpoint: {path}")
```

**监控告警**：

```python
class TrainingMonitor:
    """训练监控与告警"""
    
    def __init__(self, config):
        self.config = config
        self.metrics = {
            "loss": [],
            "grad_norm": [],
            "learning_rate": [],
            "gpu_memory": [],
        }
        self.alerts = []
    
    def update_metrics(self, step, loss, grad_norm, lr, gpu_memory):
        """更新指标"""
        self.metrics["loss"].append(loss)
        self.metrics["grad_norm"].append(grad_norm)
        self.metrics["learning_rate"].append(lr)
        self.metrics["gpu_memory"].append(gpu_memory)
        
        # 检查告警条件
        self.check_alerts(step, loss, grad_norm, gpu_memory)
    
    def check_alerts(self, step, loss, grad_norm, gpu_memory):
        """检查告警条件"""
        # 损失为 NaN
        if math.isnan(loss):
            self.send_alert(step, "CRITICAL", "Loss is NaN!")
        
        # 梯度爆炸
        if grad_norm > 10.0:
            self.send_alert(step, "WARNING", f"Large gradient norm: {grad_norm:.2f}")
        
        # 显存不足
        if gpu_memory > 0.9 * torch.cuda.get_device_properties(0).total_memory:
            self.send_alert(step, "WARNING", f"GPU memory usage: {gpu_memory/1e9:.1f} GB")
    
    def send_alert(self, step, level, message):
        """发送告警"""
        alert = {
            "step": step,
            "level": level,
            "message": message,
            "timestamp": time.time()
        }
        self.alerts.append(alert)
        
        print(f"[{level}] Step {step}: {message}")
        
        # 可集成邮件、Slack、企业微信等通知
```

### 10.5 分布式训练稳定性

**同步策略**：

```python
class DistributedSyncStrategy:
    """分布式同步策略"""
    
    def __init__(self):
        self.sync_interval = 1  # 每步同步
    
    def sync_gradients(self, model):
        """同步梯度"""
        # 使用 DDP 自动同步
        # 或手动 All-Reduce
        
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    
    def sync_batch_norm(self, model):
        """同步 BatchNorm 统计量"""
        # 使用 SyncBatchNorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    def check_sync_health(self, model):
        """检查同步健康状态"""
        # 在不同 rank 上比较梯度
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.mean().item())
        
        # 收集所有 rank 的梯度
        all_gradients = [None] * dist.get_world_size()
        dist.all_gather_object(all_gradients, gradients)
        
        # 检查差异
        for rank_r, grads_r in enumerate(all_gradients):
            for rank_g, grads_g in enumerate(all_gradients):
                if rank_r != rank_g:
                    diff = np.mean(np.abs(np.array(grads_r) - np.array(grads_g)))
                    if diff > 1e-6:
                        print(f"⚠️ Gradient mismatch between rank {rank_r} and {rank_g}: {diff}")
```

**通信优化**：

```python
def optimize_communication():
    """优化分布式通信"""
    
    # 1. 梯度压缩
    from torch.distributed.algorithms import gradient_compression
    
    # 2. 重叠通信和计算
    torch.cuda.set_stream(
        torch.cuda.Stream()
    )
    
    # 3. 使用 NCCL 后端（推荐）
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    
    # 4. 调整 NCCL 环境变量
    os.environ["NCCL_IB_DISABLE"] = "0"  # 启用 InfiniBand
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    os.environ["NCCL_DEBUG"] = "INFO"
    
    # 5. 使用 GradScaler 优化
    scaler = GradScaler()
```

**故障恢复**：

```python
class FaultToleranceSystem:
    """容错系统"""
    
    def __init__(self, model, checkpoint_dir):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.max_retries = 3
    
    def train_with_fault_tolerance(self, train_fn, max_steps):
        """带容错的训练"""
        retries = 0
        start_step = self.load_latest_checkpoint()
        
        while start_step < max_steps and retries < self.max_retries:
            try:
                train_fn(start_step, max_steps)
                break
            except Exception as e:
                retries += 1
                print(f"Training failed (attempt {retries}/{self.max_retries}): {e}")
                
                if retries < self.max_retries:
                    print("Reloading checkpoint and retrying...")
                    start_step = self.load_latest_checkpoint()
                    time.sleep(10)  # 等待 GPU 恢复
                else:
                    print("Max retries reached, aborting.")
                    raise
    
    def load_latest_checkpoint(self):
        """加载最新检查点"""
        checkpoints = glob.glob(f"{self.checkpoint_dir}/step_*.pt")
        if not checkpoints:
            return 0
        
        latest = max(checkpoints)
        checkpoint = torch.load(latest)
        
        self.model.load_state_dict(checkpoint['model_state'])
        
        return checkpoint['step']
```

**容错机制**：

```python
class ElasticTraining:
    """弹性训练（自动调整 worker 数量）"""
    
    def __init__(self):
        self.min_workers = 4
        self.max_workers = 64
        self.current_workers = None
    
    def monitor_cluster_health(self):
        """监控集群健康状态"""
        # 检查可用 GPU
        available_gpus = torch.cuda.device_count()
        
        # 检查失败的 worker
        failed_workers = self.detect_failed_workers()
        
        # 调整 worker 数量
        if len(failed_workers) > 0:
            print(f"Detected {len(failed_workers)} failed workers")
            self.remove_workers(failed_workers)
            self.rebalance_training()
    
    def detect_failed_workers(self):
        """检测失败的 worker"""
        failed = []
        for rank in range(dist.get_world_size()):
            try:
                # 发送心跳
                dist.barrier(device_ids=[rank], timeout=10)
            except Exception:
                failed.append(rank)
        return failed
    
    def rebalance_training(self):
        """重新平衡训练"""
        # 重新初始化进程组
        dist.destroy_process_group()
        
        # 使用剩余 worker 重新初始化
        new_world_size = dist.get_world_size() - len(self.failed_workers)
        dist.init_process_group(
            backend="nccl",
            world_size=new_world_size,
            rank=0,
        )
        
        print(f"Rebalanced training with {new_world_size} workers")
```

---

> **总结**：本附录详细介绍了词元化技术、混合精度训练和训练稳定性技巧，这些都是 LLM 预训练中的关键技术。掌握这些技术可以帮助你更高效、更稳定地训练大规模语言模型。
