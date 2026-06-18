# 微调技术(Fine-tuning)

> 📅 **更新时间**: 2026-06-17

---

## 目录

- [1. 微调技术概览](#1-微调技术概览)
- [2. 全量微调（Full Fine-tuning）](#2-全量微调full-fine-tuning)
- [3. 参数高效微调（PEFT）](#3-参数高效微调peft)
- [4. LoRA 技术详解](#4-lora-技术详解)
- [5. QLoRA 技术详解](#5-qlora-技术详解)
- [6. 指令微调（Instruction Tuning）](#6-指令微调instruction-tuning)
- [7. 多模态微调技术](#7-多模态微调技术)
- [8. 实战：完整微调工作流](#8-实战完整微调工作流)
- [9. 最佳实践](#9-最佳实践)
- [10. 参考资源](#10-参考资源)
- [11. Prompt Tuning 系列](#11-prompt-tuning-系列)

---

## 1. 微调技术概览

### 1.1 为什么需要微调？

预训练模型虽然具备强大的语言理解能力，但在特定任务上仍需微调：

- **领域适配**：让模型掌握特定领域的专业知识
- **任务对齐**：调整模型行为以完成特定任务（如对话、代码生成）
- **风格定制**：控制模型输出的语气、格式和风格
- **性能提升**：在特定 benchmark 上获得更好的表现

### 1.2 微调方法分类

```
微调技术
├── 全量微调（Full Fine-tuning）
│   └── 更新所有模型参数
│
└── 参数高效微调（PEFT）
    ├── LoRA 系列（LoRA, QLoRA, LoRA+）
    ├── Prompt Tuning（Prefix Tuning, P-Tuning）
    ├── Adapter 系列（Adapter, Compacter）
    └── 其他（IA3, AdaLoRA, DoRA）
```

### 1.3 方法选择决策树

```python
def choose_finetuning_method(model_size, gpu_memory, target_performance):
    """
    根据资源限制选择微调方法
    """
    if model_size <= 3e9:  # 3B 以下
        if gpu_memory >= 40:  # 40GB+
            return "full_finetuning"
        else:
            return "lora"
    
    elif model_size <= 13e9:  # 13B
        if gpu_memory >= 80:  # A100 80GB
            return "full_finetuning"
        elif gpu_memory >= 24:  # RTX 3090/4090
            return "qlora"
        else:
            return "qlora_4bit"
    
    elif model_size <= 70e9:  # 70B
        if gpu_memory >= 80 * 8:  # 多卡 A100
            return "full_finetuning"
        else:
            return "qlora"
    
    else:  # 超大模型
        return "qlora_4bit"

# 2025 年实际案例对比
resource_comparison = {
    "Full Fine-tuning 7B": {
        "vram": "100-120 GB",
        "cost": "~$50K (H100 集群)",
        "performance": "最佳"
    },
    "QLoRA 7B": {
        "vram": "24 GB (单张 RTX 4090)",
        "cost": "~$1,500",
        "performance": "接近全量微调（~95%）"
    },
    "LoRA 7B": {
        "vram": "40-48 GB",
        "cost": "~$5K",
        "performance": "接近全量微调（~97%）"
    }
}
```

## 2. 全量微调（Full Fine-tuning）

### 2.1 基本原理

全量微调更新模型的所有参数：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

def full_finetune_example():
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")
    
    # 添加 pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # 准备数据集
    train_dataset = load_custom_dataset("data/train.json")
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        report_to="wandb"
    )
    
    # 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    trainer.train()
    model.save_pretrained("./fine-tuned-model")
```

### 2.2 全量微调的挑战

**显存需求计算**：

```python
def calculate_vram_requirement(model_params, batch_size, seq_length):
    """
    计算全量微调所需显存
    """
    # 参数存储（BF16）
    param_memory = model_params * 2  # 2 bytes per param (BF16)
    
    # 优化器状态（AdamW）
    # 每个参数需要：梯度 (2B) + momentum (4B) + variance (4B) = 10B
    optimizer_memory = model_params * 10
    
    # 激活值（与前向传播相关）
    # 简化估算：每层约 batch_size * seq_length * hidden_size * 4
    activation_memory = batch_size * seq_length * 4096 * 4 * 32  # 32 layers
    
    # 总显存（GB）
    total_memory_gb = (param_memory + optimizer_memory + activation_memory) / (1024**3)
    
    return total_memory_gb

# 7B 模型示例
model_params = 7e9
batch_size = 4
seq_length = 2048

vram_needed = calculate_vram_requirement(model_params, batch_size, seq_length)
print(f"需要显存: {vram_needed:.2f} GB")  # 约 120 GB
```

### 2.3 全量微调最佳实践

```python
class FullFineTuningBestPractices:
    """2025 年全量微调最佳实践"""
    
    @staticmethod
    def setup_model(model_name):
        """模型加载优化"""
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # 使用 BF16
            device_map="auto",  # 自动分配到多卡
            attn_implementation="flash_attention_2"  # 使用 Flash Attention
        )
        return model
    
    @staticmethod
    def setup_training_args():
        """训练参数配置"""
        return TrainingArguments(
            # 基础配置
            output_dir="./output",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            
            # 学习率
            learning_rate=2e-5,  # 小学习率防止灾难性遗忘
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            
            # 精度优化
            bf16=True,
            tf32=True,
            
            # 显存优化
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            
            # 正则化
            weight_decay=0.01,
            max_grad_norm=1.0,
            
            # 日志和保存
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            save_strategy="steps",
            report_to="wandb"
        )
    
    @staticmethod
    def prevent_catastrophic_forgetting(model, original_data, target_data):
        """
        防止灾难性遗忘的策略
        """
        # 策略 1: 混合训练数据
        mixed_data = original_data.sample(frac=0.1) + target_data
        
        # 策略 2: 使用较小的学习率
        # 通常在 1e-6 到 5e-5 之间
        
        # 策略 3: 正则化技术（EWC, L2-SP）
        # 在 loss 中添加正则化项
        pass
```

## 3. 参数高效微调（PEFT）

### 3.1 PEFT 核心思想

PEFT 的核心是**冻结预训练模型参数，只训练少量新增参数**：

```python
from peft import get_peft_model, prepare_model_for_kbit_training

# PEFT 通用流程
def peft_workflow(model, peft_config):
    # 1. 准备模型（可选：量化）
    model = prepare_model_for_kbit_training(model)
    
    # 2. 应用 PEFT 方法
    model = get_peft_model(model, peft_config)
    
    # 3. 查看可训练参数
    model.print_trainable_parameters()
    # 输出: trainable params: 0.1% || all params: 100%
    
    # 4. 训练（与全量微调相同）
    trainer = Trainer(model=model, ...)
    trainer.train()
    
    return model
```

### 3.2 PEFT 方法对比

| 方法 | 参数量 | 显存需求 | 性能 | 适用场景 |
|------|--------|----------|------|----------|
| Full FT | 100% | 高 | 最佳 | 资源充足 |
| LoRA | 0.1-1% | 低 | 接近全量 | **通用推荐** |
| QLoRA | 0.1-1% | 极低 | 略低于 LoRA | **资源受限** |
| Prefix Tuning | 0.01-0.1% | 极低 | 中等 | 快速原型 |
| Adapter | 1-3% | 中 | 良好 | 多任务学习 |

## 4. LoRA 技术详解

### 4.1 LoRA 原理

LoRA（Low-Rank Adaptation）假设模型更新具有**低秩特性**：

```
ΔW = B × A

其中:
- W ∈ R^{d×k} 是预训练权重（冻结）
- B ∈ R^{d×r} 是低秩矩阵（可训练）
- A ∈ R^{r×k} 是低秩矩阵（可训练）
- r << d, k（通常 r=8, 16, 32）

前向传播:
h = Wx + ΔWx = Wx + BAx
```

```python
import torch.nn as nn

class LoRALinear(nn.Module):
    """LoRA 线性层实现"""
    
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        
        # 冻结的预训练权重
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        
        # LoRA 参数（可训练）
        self.rank = rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # 缩放因子
        self.alpha = alpha
        self.scaling = alpha / rank
    
    def forward(self, x):
        # 原始输出
        original_output = nn.functional.linear(x, self.weight)
        
        # LoRA 更新
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return original_output + lora_output

# 数学直觉
def lora_intuition():
    """
    为什么 LoRA 有效？
    
    1. 低秩假设：模型在特定任务上的调整可以用低维子空间表示
    2. 参数共享：LoRA 矩阵在不同输入间共享
    3. 无推理开销：训练后可以合并权重
    4. 模块化：可以为不同任务训练不同的 LoRA 适配器
    """
    pass
```

### 4.2 LoRA 实战配置

```python
from peft import LoraConfig, TaskType

# 标准 LoRA 配置（2025 年推荐）
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    
    # 核心参数
    r=16,              # LoRA 秩（8-32 之间）
    lora_alpha=32,     # 缩放因子（通常是 r 的 2 倍）
    lora_dropout=0.05, # 防止过拟合
    
    # 目标模块
    target_modules=[
        "q_proj",       # Query 投影
        "k_proj",       # Key 投影
        "v_proj",       # Value 投影
        "o_proj",       # Output 投影
        "gate_proj",    # FFN gate
        "up_proj",      # FFN up
        "down_proj"     # FFN down
    ],
    
    # 其他设置
    bias="none",       # 不训练 bias
    use_rslora=False,  # 2025 年新选项：RSLoRA
    use_dora=False     # 2025 年新选项：DoRA
)

def train_with_lora():
    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3-8b",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 应用 LoRA
    from peft import get_peft_model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 输出: trainable params: 0.14% || all params: 100%
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir="./lora-output",
        num_train_epochs=3,
        per_device_train_batch_size=8,  # LoRA 可以用更大的 batch size
        gradient_accumulation_steps=4,
        learning_rate=2e-4,  # LoRA 通常用更大的学习率
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=200,
        report_to="wandb"
    )
    
    # 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    trainer.train()
    
    # 保存 LoRA 权重（只有几 MB）
    model.save_pretrained("./lora-adapter")
```

### 4.3 LoRA 秩的选择

```python
def analyze_lora_rank():
    """
    LoRA 秩（r）的影响：
    
    r=4:   极快训练，可能欠拟合，适合简单任务
    r=8:   快速训练，良好平衡，推荐起点
    r=16:  标准配置，适合大多数任务
    r=32:  更好的性能，更多显存，复杂任务
    r=64+: 接近全量微调，但训练慢
    
    经验法则：
    - 从 r=16 开始
    - 如果欠拟合，增加到 32
    - 如果过拟合，减少到 8
    - alpha 通常设置为 2*r
    """
    
    rank_performance = {
        4: {"params": "0.04%", "quality": "中等", "speed": "极快"},
        8: {"params": "0.07%", "quality": "良好", "speed": "快速"},
        16: {"params": "0.14%", "quality": "优秀", "speed": "标准"},
        32: {"params": "0.28%", "quality": "最佳", "speed": "较慢"},
    }
    
    return rank_performance

# 2025 年最新发现：RSLoRA
# RSLoRA（Rank-Stabilized LoRA）改进了缩放策略
lora_config_rslora = LoraConfig(
    r=16,
    lora_alpha=32,
    use_rslora=True,  # 启用 RSLoRA
    # RSLoRA 使用 sqrt(r) 而不是 r 进行缩放
    # 使得不同秩的配置更加稳定
)
```

### 4.4 LoRA 目标模块选择

```python
def choose_target_modules(task_type):
    """
    根据任务类型选择 LoRA 目标模块
    """
    if task_type == "conversation":
        # 对话任务：只训练注意力层
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    elif task_type == "code_generation":
        # 代码生成：训练所有层
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
    
    elif task_type == "reasoning":
        # 推理任务：重点训练 FFN
        return ["gate_proj", "up_proj", "down_proj", "o_proj"]
    
    elif task_type == "knowledge_injection":
        # 知识注入：训练所有层
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "embed_tokens", "lm_head"]
    
    else:
        # 默认：训练注意力层
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

# 2025 年最佳实践
# 训练更多模块 = 更好性能 = 更多显存
# 推荐从注意力层开始，根据需要扩展
```

### 4.5 LoRA 权重合并

```python
from peft import PeftModel

def merge_lora_weights():
    """
    训练完成后合并 LoRA 权重
    合并后模型与原始模型完全相同，无额外推理开销
    """
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b")
    
    # 加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, "./lora-adapter")
    
    # 合并权重
    model = model.merge_and_unload()
    
    # 保存合并后的模型
    model.save_pretrained("./merged-model")
    tokenizer.save_pretrained("./merged-model")

# 合并的数学过程
def explain_merge():
    """
    W_merged = W_original + (B @ A) * scaling
    
    合并后:
    - 模型大小不变
    - 推理速度不变
    - 但包含了 LoRA 的学习效果
    """
    pass
```

## 5. QLoRA 技术详解

### 5.1 QLoRA 核心创新

QLoRA（Quantized LoRA）结合**量化**和**LoRA**，实现单卡微调大模型：

```
QLoRA = 4-bit 量化 + LoRA + 优化技术

核心思想:
1. 将基础模型量化到 4-bit（NF4）
2. 冻结量化后的权重
3. 添加 LoRA 适配器进行训练
4. 使用分页优化器处理显存峰值
```

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

def qlora_setup():
    """QLoRA 配置（2025 年标准）"""
    
    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        
        # NF4（Normal Float 4-bit）量化
        bnb_4bit_quant_type="nf4",
        
        # 使用 BF16 计算
        bnb_4bit_compute_dtype=torch.bfloat16,
        
        # 双重量化（节省更多显存）
        bnb_4bit_use_double_quant=True
    )
    
    # 加载量化模型
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3-8b",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # 准备模型用于 k-bit 训练
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    
    return model

# 显存对比
def compare_memory_usage():
    comparison = {
        "7B FP16": "~14 GB (仅加载)",
        "7B BF16": "~14 GB (仅加载)",
        "7B 4-bit (NF4)": "~3.5 GB (仅加载)",
        "7B Full FT": "~120 GB (训练)",
        "7B LoRA": "~24 GB (训练)",
        "7B QLoRA": "~8 GB (训练) ← 单卡 RTX 3090 即可"
    }
    return comparison
```

### 5.2 NF4 量化原理

```python
class NF4Quantization:
    """
    NF4（Normal Float 4-bit）量化
    
    核心思想：
    - 假设权重服从正态分布
    - 设计非均匀量化级别，使信息损失最小
    - 4-bit = 16 个量化级别
    """
    
    def __init__(self):
        # NF4 量化表（基于正态分布的分位数）
        self.nf4_quantization_levels = [
            -1.0, -0.6962, -0.5250, -0.3900, -0.2721, -0.1625,
            -0.0577, 0.0000, 0.0577, 0.1625, 0.2721, 0.3900,
            0.5250, 0.6962, 1.0
        ]
    
    def quantize(self, weight):
        """
        量化过程:
        1. 计算权重的绝对最大值
        2. 归一化到 [-1, 1]
        3. 映射到最近的 NF4 级别
        4. 存储 4-bit 索引
        """
        abs_max = torch.max(torch.abs(weight))
        normalized = weight / abs_max
        
        # 量化到 NF4
        quantized = self._map_to_nf4(normalized)
        
        return quantized, abs_max
    
    def dequantize(self, quantized, abs_max):
        """反量化（用于计算）"""
        # 将 4-bit 索引映射回浮点数
        float_values = self._index_to_float(quantized)
        return float_values * abs_max

# 双重量化
def double_quantization(weight):
    """
    双重量化：对量化常量也进行量化
    进一步节省显存
    """
    # 第一次量化
    q1, scale1 = quantize(weight)
    
    # 第二次量化（对 scale1 也量化）
    q2, scale2 = quantize(scale1)
    
    return q2, scale2, scale1
```

### 5.3 QLoRA 完整训练流程

```python
"""
QLoRA 完整训练示例
单卡 RTX 4090 (24GB) 微调 7B 模型
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset

def train_qlora():
    # 1. 配置量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    # 2. 加载模型
    model_name = "meta-llama/Llama-3-8b"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. 准备 k-bit 训练
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    
    # 4. 配置 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )
    
    # 5. 应用 LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # 输出: trainable params: 0.07% || all params: 100%
    
    # 6. 加载数据集
    dataset = load_dataset("json", data_files="train.json", split="train")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding=False
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 7. 训练配置
    training_args = TrainingArguments(
        output_dir="./qlora-output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,  # 使用 BF16 计算
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        report_to="wandb",
        optim="paged_adamw_8bit"  # 分页优化器
    )
    
    # 8. 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    trainer.train()
    
    # 9. 保存
    model.save_pretrained("./qlora-adapter")
    tokenizer.save_pretrained("./qlora-adapter")
    
    return model

# 推理使用
def inference_with_qlora():
    from peft import PeftModel
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3-8b",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # 加载 QLoRA 适配器
    model = PeftModel.from_pretrained(base_model, "./qlora-adapter")
    
    # 推理
    prompt = "请解释量子计算的基本原理"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 5.4 QLoRA 进阶优化（2025）

```python
# 2025 年 QLoRA 优化技术

# 1. 使用 Liger Kernels 加速训练
from liger_kernel.transformers import apply_liger_kernel_to_llama

apply_liger_kernel_to_llama()  # 自动优化 Llama 模型

# 2. 使用 Flash Attention
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2"  # 启用 Flash Attention
)

# 3. 使用 Spectrum（2025 年新技术）
# Spectrum 选择性地微调更多层
from peft import SpectrumConfig

spectrum_config = SpectrumConfig(
    r=16,
    layer_selection="stratified",  # 分层选择
    num_layers_to_tune=8  # 选择 8 层进行全量微调
)

# 4. 使用 LoRA+（改进的初始化）
from peft import LoraConfig

lora_plus_config = LoraConfig(
    r=16,
    lora_alpha=32,
    use_lora_plus=True,  # 2025 年新特性
    lora_plus_rank_alpha=2.0  # 对 A 和 B 矩阵使用不同的学习率
)
```

## 6. 指令微调（Instruction Tuning）

### 6.1 什么是指令微调？

指令微调让模型学会**遵循指令**，而不仅仅是完成语言建模：

```python
# 预训练数据 vs 指令微调数据

# 预训练数据（纯文本）
pretrain_example = """
Transformer 是一种深度学习架构，由 Vaswani 等人在 2017 年提出...
"""

# 指令微调数据（指令-响应对）
instruction_example = {
    "instruction": "解释 Transformer 的核心机制",
    "input": "",  # 可选的额外输入
    "output": "Transformer 的核心机制是自注意力（Self-Attention）..."
}

# 对话格式（2025 年标准）
conversation_example = [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是人工智能的一个分支..."}
]
```

### 6.2 指令数据格式

```python
from datasets import Dataset

def create_instruction_dataset():
    """创建指令微调数据集"""
    
    # Alpaca 格式
    alpaca_format = [
        {
            "instruction": "翻译以下句子为英文",
            "input": "今天天气很好",
            "output": "The weather is very nice today"
        }
    ]
    
    # ShareGPT 格式（对话）
    sharegpt_format = [
        {
            "conversations": [
                {"from": "human", "value": "如何学习编程？"},
                {"from": "gpt", "value": "学习编程需要以下步骤：..."}
            ]
        }
    ]
    
    # 2025 年推荐格式（ChatML）
    chatml_format = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "解释 Python 装饰器"},
                {"role": "assistant", "content": "Python 装饰器是一种..."}
            ]
        }
    ]
    
    return Dataset.from_list(chatml_format)

# 数据格式化函数
def format_chatml(messages, tokenizer):
    """
    将 ChatML 格式转换为模型输入
    """
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return text

# 示例输出（Llama-3 格式）
example_output = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
解释 Python 装饰器<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Python 装饰器是一种...<|eot_id|>
"""
```

### 6.3 指令微调数据处理

```python
from datasets import load_dataset, Dataset
import pandas as pd

class InstructionDataProcessor:
    """指令微调数据处理"""
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_multiple_sources(self, sources):
        """加载多个数据源"""
        datasets = []
        
        for source in sources:
            if source.endswith('.json'):
                data = load_dataset('json', data_files=source)['train']
            elif source.endswith('.csv'):
                data = Dataset.from_pandas(pd.read_csv(source))
            else:
                data = load_dataset(source)['train']
            
            datasets.append(data)
        
        # 合并数据集
        combined = concatenate_datasets(datasets)
        return combined
    
    def tokenize_and_format(self, dataset):
        """
        Tokenize 并格式化数据
        关键：只计算 response 部分的 loss
        """
        def process(examples):
            # 应用聊天模板
            texts = [
                self.tokenizer.apply_chat_template(messages, tokenize=False)
                for messages in examples['messages']
            ]
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            
            # 创建 labels（将 prompt 部分设为 -100）
            labels = self._create_labels(tokenized, texts)
            tokenized['labels'] = labels
            
            return tokenized
        
        return dataset.map(
            process,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    def _create_labels(self, tokenized, texts):
        """
        创建 labels：只计算 response 的 loss
        """
        labels_list = []
        
        for input_ids, text in zip(tokenized['input_ids'], texts):
            # 找到 assistant 回复的起始位置
            assistant_pattern = "<|start_header_id|>assistant<|end_header_id|>"
            assistant_start = text.find(assistant_pattern)
            
            # Tokenize prompt 以确定分割点
            prompt = text[:assistant_start + len(assistant_pattern)]
            prompt_length = len(self.tokenizer.encode(prompt))
            
            # 创建 labels
            labels = input_ids.copy()
            labels[:prompt_length] = [-100] * prompt_length
            
            labels_list.append(labels)
        
        return labels_list

# 使用示例
def prepare_instruction_data():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")
    tokenizer.pad_token = tokenizer.eos_token
    
    processor = InstructionDataProcessor(tokenizer, max_length=2048)
    
    # 加载数据
    dataset = processor.load_multiple_sources([
        "data/alpaca.json",
        "data/sharegpt.json",
        "data/custom.json"
    ])
    
    # 处理
    processed_dataset = processor.tokenize_and_format(dataset)
    
    return processed_dataset
```

### 6.4 指令微调训练

```python
def train_instruction_tuning():
    """指令微调完整流程"""
    
    # 1. 准备数据
    dataset = prepare_instruction_data()
    
    # 2. 加载模型（使用 QLoRA）
    model = setup_qlora_model("meta-llama/Llama-3-8b")
    
    # 3. 训练配置
    training_args = TrainingArguments(
        output_dir="./instruction-tuning",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        report_to="wandb",
        optim="paged_adamw_8bit",
        
        # 指令微调特殊配置
        remove_unused_columns=False,  # 保留所有列
        label_names=["labels"]  # 指定 label 名称
    )
    
    # 4. 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    trainer.train()
    model.save_pretrained("./instruction-tuned-adapter")
```

### 6.5 指令数据质量（2025 年关键发现）

```python
class DataQualityGuidelines:
    """
    2025 年指令数据质量最佳实践
    
    关键发现：
    1. 质量 >> 数量
    2. 多样性很重要
    3. 避免低质量数据
    """
    
    @staticmethod
    def quality_over_quantity():
        """
        数据质量优先级：
        
        10,000 条高质量数据 > 100,000 条低质量数据
        
        高质量数据特征：
        - 准确的回答
        - 详细的解释
        - 逻辑清晰
        - 无事实错误
        - 语言自然流畅
        """
        pass
    
    @staticmethod
    def ensure_diversity():
        """
        数据多样性：
        
        涵盖多个领域：
        - 问答（30%）
        - 代码（20%）
        - 创意写作（15%）
        - 逻辑推理（15%）
        - 数学（10%）
        - 其他（10%）
        """
        diversity_targets = {
            "qa": 0.30,
            "code": 0.20,
            "creative_writing": 0.15,
            "reasoning": 0.15,
            "math": 0.10,
            "other": 0.10
        }
        return diversity_targets
    
    @staticmethod
    def filter_low_quality():
        """过滤低质量数据"""
        filters = [
            "长度过短（< 50 tokens）",
            "包含敏感信息",
            "重复内容",
            "语法错误",
            "事实错误",
            "自相矛盾"
        ]
        return filters

# 自动质量评分
class DataQualityScorer:
    """使用模型自动评分数据质量"""
    
    def __init__(self):
        self.scorer = AutoModelForSequenceClassification.from_pretrained(
            "quality-scorer-v2"
        )
    
    def score(self, instruction, response):
        """为数据样本评分"""
        # 多维度评分
        scores = {
            "relevance": self.score_relevance(instruction, response),
            "fluency": self.score_fluency(response),
            "informativeness": self.score_informativeness(response),
            "factual_accuracy": self.score_factual_accuracy(response)
        }
        
        overall_score = sum(scores.values()) / len(scores)
        return overall_score, scores
    
    def filter_dataset(self, dataset, threshold=0.7):
        """过滤低质量数据"""
        filtered_data = []
        
        for sample in dataset:
            score, _ = self.score(sample['instruction'], sample['output'])
            if score >= threshold:
                filtered_data.append(sample)
        
        return filtered_data
```

## 7. 多模态微调技术

### 7.1 多模态模型架构

```python
class VisionLanguageModel(nn.Module):
    """视觉语言模型（2025 年架构）"""
    
    def __init__(self, vision_model, language_model, projector):
        super().__init__()
        
        # 视觉编码器（如 CLIP、SigLIP）
        self.vision_encoder = vision_model
        
        # 投影层（视觉特征 → 文本空间）
        self.projector = projector
        
        # 语言模型（LLM）
        self.language_model = language_model
    
    def forward(self, images, input_ids, attention_mask, labels):
        # 1. 编码图像
        image_features = self.vision_encoder(images)
        
        # 2. 投影到文本空间
        image_tokens = self.projector(image_features)
        
        # 3. 与文本 token 拼接
        embeddings = self.language_model.get_input_embeddings()(input_ids)
        combined_embeddings = self.combine_image_text(
            embeddings, image_tokens
        )
        
        # 4. 语言模型处理
        outputs = self.language_model(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs

# 主流多模态模型（2025-2026）
multimodal_models_2025 = {
    "LLaVA-NeXT": "高质量视觉理解",
    "Qwen2-VL": "支持视频理解",
    "Phi-3-Vision": "轻量级，高效",
    "InternVL2": "强大的多语言支持",
    "Molmo": "Allen AI 出品，开源"
}
```

### 7.2 多模态数据格式

```python
# 多模态指令数据格式
multimodal_example = {
    "images": ["path/to/image1.jpg"],
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image1.jpg"},
                {"type": "text", "text": "这张图片中有什么？"}
            ]
        },
        {
            "role": "assistant",
            "content": "这是一张美丽的风景照，展示了..."
        }
    ]
}

# 多模态数据处理
class MultimodalDataProcessor:
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
    
    def process(self, example):
        # 处理图像
        images = self.load_and_process_images(example['images'])
        
        # 处理文本
        text = self.tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False
        )
        
        # Tokenize 文本
        inputs = self.tokenizer(text, return_tensors="pt")
        
        return {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "pixel_values": images,
            "labels": inputs['input_ids'].clone()
        }
    
    def load_and_process_images(self, image_paths):
        from PIL import Image
        
        images = []
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            images.append(image)
        
        # 预处理
        processed = self.image_processor(images, return_tensors="pt")
        return processed['pixel_values']
```

### 7.3 多模态微调示例

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def finetune_multimodal_model():
    """多模态模型微调"""
    
    # 1. 加载模型和处理器
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    processor = LlavaNextProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 2. 应用 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    
    # 3. 准备数据集
    dataset = load_multimodal_dataset("data/multimodal_train.json")
    
    # 4. 训练配置
    training_args = TrainingArguments(
        output_dir="./multimodal-finetune",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False
    )
    
    # 5. 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=multimodal_collator
    )
    
    trainer.train()
    model.save_pretrained("./multimodal-adapter")

# 多模态推理
def multimodal_inference():
    processor = LlavaNextProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "./multimodal-adapter",
        torch_dtype=torch.bfloat16
    )
    
    # 加载图像
    image = Image.open("test.jpg")
    
    # 准备输入
    prompt = "描述这张图片"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # 生成
    outputs = model.generate(**inputs, max_length=512)
    print(processor.decode(outputs[0], skip_special_tokens=True))
```

## 8. 实战：完整微调工作流

### 8.1 端到端微调管道

```python
"""
完整微调工作流（2025 年最佳实践）
"""

import os
from dataclasses import dataclass
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

@dataclass
class FinetuningConfig:
    """微调配置"""
    # 模型
    model_name: str = "meta-llama/Llama-3-8b"
    
    # 方法选择
    method: str = "qlora"  # full, lora, qlora
    
    # 数据
    train_data: str = "data/train.json"
    max_length: int = 2048
    
    # 训练
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    
    # 输出
    output_dir: str = "./output"
    save_steps: int = 200
    logging_steps: int = 10

class FinetuningPipeline:
    """端到端微调管道"""
    
    def __init__(self, config: FinetuningConfig):
        self.config = config
        self.setup()
    
    def setup(self):
        """初始化"""
        print("1. 初始化组件...")
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()
        self.dataset = self.prepare_dataset()
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def load_model(self):
        print(f"   加载模型: {self.config.model_name}")
        print(f"   方法: {self.config.method}")
        
        if self.config.method == "qlora":
            return self.load_qlora_model()
        elif self.config.method == "lora":
            return self.load_lora_model()
        else:
            return self.load_full_model()
    
    def load_qlora_model(self):
        """加载 QLoRA 模型"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none"
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model
    
    def load_lora_model(self):
        """加载 LoRA 模型"""
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model
    
    def load_full_model(self):
        """加载全量微调模型"""
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        return model
    
    def prepare_dataset(self):
        """准备数据集"""
        print("2. 准备数据集...")
        
        dataset = load_dataset('json', data_files=self.config.train_data)['train']
        
        def tokenize(examples):
            texts = examples['text']
            return self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_length,
                padding=False
            )
        
        tokenized = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized
    
    def train(self):
        """训练"""
        print("3. 开始训练...")
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=self.config.warmup_ratio,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            report_to="wandb",
            optim="paged_adamw_8bit" if self.config.method == "qlora" else "adamw_torch"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )
        
        trainer.train()
        
        print("4. 保存模型...")
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        print("✓ 训练完成!")

# 运行
if __name__ == "__main__":
    config = FinetuningConfig(
        model_name="meta-llama/Llama-3-8b",
        method="qlora",
        train_data="data/train.json",
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4,
        output_dir="./qlora-finetuned"
    )
    
    pipeline = FinetuningPipeline(config)
    pipeline.train()
```

## 9. 最佳实践

### 9.1 方法选择指南

```yaml
# 根据场景选择微调方法

场景 1: 资源充足（多张 A100）
  方法: 全量微调
  学习率: 2e-5
  Epochs: 1-3
  Batch Size: 32-64

场景 2: 单张 A100 (40GB)
  方法: LoRA
  秩: 16
  学习率: 2e-4
  Epochs: 3-5

场景 3: 单张 RTX 4090 (24GB)
  方法: QLoRA
  秩: 16
  学习率: 2e-4
  Epochs: 3-5

场景 4: 快速原型
  方法: QLoRA (r=8)
  学习率: 3e-4
  Epochs: 1-2
```

### 9.2 超参数推荐

```python
def get_hyperparameters(method, task_complexity="medium"):
    """获取推荐超参数"""
    
    configs = {
        "full": {
            "learning_rate": 2e-5,
            "batch_size": 32,
            "epochs": "1-3",
            "warmup_ratio": 0.03,
            "weight_decay": 0.01
        },
        "lora": {
            "learning_rate": 2e-4,
            "rank": 16 if task_complexity == "medium" else 32,
            "alpha": 32,
            "dropout": 0.05,
            "epochs": "3-5"
        },
        "qlora": {
            "learning_rate": 2e-4,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "epochs": "3-5"
        }
    }
    
    return configs[method]
```

### 9.3 常见陷阱

| 陷阱 | 症状 | 解决方案 |
|------|------|----------|
| 学习率过高 | Loss 震荡 | 降低学习率 10 倍 |
| 学习率过低 | Loss 不下降 | 增加学习率 5-10 倍 |
| 过拟合 | 训练 loss 低，验证 loss 高 | 增加 dropout，减少 epochs |
| 欠拟合 | 训练 loss 高 | 增加 rank，增加 epochs |
| 显存溢出 | CUDA OOM | 减小 batch size，使用 QLoRA |
| 梯度爆炸 | Loss 变成 NaN | 启用梯度裁剪 |

## 10. 参考资源

- [Fine-Tuning LLMs in 2025 - Phil Schmid](https://www.philschmid.de/fine-tune-llms-in-2025)
- [LoRA & QLoRA Fine-Tuning Guide - Meta Intelligence](https://www.meta-intelligence.tech/en/insight-lora-finetuning)
- [Fine-Tuning Infrastructure: LoRA, QLoRA, and PEFT at Scale (2025)](https://introl.com/blog/fine-tuning-infrastructure-lora-qlora-peft-scale-guide-2025)
- [Key Insights on Instruction Tuning](https://aiexpjourney.substack.com/p/key-insights-and-best-practices-on)
- [LLM Research Insights: Instruction Masking and New LoRA](https://magazine.sebastianraschka.com/p/llm-research-insights-instruction)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [BitsAndBytes Documentation](https://huggingface.co/docs/bitsandbytes)
- [Liger Kernels - Training Optimization](https://github.com/linkedin/Liger-Kernel)
- [Vision-Language Models Overview (2025-2026)](https://github.com/zli12321/Vision-Language-Models-Overview)
- [Evaluating LoRA, QLoRA, and Full Fine-Tuning (2025)](https://www.atlantis-press.com/proceedings/iwadic-25/126023308)
---

## 11. Prompt Tuning 系列

### 11.1 Prompt Tuning 基础

Prompt Tuning 是一种**参数高效微调方法**,通过在输入中添加可学习的提示向量来适配下游任务,而不修改模型主体参数。

#### 11.1.1 软提示 vs 硬提示

```python
# 硬提示(Hard Prompt) - 离散 token
hard_prompt = "Please summarize the following text:"
# 特点:
# - 人类可读
# - 需要手工设计或搜索
# - 灵活性有限

# 软提示(Soft Prompt) - 连续向量
soft_prompt = nn.Parameter(torch.randn(num_virtual_tokens, hidden_size))
# 特点:
# - 不可读,是连续向量
# - 自动学习
# - 更灵活,性能通常更好
```

**对比分析**:

| 维度 | 硬提示 | 软提示 |
|------|--------|--------|
| 可读性 | ✓ 人类可读 | ✗ 连续向量 |
| 设计方式 | 手工/搜索 | 自动学习 |
| 参数量 | 0(不增加参数) | num_tokens × hidden_size |
| 性能 | 中等 | 通常更好 |
| 可迁移性 | 差 | 好 |

#### 11.1.2 Prompt Embeddings

```python
class PromptEmbeddings(nn.Module):
    """Prompt Embeddings 实现"""
    
    def __init__(self, num_virtual_tokens, hidden_size, vocab_size):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        
        # 可学习的提示向量
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_virtual_tokens, hidden_size)
        )
        
        # 词嵌入层(冻结)
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.word_embeddings.requires_grad_(False)
    
    def forward(self, input_ids):
        # 获取输入 token 的嵌入
        token_embeddings = self.word_embeddings(input_ids)
        
        # 复制 prompt embeddings 到 batch 中的每个样本
        batch_size = input_ids.size(0)
        prompt_embeddings = self.prompt_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # 拼接:[prompt_embeddings, token_embeddings]
        combined_embeddings = torch.cat([prompt_embeddings, token_embeddings], dim=1)
        
        return combined_embeddings
```

**数学原理**:

```
传统输入: h = Embedding(x)
Prompt Tuning: h' = Concat(P, Embedding(x))

其中:
- P ∈ R^{n×d} 是可学习的 prompt 矩阵
- n 是虚拟 token 数量
- d 是隐藏层维度
- x 是输入 token IDs
```

#### 11.1.3 参数效率

```python
def calculate_prompt_params(num_virtual_tokens, hidden_size, model_params):
    """计算 Prompt Tuning 的参数量"""
    prompt_params = num_virtual_tokens * hidden_size
    percentage = (prompt_params / model_params) * 100
    
    return {
        "prompt_params": prompt_params,
        "model_params": model_params,
        "percentage": f"{percentage:.4f}%"
    }

# 示例:T5-XXL (11B 参数)
result = calculate_prompt_params(
    num_virtual_tokens=20,
    hidden_size=4096,
    model_params=11e9
)
print(result)
# 输出: {'prompt_params': 81920, 'model_params': 11000000000.0, 'percentage': '0.0007%'}
```

**参数效率优势**:
- 仅需 **0.001% - 0.01%** 的参数
- 单个 prompt 可以适配多个任务
- 存储成本极低(几 MB)

#### 11.1.4 适用场景

```python
def when_to_use_prompt_tuning():
    """
    Prompt Tuning 最适合的场景:
    
    ✓ 适合:
    1. 分类任务(情感分析、主题分类)
    2. 快速原型验证
    3. 多任务学习(每个任务一个 prompt)
    4. 资源极其受限的环境
    5. 需要快速切换任务的场景
    
    ✗ 不适合:
    1. 需要深度领域适配的任务
    2. 复杂生成任务(代码生成、长文本)
    3. 需要显著改变模型行为的场景
    4. 对性能要求极高的生产环境
    """
    pass
```

**实践建议**:
- 从 **20-100 个虚拟 token** 开始
- 对于简单任务,20 个 token 通常足够
- 对于复杂任务,可以尝试 100-200 个 token
- 超过 200 个 token 通常收益递减

### 11.2 P-Tuning v1

P-Tuning v1 引入了**连续提示向量**的概念,并通过 LSTM 编码器生成提示。

#### 11.2.1 连续提示向量

```python
class PTuningV1(nn.Module):
    """P-Tuning v1 实现"""
    
    def __init__(self, num_virtual_tokens, hidden_size, embedding_size):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.hidden_size = hidden_size
        
        # 可学习的 token embeddings(锚点)
        self.anchor_tokens = nn.Parameter(
            torch.randn(num_virtual_tokens, embedding_size)
        )
        
        # LSTM 编码器
        self.lstm_encoder = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size // 2,  # 双向 LSTM
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, batch_size):
        # 通过 LSTM 编码锚点 token
        lstm_output, _ = self.lstm_encoder(self.anchor_tokens.unsqueeze(0))
        
        # 投影到模型隐藏空间
        prompt_embeddings = self.projection(lstm_output.squeeze(0))
        
        # 扩展到 batch
        prompt_embeddings = prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        return prompt_embeddings
```

#### 11.2.2 LSTM 编码器

**为什么使用 LSTM?**

```python
def why_lstm_encoder():
    """
    LSTM 编码器的优势:
    
    1. 捕获 token 间的依赖关系
       - 单向 LSTM:从左到右的依赖
       - 双向 LSTM:完整的上下文信息
    
    2. 参数效率
       - LSTM 参数 << 直接学习所有 prompt embeddings
       - 例如:100 tokens, 768 dim
         * 直接学习: 100 × 768 = 76,800 参数
         * LSTM: 约 10,000 参数(节省 87%)
    
    3. 正则化效果
       - LSTM 的结构约束防止过拟合
       - 更平滑的提示空间
    """
    pass
```

**LSTM 配置推荐**:

```python
# 小模型(< 1B 参数)
lstm_config_small = {
    "hidden_size": 256,
    "num_layers": 1,
    "bidirectional": False
}

# 中型模型(1B - 10B)
lstm_config_medium = {
    "hidden_size": 512,
    "num_layers": 2,
    "bidirectional": True
}

# 大型模型(> 10B)
lstm_config_large = {
    "hidden_size": 1024,
    "num_layers": 2,
    "bidirectional": True
}
```

#### 11.2.3 位置插入

```python
class PositionalPromptInjection:
    """Prompt 插入位置策略"""
    
    @staticmethod
    def prepend_to_input(prompt_embeddings, token_embeddings):
        """
        策略 1: 在输入前插入(最常用)
        
        输入: [CLS] token1 token2 ... [SEP]
        输出: [prompt1, prompt2, ..., CLS, token1, token2, ..., SEP]
        """
        return torch.cat([prompt_embeddings, token_embeddings], dim=1)
    
    @staticmethod
    def insert_after_cls(prompt_embeddings, cls_embedding, token_embeddings):
        """
        策略 2: 在 CLS 后插入
        
        输入: [CLS] token1 token2 ... [SEP]
        输出: [CLS, prompt1, prompt2, ..., token1, token2, ..., SEP]
        """
        return torch.cat([cls_embedding, prompt_embeddings, token_embeddings], dim=1)
    
    @staticmethod
    def insert_at_all_layers(prompt_embeddings, hidden_states_list):
        """
        策略 3: 在每一层插入(P-Tuning v2)
        """
        updated_states = []
        for hidden_states in hidden_states_list:
            updated = torch.cat([prompt_embeddings, hidden_states], dim=1)
            updated_states.append(updated)
        return updated_states
```

#### 11.2.4 效果分析

```python
# P-Tuning v1 性能对比(基于原始论文)
performance_comparison = {
    "SuperGLUE Benchmark": {
        "Fine-tuning": 87.5,
        "P-Tuning v1": 85.2,
        "Prompt Tuning": 82.1,
        "Gap vs Fine-tuning": -2.3
    },
    "参数效率": {
        "Fine-tuning": "100%",
        "P-Tuning v1": "0.01%",
        "Prompt Tuning": "0.001%"
    },
    "训练速度": {
        "Fine-tuning": "1x",
        "P-Tuning v1": "3-5x",
        "Prompt Tuning": "5-8x"
    }
}

# 关键发现
def p_tuning_v1_insights():
    """
    P-Tuning v1 的关键发现:
    
    1. LSTM 编码器比直接学习 prompt embeddings 更有效
    2. 双向 LSTM 优于单向 LSTM
    3. 在大型模型上性能接近全量微调
    4. 在小型模型上性能下降明显
    5. 对于序列标注任务效果较差
    """
    pass
```

### 11.3 P-Tuning v2

P-Tuning v2 是对 v1 的重大改进,主要解决了**深度提示**和**多任务适应性**问题。

#### 11.3.1 多层提示

```python
class PTuningV2(nn.Module):
    """P-Tuning v2 实现 - 多层提示"""
    
    def __init__(self, num_virtual_tokens, hidden_size, num_layers):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.num_layers = num_layers
        
        # 为每一层学习独立的 prompt
        self.prompts = nn.Parameter(
            torch.randn(num_layers, 2, num_virtual_tokens, hidden_size)
        )
        # 2 表示 key 和 value
    
    def forward(self, layer_idx, batch_size):
        """
        获取指定层的 prompt
        
        Args:
            layer_idx: 层索引 (0 到 num_layers-1)
            batch_size: batch 大小
        
        Returns:
            past_key_values: (key, value) 对
        """
        # 获取该层的 prompt
        layer_prompt = self.prompts[layer_idx]
        
        # 扩展到 batch
        layer_prompt = layer_prompt.unsqueeze(1).expand(
            -1, batch_size, -1, -1
        )
        
        # 返回 (key, value)
        key_prompt = layer_prompt[0]  # (batch, num_tokens, hidden_size)
        value_prompt = layer_prompt[1]  # (batch, num_tokens, hidden_size)
        
        return key_prompt, value_prompt
    
    def get_all_past_key_values(self, batch_size):
        """获取所有层的 past_key_values"""
        past_key_values = []
        for layer_idx in range(self.num_layers):
            key, value = self.forward(layer_idx, batch_size)
            past_key_values.append((key, value))
        return tuple(past_key_values)
```

**与 v1 的对比**:

| 特性 | P-Tuning v1 | P-Tuning v2 |
|------|-------------|-------------|
| 提示层数 | 仅输入层 | 所有层 |
| 参数量 | 少 | 更多(但仍然高效) |
| 性能 | 中等 | 显著提升 |
| 适用任务 | 简单任务 | 复杂任务 |
| 实现复杂度 | 低 | 中等 |

#### 11.3.2 任务特定头

```python
class TaskSpecificHeads(nn.Module):
    """为不同任务使用不同的提示头"""
    
    def __init__(self, num_tasks, num_virtual_tokens, hidden_size, num_layers):
        super().__init__()
        self.num_tasks = num_tasks
        
        # 每个任务独立的 prompts
        self.task_prompts = nn.ModuleList([
            PTuningV2(num_virtual_tokens, hidden_size, num_layers)
            for _ in range(num_tasks)
        ])
        
        # 任务路由器(可选)
        self.task_router = nn.Linear(hidden_size, num_tasks)
    
    def forward(self, input_embeddings, task_id=None):
        """
        根据任务 ID 选择对应的 prompt
        """
        batch_size = input_embeddings.size(0)
        
        if task_id is not None:
            # 使用指定任务的 prompt
            prompt_module = self.task_prompts[task_id]
        else:
            # 自动选择任务(通过路由器)
            task_logits = self.task_router(input_embeddings.mean(dim=1))
            task_id = torch.argmax(task_logits, dim=-1)
            prompt_module = self.task_prompts[task_id]
        
        # 获取 prompts
        past_key_values = prompt_module.get_all_past_key_values(batch_size)
        
        return past_key_values
```

#### 11.3.3 性能提升

```python
# P-Tuning v2 vs v1 性能对比
performance_improvement = {
    "Natural Language Inference": {
        "v1": 68.5,
        "v2": 73.2,
        "improvement": "+4.7"
    },
    "Sentiment Analysis": {
        "v1": 85.3,
        "v2": 89.1,
        "improvement": "+3.8"
    },
    "Question Answering": {
        "v1": 62.1,
        "v2": 68.9,
        "improvement": "+6.8"
    },
    "Text Summarization": {
        "v1": 45.2,
        "v2": 52.8,
        "improvement": "+7.6"
    }
}

def why_v2_better():
    """
    P-Tuning v2 性能提升的原因:
    
    1. 多层提示
       - 在 Transformer 的每一层都添加提示
       - 允许更深层次的适配
    
    2. 移除 LSTM 编码器
       - 直接学习 prompt embeddings
       - 减少信息瓶颈
    
    3. 任务特定头
       - 不同任务使用不同提示
       - 避免任务间干扰
    
    4. 更好的初始化
       - 使用词嵌入初始化
       - 加速收敛
    """
    pass
```

#### 11.3.4 实现细节

```python
class PTuningV2Trainer:
    """P-Tuning v2 训练器"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.setup_prompts()
    
    def setup_prompts(self):
        """初始化 prompts"""
        num_layers = self.model.config.num_hidden_layers
        hidden_size = self.model.config.hidden_size
        
        self.prompt_tuning = PTuningV2(
            num_virtual_tokens=self.config.num_virtual_tokens,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        # 使用词嵌入初始化
        self._initialize_from_vocab()
        
        # 冻结主模型
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _initialize_from_vocab(self):
        """从词嵌入初始化 prompts"""
        vocab_embeddings = self.model.get_input_embeddings().weight
        
        # 随机选择 num_virtual_tokens 个词向量
        indices = torch.randperm(vocab_embeddings.size(0))[:self.config.num_virtual_tokens]
        selected_embeddings = vocab_embeddings[indices]
        
        # 用这些向量初始化 prompts
        with torch.no_grad():
            for layer_idx in range(self.prompt_tuning.num_layers):
                self.prompt_tuning.prompts[layer_idx, 0].copy_(selected_embeddings)
                self.prompt_tuning.prompts[layer_idx, 1].copy_(selected_embeddings)
    
    def inject_prompts(self, model_inputs):
        """将 prompts 注入到模型中"""
        batch_size = model_inputs['input_ids'].size(0)
        
        # 获取所有层的 prompts
        past_key_values = self.prompt_tuning.get_all_past_key_values(batch_size)
        
        # 添加到 model_inputs
        model_inputs['past_key_values'] = past_key_values
        
        # 调整 attention_mask
        num_virtual_tokens = self.config.num_virtual_tokens
        prompt_attention_mask = torch.ones(
            batch_size, num_virtual_tokens,
            device=model_inputs['attention_mask'].device
        )
        model_inputs['attention_mask'] = torch.cat(
            [prompt_attention_mask, model_inputs['attention_mask']], dim=1
        )
        
        return model_inputs
    
    def train(self, train_dataloader, optimizer, num_epochs):
        """训练循环"""
        self.prompt_tuning.train()
        
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                # 注入 prompts
                batch = self.inject_prompts(batch)
                
                # 前向传播
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # 反向传播
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
```

### 11.4 Prefix Tuning

Prefix Tuning 为每个 Transformer 层添加**可学习的前缀参数**,修改注意力机制。

#### 11.4.1 前缀参数

```python
class PrefixTuning(nn.Module):
    """Prefix Tuning 实现"""
    
    def __init__(self, num_prefix_tokens, hidden_size, num_layers, num_attention_heads):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        
        # 为每一层学习独立的前缀
        self.prefix_params = nn.ParameterDict()
        for layer_idx in range(num_layers):
            self.prefix_params[f"layer_{layer_idx}"] = nn.Parameter(
                torch.randn(2, num_prefix_tokens, hidden_size)
            )
            # 2 表示 key 和 value
    
    def get_prefix_for_layer(self, layer_idx, batch_size):
        """获取指定层的前缀"""
        prefix = self.prefix_params[f"layer_{layer_idx}"]
        
        # 扩展到 batch
        prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 分离 key 和 value
        key_prefix = prefix[0]  # (batch, num_prefix, hidden_size)
        value_prefix = prefix[1]  # (batch, num_prefix, hidden_size)
        
        # 重塑为多头格式
        key_prefix = key_prefix.view(
            batch_size, self.num_prefix_tokens, 
            self.num_attention_heads, self.head_dim
        ).transpose(1, 2)  # (batch, heads, num_prefix, head_dim)
        
        value_prefix = value_prefix.view(
            batch_size, self.num_prefix_tokens,
            self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        
        return key_prefix, value_prefix
```

#### 11.4.2 注意力修改

```python
class ModifiedAttention(nn.Module):
    """修改后的注意力机制(集成 Prefix)"""
    
    def __init__(self, original_attention, prefix_tuning, layer_idx):
        super().__init__()
        self.original_attention = original_attention
        self.prefix_tuning = prefix_tuning
        self.layer_idx = layer_idx
    
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size = hidden_states.size(0)
        
        # 获取前缀
        key_prefix, value_prefix = self.prefix_tuning.get_prefix_for_layer(
            self.layer_idx, batch_size
        )
        
        # 原始注意力计算
        query = self.original_attention.query(hidden_states)
        key = self.original_attention.key(hidden_states)
        value = self.original_attention.value(hidden_states)
        
        # 拼接前缀
        key = torch.cat([key_prefix, key], dim=2)
        value = torch.cat([value_prefix, value], dim=2)
        
        # 调整 attention_mask
        if attention_mask is not None:
            num_prefix = self.prefix_tuning.num_prefix_tokens
            prefix_mask = torch.zeros(
                batch_size, 1, 1, num_prefix,
                device=attention_mask.device
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
        
        # 计算注意力
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # 加权求和
        context = torch.matmul(attention_probs, value)
        
        # 输出投影
        output = self.original_attention.output(context)
        
        return output, attention_probs
```

#### 11.4.3 生成任务优化

```python
class PrefixTuningForGeneration:
    """Prefix Tuning 在生成任务中的优化"""
    
    def __init__(self, model, prefix_tuning):
        self.model = model
        self.prefix_tuning = prefix_tuning
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """准备生成输入"""
        batch_size = input_ids.size(0)
        
        # 获取前缀
        past_key_values = []
        for layer_idx in range(self.model.config.num_hidden_layers):
            key_prefix, value_prefix = self.prefix_tuning.get_prefix_for_layer(
                layer_idx, batch_size
            )
            past_key_values.append((key_prefix, value_prefix))
        
        kwargs['past_key_values'] = tuple(past_key_values)
        
        return input_ids, kwargs
    
    def generate(self, input_ids, max_length=100, **kwargs):
        """使用 Prefix Tuning 生成文本"""
        input_ids, model_kwargs = self.prepare_inputs_for_generation(
            input_ids, **kwargs
        )
        
        # 使用模型的 generate 方法
        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            **model_kwargs
        )
        
        return outputs
    
    def optimize_for_inference(self):
        """
        推理优化:
        1. 前缀可以预先计算并缓存
        2. 减少重复计算
        3. 提高生成速度
        """
        # 预计算前缀(对于固定任务)
        self.cached_prefix = {}
        for layer_idx in range(self.model.config.num_hidden_layers):
            key, value = self.prefix_tuning.get_prefix_for_layer(layer_idx, 1)
            self.cached_prefix[layer_idx] = (key, value)
```

**生成任务性能**:

```python
generation_performance = {
    "Summarization (CNN/DailyMail)": {
        "Fine-tuning": {"ROUGE-1": 42.5, "ROUGE-2": 19.8, "ROUGE-L": 38.9},
        "Prefix Tuning": {"ROUGE-1": 41.2, "ROUGE-2": 19.1, "ROUGE-L": 37.8},
        "Gap": "-3%"
    },
    "Translation (WMT14)": {
        "Fine-tuning": {"BLEU": 28.5},
        "Prefix Tuning": {"BLEU": 27.8},
        "Gap": "-2.5%"
    },
    "Data-to-Text (WebNLG)": {
        "Fine-tuning": {"BLEU": 56.2},
        "Prefix Tuning": {"BLEU": 55.1},
        "Gap": "-2%"
    }
}
```

#### 11.4.4 代码示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def prefix_tuning_example():
    """Prefix Tuning 完整示例"""
    
    # 1. 加载模型
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. 创建 Prefix Tuning 配置
    from peft import PrefixTuningConfig, get_peft_model
    
    prefix_config = PrefixTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=20,
        prefix_projection=True  # 使用投影层
    )
    
    # 3. 应用 Prefix Tuning
    model = get_peft_model(model, prefix_config)
    model.print_trainable_parameters()
    # 输出: trainable params: 0.08% || all params: 100%
    
    # 4. 训练
    from transformers import TrainingArguments, Trainer
    
    training_args = TrainingArguments(
        output_dir="./prefix-tuning-output",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        learning_rate=1e-3,  # Prefix Tuning 通常用更大的学习率
        warmup_steps=100,
        logging_steps=50,
        save_steps=500
    )
```
