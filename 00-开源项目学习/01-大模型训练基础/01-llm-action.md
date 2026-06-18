# llm-action - 大模型实战教程学习指南

## 项目概述

**一句话总结**：这是一个由一线大厂工程师沉淀的**大模型全栈技术教程**，覆盖从训练、推理、压缩到部署的完整工程链路，是目前中文社区最系统的大模型实战知识库之一。

### 核心亮点

- 🎯 **工程导向**：不是纯理论，而是"理论+代码+实践"三位一体，每个技术点都有配套代码
- 📚 **体系完整**：涵盖 15+ 技术模块，从底层 AI 芯片到上层应用开发，形成完整知识闭环
- 🔥 **实战丰富**：包含 Alpaca、ChatGLM、LLaMA 等 10+ 主流模型的训练/微调/推理实战
- 🚀 **持续更新**：作者持续维护，紧跟技术前沿（已覆盖 DeepSeek、Qwen 等最新模型）

### 适合谁学

| 学习者类型 | 你能获得什么 |
|-----------|-------------|
| **AI 入门者** | 系统的大模型知识地图，避免碎片化学习 |
| **算法工程师** | 分布式训练、微调、推理优化的工程实践指南 |
| **运维工程师** | AI 集群部署、LLMOps、性能压测的实操手册 |
| **学生/研究者** | 论文背后的工程实现细节和代码参考 |

---

## 核心架构解析

### 教程整体结构

这个项目就像一个**大模型技术的"百科全书"**，按技术领域分为以下模块：

```
llm-action/
├── 📖 docs/                    # 核心教程文档（理论讲解）
├── 🏋️ llm-train/              # 模型训练（预训练、微调、RLHF）
├── 🚀 llm-inference/          # 模型推理（vLLM、TensorRT-LLM 等）
├── 📦 llm-compression/        # 模型压缩（量化、剪枝、蒸馏）
├── 📊 llm-eval/               # 模型评测（效果评估、性能压测）
├── 🧠 llm-algo/               # 算法架构（Transformer、LLaMA、ChatGLM 等）
├── 💻 ai-framework/           # AI 框架（PyTorch、DeepSpeed、Megatron 等）
├── 🔧 ai-compiler/            # AI 编译器（TVM、XLA 等）
├── 🌐 ai-infra/               # AI 基础设施（GPU、集群、网络）
├── 🇨🇳 llm-localization/     # 国产化适配（昇腾、海光等）
├── 🛠️ llm-tools/              # 实用工具集
├── 📝 llm-interview/          # 面试题集锦
└── 🎯 llm-application/        # 应用开发
```

### 技术模块详解

#### 1️⃣ **模型训练模块** (`llm-train/`)

这是项目的**核心精华**，包含：

- **全量微调实战**：Alpaca、BELLE、Vicuna 等经典模型训练代码
- **高效微调（PEFT）**：LoRA、QLoRA、P-Tuning、Prefix Tuning 等 6 种主流方法
- **分布式训练**：PyTorch DDP、DeepSpeed、Megatron-LM 多机多卡训练
- **RLHF 训练**：基于 DeepSpeed Chat 的一键式 RLHF 训练流程
- **内存优化**：GaLore 技术（一张 4090 也能预训练 7B 模型）

#### 2️⃣ **模型推理模块** (`llm-inference/`)

- **推理框架对比**：vLLM、TensorRT-LLM、FasterTransformer、SGLang 等
- **推理优化技术**：KV Cache、PagedAttention、Continuous Batching
- **解码策略**：Greedy Search、Beam Search、Top-k/Top-p 采样
- **服务化部署**：Triton 推理服务框架实战

#### 3️⃣ **模型压缩模块** (`llm-compression/`)

- **量化技术**：GPTQ、AWQ、SmoothQuant、FP8/FP4 等 10+ 种方案
- **剪枝技术**：结构化剪枝（SliceGPT）、非结构化剪枝（Wanda）
- **知识蒸馏**：MINILLM、CoT 蒸馏等方法

#### 4️⃣ **分布式训练并行技术** (`docs/llm-base/distribution-parallelism/`)

这是一个**九篇系列教程**，系统讲解：

1. 分布式并行概述
2. 数据并行（Data Parallelism）
3. 流水线并行（Pipeline Parallelism）
4. 张量并行（Tensor Parallelism）
5. 序列并行（Sequence Parallelism）
6. 多维混合并行
7. 自动并行（Alpa）
8. MOE 并行
9. 总结与实践建议

#### 5️⃣ **算法架构模块** (`llm-algo/`)

解析主流大模型的内部结构：

- **基础架构**：Transformer、BERT、GPT 系列
- **开源模型**：LLaMA/LLaMA2、ChatGLM/ChatGLM2/ChatGLM3、Bloom、Qwen、DeepSeek
- **关键技术**：RoPE 旋转位置编码、MoE 混合专家架构

### 学习路径规划

根据学习目标，推荐以下路径：

```
🎯 路径 1：快速上手微调大模型
  1. 阅读 docs/llm-base/ 了解基础概念
  2. 学习 llm-train/peft/ 中的 LoRA 微调
  3. 运行 alpaca-lora/ 示例代码

🎯 路径 2：深入分布式训练
  1. 精读 docs/llm-base/distribution-parallelism/ 九篇教程
  2. 学习 ai-framework/deepspeed/ 和 megatron-lm/
  3. 实践 llm-train/megatron-deepspeed/ 预训练

🎯 路径 3：模型推理与部署
  1. 阅读 llm-inference/README.md 了解推理框架
  2. 学习 vLLM、TensorRT-LLM 使用方法
  3. 掌握 KV Cache、解码策略等优化技术

🎯 路径 4：模型压缩与优化
  1. 学习 llm-compression/quantization/ 量化技术
  2. 理解 GPTQ、AWQ 等方案原理
  3. 实践模型量化推理
```

---

## 代码逻辑主线

### 模块 1：LoRA 高效微调

**核心代码位置**：`llm-train/alpaca-lora/`

**技术原理**（用比喻理解）：
> 想象你要改造一座大楼（大模型），全量微调是重新装修整栋楼，而 LoRA 是在关键房间外挂"小模块"。这些模块参数量很小（通常只有原模型的 0.1%），但能显著改变模型行为。

**关键代码片段**（`llm-train/peft/clm/peft_lora_clm.ipynb`）：

```python
from peft import LoraConfig, get_peft_model

# 配置 LoRA 参数
config = LoraConfig(
    r=16,                          # 低秩矩阵的秩（越小参数越少）
    lora_alpha=32,                 # 缩放系数
    target_modules=["q_proj", "v_proj"],  # 要修改的注意力层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 将 LoRA 应用到基础模型
model = get_peft_model(base_model, config)
model.print_trainable_parameters()  # 查看可训练参数比例
```

**实践要点**：
- **r 值选择**：通常 8~16 即可，太大失去"高效"意义
- **target_modules**：一般选择注意力层的 `q_proj`、`v_proj`
- **显存节省**：7B 模型用 QLoRA（4bit 量化+LoRA）仅需 ~8GB 显存

---

### 模块 2：DeepSpeed 分布式训练

**核心代码位置**：`ai-framework/deepspeed/`、`llm-train/deepspeedchat/`

**技术原理**（类比理解）：
> 假设你要搬运 1000 块砖（模型参数），一个人搬太慢（单卡），DeepSpeed 的策略是：
> - **ZeRO-1**：把砖分给 10 个人搬（优化器状态分片）
> - **ZeRO-2**：连搬砖的工具也分开拿（梯度分片）
> - **ZeRO-3**：连人站的位置都分散开（模型参数分片）

**DeepSpeed 配置文件示例**（`llm-train/alpaca/ds_config.json`）：

```json
{
  "zero_optimization": {
    "stage": 3,              // ZeRO-3 级别
    "offload_optimizer": {
      "device": "cpu",       // 优化器卸载到 CPU
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",       // 参数卸载到 CPU
      "pin_memory": true
    }
  },
  "fp16": {
    "enabled": true          // 混合精度训练
  },
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4
}
```

**运行命令**：
```bash
deepspeed --num_gpus=8 train.py --deepspeed ds_config.json
```

**实践要点**：
- **ZeRO-2 vs ZeRO-3**：ZeRO-3 更省显存但通信开销更大
- **CPU Offload**：显存不够时用 CPU 内存换，但速度会降低 30%~50%
- **混合精度**：启用 FP16/BF16 可加速 2~3 倍，BF16 更稳定

---

### 模块 3：模型量化（GPTQ）

**核心代码位置**：`llm-compression/quantization/`

**技术原理**（通俗解释）：
> 模型量化就像把高清照片（FP16，16bit）压缩成普通照片（INT8，8bit），文件变小了，但肉眼几乎看不出区别。GPTQ 的核心是"逐层量化，逐层校准"，保证压缩后质量不下降。

**量化代码示例**：

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 加载模型和量化配置
quantize_config = BaseQuantizeConfig(
    bits=4,                    // 量化到 4bit
    group_size=128,            // 分组大小
    damp_percent=0.01
)

model = AutoGPTQForCausalLM.from_pretrained(
    "model_path",
    quantize_config=quantize_config
)

# 执行量化（需要校准数据）
model.quantize(calibration_data)

# 保存量化模型
model.save_quantized("quantized_model")
```

**实践要点**：
- **4bit vs 8bit**：4bit 模型大小减少 75%，质量损失约 1~2%
- **校准数据**：只需 128~512 条样本即可
- **推理加速**：量化后推理速度提升 2~3 倍

---

### 模块 4：vLLM 推理优化

**核心代码位置**：`llm-inference/vllm/`

**技术原理**（类比说明）：
> 传统推理像"排队办事"，每次只能服务一个请求。vLLM 的 **PagedAttention** 像"医院分诊系统"，把多个请求的 KV Cache 分页管理，实现**Continuous Batching**（连续批处理），吞吐量提升 10~24 倍。

**vLLM 使用示例**：

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# 配置生成参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# 批量推理
prompts = ["Hello, my name is", "The capital of France is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

**启动推理服务**：
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 2
```

**实践要点**：
- **Tensor Parallel**：多卡推理时设置 `--tensor-parallel-size`
- **KV Cache 管理**：vLLM 自动管理，无需手动优化
- **吞吐量优化**：批量请求比单请求效率高 10 倍以上

---

### 模块 5：RLHF 训练（DeepSpeed Chat）

**核心代码位置**：`llm-train/deepspeedchat/`

**技术原理**（比喻理解）：
> RLHF 就像"教小孩说话"：
> 1. **SFT 阶段**：先教他标准答案（指令微调）
> 2. **Reward 阶段**：请老师打分（奖励模型）
> 3. **RL 阶段**：根据反馈不断改进（PPO 优化）

**训练流程**：
```bash
# Step 1: 监督微调（SFT）
deepspeed train_sft.py --deepspeed ds_sft.json

# Step 2: 训练奖励模型（RM）
deepspeed train_reward.py --deepspeed ds_rm.json

# Step 3: RLHF 优化（PPO）
deepspeed train_rl.py --deepspeed ds_rl.json
```

**实践要点**：
- **数据质量**：SFT 数据质量决定模型上限
- **Reward 模型**：需要人工标注的偏好数据
- **训练稳定性**：PPO 训练容易不稳定，建议用小学习率

---

## 快速上手实践

### 环境配置步骤

**Step 1：基础环境准备**

```bash
# 创建 Conda 环境
conda create -n llm python=3.10 -y
conda activate llm

# 安装 PyTorch（根据你的 GPU 型号）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装核心依赖
pip install transformers accelerate peft bitsandbytes
pip install deepspeed trl
```

**Step 2：验证 GPU 环境**

```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"CUDA 版本: {torch.version.cuda}")
```

**Step 3：下载项目代码**

```bash
git clone https://github.com/liguodongiot/llm-action.git
cd llm-action
```

---

### 运行第一个示例：LoRA 微调 Alpaca

**Step 1：进入示例目录**

```bash
cd llm-train/alpaca-lora
```

**Step 2：安装依赖**

```bash
pip install -r requirements.txt
```

**Step 3：准备数据**

项目已提供示例数据格式（`alpaca_data.json`），结构如下：

```json
[
  {
    "instruction": "将以下英文翻译成中文",
    "input": "Hello, how are you?",
    "output": "你好，你怎么样？"
  }
]
```

**Step 4：启动训练**

```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'alpaca_data.json' \
    --output_dir './lora-alpaca' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05
```

**显存需求**：
- 7B 模型 + LoRA：~12GB（单卡 RTX 3090/4090 可运行）
- 7B 模型 + QLoRA（4bit）：~8GB（消费级显卡友好）

---

### 预期输出与验证方法

**训练过程中的输出**：

```
Step 100: loss = 1.234
Step 200: loss = 0.987
Step 300: loss = 0.856
...
Training completed! Model saved to ./lora-alpaca
```

**验证方法 1：加载模型推理**

```python
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

base_model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
model = PeftModel.from_pretrained(base_model, "./lora-alpaca")
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

prompt = "将以下英文翻译成中文：Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**验证方法 2：合并 LoRA 权重**

```bash
python export_hf_checkpoint.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights './lora-alpaca' \
    --output_dir './merged_model'
```

合并后的模型可像普通模型一样使用，无需 LoRA 加载逻辑。

---

## 核心知识点总结

以下是学习本项目**必须掌握的 8 个核心概念**：

### 1. Transformer 架构

**是什么**：大模型的"地基"，由 Attention（注意力机制）和 FFN（前馈网络）组成。

**为什么重要**：99% 的大模型（GPT、LLaMA、ChatGLM）都基于 Transformer。不理解它，就无法理解后续所有技术。

**关键组件**：
- **Self-Attention**：让模型理解句子中词与词的关系
- **Multi-Head**：多个注意力头，捕捉不同维度的信息
- **Position Encoding**：告诉模型词的顺序（如 RoPE）

---

### 2. 预训练 vs 微调 vs 对齐

**通俗理解**：
- **预训练**：让模型"博览群书"，学习通用知识（耗时最长，成本最高）
- **微调（SFT）**：教模型"专业技能"，如医疗、法律领域知识
- **对齐（RLHF/DPO）**：让模型"学会说话"，符合人类偏好

**为什么重要**：这是大模型训练的三大阶段，每个阶段目标不同、数据不同、技术不同。

---

### 3. 分布式并行技术

**核心思想**：模型太大（几百 GB），单张 GPU 装不下（通常 40~80GB），必须拆分到多张卡。

**四种并行方式**：

| 并行方式 | 拆分对象 | 适用场景 |
|---------|---------|---------|
| 数据并行（DP） | 训练数据 | 模型能单卡装下，加速训练 |
| 流水线并行（PP） | 模型层数 | 超大型模型预训练 |
| 张量并行（TP） | 矩阵运算 | 推理阶段低延迟 |
| ZeRO（零冗余） | 优化器状态 | 微调阶段省显存 |

**为什么重要**：不懂分布式训练，就无法训练或微调 7B 以上的模型。

---

### 4. LoRA 高效微调

**核心思想**：不改动原始模型权重，只在某些层旁边"并联"小矩阵，训练时只更新这些小矩阵。

**为什么重要**：
- 显存需求降低 70%+
- 训练速度提升 2~3 倍
- 一个基础模型可对应多个 LoRA 插件（节省存储）

---

### 5. 模型量化

**核心思想**：把模型参数从高精度（FP16，16bit）转换为低精度（INT8/INT4，8/4bit），模型体积缩小 50%~75%。

**为什么重要**：
- 部署成本大幅降低（7B 模型从 14GB 降到 3.5GB）
- 推理速度提升 2~4 倍
- 让消费级显卡也能跑大模型

---

### 6. KV Cache

**是什么**：推理时缓存已生成 token 的 Key 和 Value，避免重复计算。

**为什么重要**：
- 推理速度提升 3~5 倍
- 显存占用的主要部分（有时比模型本身还大）
- vLLM、TensorRT-LLM 等框架的核心优化点

**类比理解**：就像"记住已读内容"，不用每次重新读一遍。

---

### 7. RLHF（基于人类反馈的强化学习）

**是什么**：让人类标注员对模型输出打分，用强化学习（PPO）优化模型，使其更符合人类偏好。

**为什么重要**：
- ChatGPT 的核心技术
- 让模型从"能回答问题"到"回答得好"
- 减少有害输出、提高有用性

---

### 8. 推理服务化

**是什么**：把训练好的模型变成 API 服务，供其他应用调用。

**关键技术**：
- **Triton**：NVIDIA 官方推理服务框架
- **vLLM API Server**：自带 OpenAI 兼容接口
- **Continuous Batching**：提高吞吐量

**为什么重要**：模型最终要落地到产品，必须学会部署服务。

---

## 常见疑问解答

### Q1：我没有 A100/H100 这种专业显卡，能学习这个项目吗？

**答：完全可以！** 这正是本项目的价值所在。

- **LoRA 微调 7B 模型**：RTX 3090/4090（24GB）就够了
- **QLoRA（4bit 量化）**：甚至 12GB 显存的显卡也能跑
- **推理测试**：CPU 也能跑（用 llama.cpp），只是慢一点
- **理论学习**：所有教程文档不依赖硬件

**建议学习路径**：
1. 先读 `docs/` 下的理论教程
2. 用 QLoRA 在消费级显卡上微调小模型（7B 以下）
3. 理解原理后，再考虑用云端 GPU（AutoDL、阿里云等）跑大模型

---

### Q2：项目内容太多了，我该从哪里开始？

**答：根据你的目标选择路径：**

**目标 1：快速上手微调一个模型**
```
docs/llm-base/（基础概念，1-2 天）
  ↓
llm-train/peft/（LoRA 微调教程，2-3 天）
  ↓
llm-train/alpaca-lora/（跑通第一个示例，1 天）
```

**目标 2：系统学习大模型技术**
```
按照 README.md 的目录结构，从上到下逐模块学习
重点关注：训练 → 微调 → 推理 → 压缩 → 部署
```

**目标 3：准备面试/找工作**
```
直接看 llm-interview/ 目录下的面试题
针对薄弱环节，反向查找对应教程
```

---

### Q3：教程中的代码还能跑吗？会不会过时了？

**答：需要区分看待：**

**仍然适用的内容（占 80%+）**：
- 技术原理（Transformer、LoRA、量化等）没有变
- DeepSpeed、PEFT、vLLM 等框架的**核心用法**没有变
- 分布式并行、RLHF 等**架构设计**没有变

**需要注意的变化**：
- 具体版本号可能需要更新（如 `transformers==4.28` → `4.35+`）
- 部分模型有新的版本（如 LLaMA → LLaMA2 → LLaMA3）
- 某些早期方案已被更优方案替代（如 QLoRA 替代了部分 LoRA 场景）

**建议**：
- 理解原理为主，代码版本为辅
- 运行代码前，先检查框架的最新文档
- 遇到报错，优先查 GitHub Issues 和官方文档

---

### Q4：学完这个项目，能达到什么水平？

**答：取决于你的学习深度：**

**入门级（读完教程，跑通示例）**：
- ✅ 能理解大模型训练/微调/推理的基本流程
- ✅ 能用 LoRA 微调 7B~13B 模型
- ✅ 能用 vLLM 部署推理服务
- 💼 适合：AI 应用开发、Prompt 工程师

**进阶级（理解原理，能调优）**：
- ✅ 能配置 DeepSpeed 进行分布式训练
- ✅ 能根据显存选择合适的微调策略（全量/LoRA/QLoRA）
- ✅ 能进行模型量化，平衡精度和性能
- 💼 适合：算法工程师、MLOps 工程师

**专家级（阅读源码，能改进）**：
- ✅ 能自定义分布式并行策略
- ✅ 能改进量化/剪枝算法
- ✅ 能优化推理框架的性能瓶颈
- 💼 适合：AI 架构师、研究工程师

---

### Q5：这个项目和我主仓库里的其他教程（如 LangChain、Agent 开发）是什么关系？

**答：这是"底层基础"与"上层应用"的关系：**

```
                    ┌─────────────────┐
                    │   AI 应用开发    │  ← LangChain、Agent、RAG
                    ├─────────────────┤
                    │  模型服务化部署   │  ← vLLM、Triton、FastAPI
                    ├─────────────────┤
llm-action 专注这里  │  训练/微调/推理   │  ← 本项目的核心内容
                    ├─────────────────┤
                    │   AI 基础设施     │  ← GPU、集群、网络
                    └─────────────────┘
```

**建议学习顺序**：
1. **先学 llm-action**：理解模型是怎么训练和部署的（2~4 周）
2. **再学 LangChain/Agent**：在上层构建应用（1~2 周）
3. **最后学工程化**：如何把应用变成生产级系统（持续学习）

**类比理解**：
- llm-action 教你"如何造汽车引擎"
- LangChain/Agent 教你"如何驾驶汽车"
- 两者结合，你才能"既懂原理，又会开车"

---

## 学习建议与资源

### 📌 高效学习 tips

1. **先宏观后微观**：先读 README 了解全貌，再深入具体模块
2. **理论+实践结合**：读完教程立即跑代码，不要只看不练
3. **建立知识地图**：用思维导图整理各模块关系（参考项目目录结构）
4. **关注技术演进**：注意教程中的时间戳，了解技术发展历程
5. **加入社区**：项目 README 提供了微信学习群，可进群交流

### 📚 配套资源

- **作者知乎**：吃果冻不吐果冻皮（教程原文链接）
- **配套代码**：每个教程都有对应的代码目录
- **面试题集**：`llm-interview/` 目录下的 1000+ 道面试题
- **环境配置**：`docs/llm-base/a800-env-install.md`（GPU 服务器环境搭建）

### 🎯 下一步行动

1. ⭐ **Star 项目**：`https://github.com/liguodongiot/llm-action`
2. 📖 **精读 README**：了解完整技术图谱
3. 🏃 **跑通第一个示例**：从 `llm-train/alpaca-lora/` 开始
4. 📝 **做学习笔记**：建议按模块整理知识卡片
5. 💬 **加入交流群**：微信联系作者进群（README 有二维码）

---

**文档版本**：v1.0  
**最后更新**：2025-06-15  
**适用项目版本**：llm-action 最新版本  
**文档作者**：基于 llm-action 项目提炼
