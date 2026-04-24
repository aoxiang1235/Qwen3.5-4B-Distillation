# Qwen3.5-4B-Distillation 训练排障记录

## 1) 当前可用环境

- 云平台: Chameleon Cloud
- 镜像: `Ubuntu22.04-CUDA12.3-Guix20240617`
- 服务器: `129.114.108.221` (`shanghai-gpu`)
- GPU: `Tesla P100-PCIE-16GB` x2
- 驱动: `550.54.15`
- CUDA (driver report): `12.4`

已验证 `torch.cuda.is_available() == True`。

---

## 2) 当前稳定训练配置

### 运行命令

```bash
nohup python main.py --batch_size 1 --grad_accum 16 --max_length 384 > train.log 2>&1 &
```

### 蒸馏参数（本次实跑）

- 基座模型: `Qwen/Qwen2.5-3B-Instruct`
- 任务形式: Causal LM + LoRA 微调
- LoRA:
  - `r=16`
  - `lora_alpha=32`
  - `lora_dropout=0.05`
  - target modules: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- 训练参数:
  - `batch_size=1`
  - `grad_accum=16`
  - `max_length=384`
  - `epochs=1.0`（默认）
  - `lr=2e-4`（默认）
  - `evaluation_strategy=steps`
  - `eval_steps=100`
  - `save_steps=100`
  - `logging_steps=20`
- 精度与显存:
  - 模型 dtype: `torch.float16`（P100 不支持 bf16）
  - Trainer: `fp16=False` / `bf16=False`（避免 AMP scaler 冲突）
  - 4bit: 本次最终未启用（未安装 `bitsandbytes`）

### 观测到的速度

- 单 step 约 15 秒
- 1 epoch 约 7.7 小时（1828 steps 粗估）

### 当前效果（实时）

- 已稳定完成:
  - `[1/5] 读取数据`
  - `[2/5] 数据集准备完成`
  - `[3/5] 加载 tokenizer/model`
  - `[4/5] Tokenize`
- 已进入:
  - `[5/5] 开始训练`
- 训练 loop 可持续推进（step 递增），不再出现“启动即崩溃”问题。

---

## 3) 过程中遇到的问题与解决方案

### 问题 A: 远程旧机驱动过老，CUDA 不可用

现象:
- `The NVIDIA driver on your system is too old`
- `torch.cuda.is_available() == False`

原因:
- 旧节点驱动仅 `418.x` / CUDA 10.1，和 `torch+cu121` 不兼容。

解决:
- 切换到新镜像和新节点（上面这台）。

---

### 问题 B: Python/环境混乱（Python 3.6 导致依赖装不上）

现象:
- `peft` 无法安装
- `dataclasses` 缺失等异常

原因:
- 实际在用 `anaconda3` 的 Python 3.6，不是目标环境。

解决:
- 在项目内使用独立 `.venv` + Python 3.10。

---

### 问题 C: `pyarrow/pandas` 与 `numpy` ABI 冲突

现象:
- `_ARRAY_API not found`
- `numpy.core.multiarray failed to import`

原因:
- `numpy 2.x` 与已编译二进制轮子不匹配。

解决:
- 固定兼容版本:
  - `numpy==1.26.4`
  - `pyarrow==14.0.2`
  - `pandas==2.2.2`

---

### 问题 D: Hugging Face 相关依赖冲突

现象:
- `transformers` 要求 `huggingface-hub<1.0`
- `datasets` 或其他安装过程把 hub 升到 `1.x`

解决:
- 固定版本并统一安装:
  - `transformers==4.40.2`
  - `datasets==2.16.1`
  - `huggingface-hub==0.36.2`
  - `fsspec==2023.10.0`

---

### 问题 E: `TrainingArguments` 参数名兼容问题

现象:
- `unexpected keyword argument 'evaluation_strategy'` 或 `eval_strategy`

原因:
- 代码参数名与当前 `transformers` 版本不匹配。

最终方案:
- 当前环境（`transformers 4.40.2`）使用 `evaluation_strategy`。

---

### 问题 F: `Trainer` 参数兼容问题

现象:
- `Trainer.__init__() got an unexpected keyword argument 'tokenizer'` 或 `processing_class`

原因:
- 代码在不同版本间来回切换，参数被误替换。

最终方案:
- 当前环境固定使用 `tokenizer=tokenizer`。

---

### 问题 G: labels 组 batch 失败

现象:
- `Unable to create tensor ... features (labels) ...`

原因:
- 手动设置 `tokenized["labels"]` 与 collator 行为冲突。

解决:
- 删除手动 `labels` 赋值，让 `DataCollatorForLanguageModeling` 统一处理。

---

### 问题 H: 显存 OOM

现象:
- `torch.OutOfMemoryError: CUDA out of memory`

解决:
- 降训练负载:
  - `batch_size=1`
  - `max_length=384`
  - 提高 `grad_accum` 保持有效 batch（例如 16）

---

### 问题 I: P100 上 bfloat16/amp 相关报错

现象:
- `... not implemented for 'BFloat16'`
- `Attempting to unscale FP16 gradients`

原因:
- P100 不支持 bf16，且混合精度设置不一致。

解决:
- 将模型 dtype 从 `torch.bfloat16` 改为 `torch.float16`
- 训练参数使用稳定组合，避免 scaler 冲突。

---

## 4) 当前建议的版本锁（参考）

```txt
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
transformers==4.40.2
accelerate==0.29.3
peft==0.10.0
datasets==2.16.1
huggingface-hub==0.36.2
fsspec==2023.10.0
numpy==1.26.4
pyarrow==14.0.2
pandas==2.2.2
```

---

## 5) 常用检查命令

```bash
# 看训练日志
tail -f ~/Qwen3.5-4B-Distillation/train.log

# 看训练进程
pgrep -af "python main.py"

# 看GPU占用
nvidia-smi
```

---

## 6) 训练记录（不含环境快照）

### 6.1 训练配置记录

- 训练命令:
  - `nohup python main.py --batch_size 1 --grad_accum 16 --max_length 384 > train.log 2>&1 &`
- 关键参数:
  - `batch_size=1`
  - `grad_accum=16`
  - `max_length=384`
  - `epochs=1.0`
  - `lr=2e-4`
  - `evaluation_strategy=steps`
  - `eval_steps=100`
  - `save_steps=100`
  - `logging_steps=20`

### 6.2 过程指标记录

- 训练总 steps: `1828`
- 单 step 耗时: 约 `15s/step`
- 预估单 epoch 总耗时: 约 `7.7h`
- 已观测稳定阶段:
  - `[1/5] 读取数据`
  - `[2/5] 数据集准备完成`
  - `[3/5] 加载 tokenizer/model`
  - `[4/5] Tokenize`
  - `[5/5] 开始训练`

### 6.3 训练健康度记录

- 当前状态: 已进入训练循环，step 可递增
- 历史主要异常:
  - `CUDA OOM`
  - `BFloat16 not supported on P100`
  - `Attempting to unscale FP16 gradients`
- 当前结论:
  - 兼容性问题已逐项修复
  - 训练可持续推进（不再启动即崩）

### 6.4 产物与日志记录

- 训练日志: `~/Qwen3.5-4B-Distillation/train.log`
- 监控命令:
  - `tail -f ~/Qwen3.5-4B-Distillation/train.log`
  - `pgrep -af "python main.py"`
  - `nvidia-smi`
- 推荐补充记录:
  - 首次出现 `loss` 的时间与步数
  - 每 100 steps 的平均耗时
  - 每次 `eval_steps` 的 `eval_loss`
  - 最终 checkpoint 路径与保存时间

---

## 7) 当前稳定参数快照（2026-04-24）

### 7.1 已验证稳定配置（P100）

- 数据: `data/train.jsonl`
- 模型: `Qwen/Qwen2.5-3B-Instruct`
- `model_dtype=float32`
- `batch_size=1`
- `grad_accum=4`
- `max_length=384`
- `lr=1e-6`
- `max_grad_norm=1.0`（代码默认）
- `logging_steps=4`
- 训练策略: `evaluation_strategy=steps`, `eval_steps=100`, `save_steps=100`

### 7.2 验证结果快照

- 30-step 冒烟:
  - `train_loss=2.0052`
  - loss 区间约 `1.81 ~ 2.15`
- 120-step 冒烟:
  - `eval_loss=1.8475`
  - `train_loss=1.9214`
  - loss 区间约 `1.56 ~ 2.15`
  - 无 `nan/inf/loss=0`

### 7.3 正式长跑启动命令（与稳定配置一致）

```bash
nohup python main.py \
  --train_jsonl data/train.jsonl \
  --output_dir distilled-qwen-f32-full \
  --batch_size 1 \
  --grad_accum 4 \
  --max_length 384 \
  --lr 1e-6 \
  --logging_steps 4 \
  --model_dtype float32 \
  > train_f32_full.log 2>&1 &
```

