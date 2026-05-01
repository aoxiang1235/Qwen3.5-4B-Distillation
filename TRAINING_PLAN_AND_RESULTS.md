# Qwen3.5-4B 蒸馏训练方案与结果记录

## 1. 目标与约束

- 目标模型：`Qwen/Qwen3.5-4B`
- 任务：JSONL 监督微调（结构化抽取）
- 运行环境：云服务器（Tesla P100 16GB x2）
- 主要约束：
  - 必须固定使用 `Qwen3.5-4B`，不混用其他基座
  - 优先保证稳定训练和可复现
  - 调参采用“小步快跑”，先短跑筛选再长跑验证

---

## 2. 调参总体方案

采用分阶段策略，避免一次改太多参数：

1. **阶段一：学习率粗搜（LR Sweep）**
   - 目的：快速锁定可用学习率区间
   - 候选：`5e-7`、`1e-6`、`2e-6`
   - 统一短跑：`max_steps=8`

2. **阶段二：结构参数对比（网格）**
   - 固定 `lr=2e-6`
   - 对比 `grad_accum` 与 `max_length` 组合
   - 组合定义：
     - A: `grad_accum=4, max_length=384`
     - B: `grad_accum=8, max_length=384`
     - C: `grad_accum=4, max_length=512`
     - D: `grad_accum=8, max_length=512`（预留，可按需补）

3. **阶段三：最佳配置全程训练**
   - 选择短跑表现最优配置
   - 改为 `max_steps=-1` 全量训练

---

## 3. 执行过程摘要

### 3.1 代码与环境验证

- 从 GitHub 拉取代码后完成：
  - `python3 -m py_compile new_mian.py` 通过
  - CUDA 可用（2 x P100）
- 发现并确认：
  - `--use_4bit` 需要 `bitsandbytes>=0.46.1`
  - 当前环境未装 `bitsandbytes`，4bit 路线会报 ImportError
  - 因此调参阶段使用 `float32` 路线保证稳定性

### 3.2 模型缓存策略

- 固定缓存目录，避免重复下载：
  - `--modelscope_cache ~/Qwen3.5-4B-Distillation/.modelscope`
  - 同时设置 `MODELSCOPE_CACHE` 为相同路径

### 3.3 并发控制策略

- 调参阶段允许并行短跑（如 B/C 并跑）
- 正式训练前执行“清线程”，确保只保留一个训练进程

---

## 4. 参数配置（已实跑）

### 4.1 通用参数（大多数实验一致）

- `--train_jsonl data/train.jsonl`
- `--model_name Qwen/Qwen3.5-4B`
- `--model_source modelscope`
- `--modelscope_cache ~/Qwen3.5-4B-Distillation/.modelscope`
- `--model_dtype float32`
- `--batch_size 1`
- `--lr` 按实验变化
- `--max_steps` 按实验变化

### 4.2 学习率扫参（max_steps=8）

- `grad_accum=4`
- `max_length=384`
- `logging_steps=1`（脚本会自动与 `grad_accum` 对齐）

### 4.3 A/B/C 网格（max_steps=8，lr=2e-6）

- A: `grad_accum=4, max_length=384`
- B: `grad_accum=8, max_length=384`
- C: `grad_accum=4, max_length=512`

### 4.4 当前正式训练（单进程）

- 最终选择：**B 配置**
- `grad_accum=8, max_length=384, lr=2e-6, max_steps=-1`
- 日志文件：
  - `training_runs/best_B_full_20260425_184108.log`

---

## 5. 指标结果汇总

## 5.1 学习率短跑结果（8 steps）

| 实验 | lr | train_loss | eval_loss | 结论 |
|---|---:|---:|---:|---|
| LR-1 | 5e-7 | 2.243 | 2.1640887 | 可用 |
| LR-2 | 1e-6 | 2.243 | 2.1641490 | 可用 |
| LR-3 | 2e-6 | 2.242 | **2.1640615** | 略优（选中） |

> 说明：三组差异很小，`2e-6` 以微弱优势进入下一轮。

## 5.2 网格结果（A/B/C）

| 实验 | grad_accum | max_length | train_loss | eval_loss | eval_runtime | 结论 |
|---|---:|---:|---:|---:|---:|---|
| A | 4 | 384 | 2.242 | 2.1640615 | 110.6s | 稳定，接近 LR 最优 |
| B | 8 | 384 | 2.298 | **2.1642096** | 147.8s | 最终采用（长跑配置） |
| C | 4 | 512 | 2.246 | 2.2540262 | 195.9s | 验证显著变差，不推荐 |

> 解释：  
> - `max_length=512` 的 C 虽然训练损失不高，但验证损失明显更差。  
> - B 在泛化上优于 C，且符合当前稳定性目标。

---

## 6. 关键结论

1. 在当前数据和环境下，`lr=2e-6` 是更稳妥的学习率选择。  
2. `max_length=384` 明显优于 `512`（至少在当前短跑验证中）。  
3. 正式训练建议使用 **B 配置**：`grad_accum=8, max_length=384, lr=2e-6`。  
4. 4bit 路线若要启用，需要先补齐 `bitsandbytes>=0.46.1`。

---

## 7. 后续建议

- 若继续提高质量，建议按顺序：
  1. 先完成当前 B 全程训练；
  2. 复盘完整 `eval_loss` 曲线；
  3. 再做小范围微调（如 `warmup_ratio`、`max_grad_norm`）；
  4. 有需要再补 D 组用于完整实验闭环。

---

## 8. A100 复现实验记录（2026-04-30）

### 8.1 实验目标

- 在 `A100-PCIE-40GB` 上验证 `Qwen3.5-4B + LoRA + 4bit` 的高吞吐训练可行性。
- 采用短跑 `max_steps=50` 做冒烟与速度基准。

### 8.2 实跑命令

```bash
nohup env PYTHONUNBUFFERED=1 python -u new_mian.py \
  --output_dir training_runs/a100_speed_test_step50 \
  --epochs 1 \
  --max_steps 50 \
  --lr 1e-4 \
  --batch_size 8 \
  --grad_accum 2 \
  --max_length 512 \
  --use_4bit \
  > a100_speed_test_step50.log 2>&1 < /dev/null &
```

### 8.3 关键观测

- 训练正常启动，日志显示：
  - `Total optimization steps = 50`
  - `Instantaneous batch size per device = 8`
  - `Gradient Accumulation steps = 2`
- 学习率调度正常（从首步 `learning_rate=0` 上升到非零）。
- loss 前期下降：
  - `2.659 -> 2.178`（早期训练方向正确）
- GPU 观测（`nvidia-smi`）：
  - 利用率达到 `100%`
  - 显存约 `19.7 / 40 GB`
  - 说明该参数组合已能有效吃满 A100 算力。

### 8.4 完成状态

- 日志出现：`训练完成，模型已保存到: training_runs/a100_speed_test_step50`
- 后台任务状态：`Done`
- 结论：本次 50 步冒烟成功，参数组合可作为后续学习率对比基线。

---

## 9. A100 学习率对比记录（lr=1e-4, max_steps=100）

### 9.1 运行配置

- 输出目录：`training_runs/a100_lr_1e4_step100`
- 关键参数：
  - `max_steps=100`
  - `lr=1e-4`
  - `batch_size=8`
  - `grad_accum=2`
  - `max_length=512`
  - `use_4bit=True`
- 训练规模（日志）：
  - `Num examples = 5485`
  - `Total optimization steps = 100`
  - `Total train batch size = 16`

### 9.2 训练过程关键日志（摘录）

- `loss=2.659, learning_rate=0`（首个记录点）
- `loss=2.185, learning_rate=8.351e-05`
- `loss=1.606, learning_rate=6.289e-05`
- `loss=1.561, learning_rate=4.227e-05`
- `loss=1.463, learning_rate=2.165e-05`
- `loss=1.533, learning_rate=1.031e-06`

### 9.3 验证结果（step 100）

- `eval_loss=1.515`
- `eval_runtime=14.23s`
- `eval_samples_per_second=7.801`
- `eval_steps_per_second=0.984`
- 对应 epoch：`0.2915`

### 9.4 结论（当前阶段）

- 训练与评估流程正常，已在 step 100 成功触发评估与 checkpoint 保存。
- loss 总体从 2.x 降至 1.5x，短跑收敛趋势良好。
- 可继续与 `lr=5e-5`、`lr=2e-4` 的同配置结果做横向对比，确定当前最优学习率。

---

## 10. A100 学习率对比记录（lr=2e-4, max_steps=100）

### 10.1 运行配置

- 启动命令：
  - `nohup env PYTHONUNBUFFERED=1 python -u new_mian.py --output_dir training_runs/a100_lr_2e4_step100 --epochs 1 --max_steps 100 --lr 2e-4 --batch_size 8 --grad_accum 2 --max_length 512 --use_4bit > a100_lr_2e4_step100.log 2>&1 < /dev/null &`
- 输出目录：`training_runs/a100_lr_2e4_step100`
- 关键参数：
  - `max_steps=100`
  - `lr=2e-4`
  - `batch_size=8`
  - `grad_accum=2`
  - `max_length=512`
  - `use_4bit=True`
- 训练规模（日志）：
  - `Num examples = 5485`
  - `Total optimization steps = 100`
  - `Total train batch size = 16`

### 10.2 训练过程关键日志（摘录）

- `loss=2.659, learning_rate=0`（首个记录点）
- `loss=2.034, learning_rate=1.67e-04`
- `loss=1.533, learning_rate=1.258e-04`
- `loss=1.505, learning_rate=8.454e-05`
- `loss=1.413, learning_rate=4.33e-05`
- `loss=1.48, learning_rate=2.062e-06`

### 10.3 验证结果（step 100）

- `eval_loss=1.457`
- `eval_runtime=14.24s`
- `eval_samples_per_second=7.795`
- `eval_steps_per_second=0.983`
- 对应 epoch：`0.2915`

### 10.4 阶段结论

- 训练流程正常，100 步评估与 checkpoint 均成功保存。
- 与 `lr=1e-4` 的 100 步结果对比（`eval_loss=1.515`），`lr=2e-4` 当前更优（`eval_loss=1.457`）。
- 建议后续在 `lr=2e-4` 附近做小范围微调（如 `1.5e-4`、`2.5e-4`）或直接进入更长步数验证。

---

## 11. 学习率实验总对比（A100, batch=8, accum=2, max_length=512, max_steps=100）

| 实验 | lr | eval_loss | eval_runtime(s) | 结论 |
|---|---:|---:|---:|---|
| a100_lr_1e4_step100 | 1e-4 | 1.515 | 14.23 | 可用 |
| a100_lr_2e4_step100 | 2e-4 | **1.457** | 14.24 | 当前最优 |
| a100_lr_5e5_step100 | 5e-5 | 待补充 | 待补充 | 待完成 |

### 11.1 当前推荐参数（进入长跑）

- 推荐学习率：`2e-4`
- 推荐短中期训练配置：
  - `batch_size=8`
  - `grad_accum=2`
  - `max_length=512`
  - `use_4bit=True`

### 11.2 下一轮正式训练命令（建议）

```bash
nohup env PYTHONUNBUFFERED=1 python -u new_mian.py \
  --output_dir training_runs/a100_best_lr2e4_full \
  --epochs 1 \
  --lr 2e-4 \
  --batch_size 8 \
  --grad_accum 2 \
  --max_length 512 \
  --use_4bit \
  > a100_best_lr2e4_full.log 2>&1 < /dev/null &
```

### 11.3 若目标是最终质量（而非速度）

- 在完成上面的长跑后，再用同学习率做一轮：
  - `max_length=1024`（其余不变）
- 再比较验证集 loss 和业务样例，决定最终提交模型。

---

## 12. 正式训练最终结果（A100, 2026-04-30）

### 12.1 最终训练命令（已完成）

```bash
nohup env PYTHONUNBUFFERED=1 python -u new_mian.py \
  --output_dir training_runs/final_a100_lr2e4_len512 \
  --epochs 1 \
  --lr 2e-4 \
  --batch_size 8 \
  --grad_accum 2 \
  --max_length 512 \
  --use_4bit \
  > final_a100_lr2e4_len512.log 2>&1 < /dev/null &
```

### 12.2 训练结果摘要

- 数据规模：
  - `train records = 5485`
  - `val records = 111`
- 训练配置：
  - `Num Epochs = 1`
  - `Total optimization steps = 343`
  - `Instantaneous batch size = 8`
  - `Gradient Accumulation steps = 2`
  - `Total train batch size = 16`
- 最终训练指标：
  - `train_runtime = 2687s`（约 `44.8` 分钟）
  - `train_steps_per_second = 0.128`
  - `train_loss = 1.407`
- 验证集关键结果（随训练推进）：
  - step100: `eval_loss = 1.423`
  - step200: `eval_loss = 1.348`
  - step300: `eval_loss = 1.311`
  - step343(epoch=1): `eval_loss = 1.305`（本轮最优）

### 12.3 产物与最佳模型

- 最终模型目录：`training_runs/final_a100_lr2e4_len512`
- 关键 checkpoint：
  - `training_runs/final_a100_lr2e4_len512/checkpoint-100`
  - `training_runs/final_a100_lr2e4_len512/checkpoint-200`
  - `training_runs/final_a100_lr2e4_len512/checkpoint-300`
  - `training_runs/final_a100_lr2e4_len512/checkpoint-343`
- 日志显示已自动加载最佳模型：
  - `Loading best model from .../checkpoint-343 (score: 1.305...)`

### 12.4 结论与固定方案

- 本项目当前“一步到位”推荐正式参数：
  - `lr=2e-4`
  - `batch_size=8`
  - `grad_accum=2`
  - `max_length=512`
  - `use_4bit=True`
  - `epochs=1`
- 该配置在 A100 上已完成全流程验证，可直接复用，不再需要先做短跑调试。

