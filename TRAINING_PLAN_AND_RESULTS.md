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

