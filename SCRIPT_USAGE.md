# 脚本作用说明（Qwen3.5-4B-Distillation）

本文档汇总当前项目里主要脚本/日志文件的用途，便于快速判断“该用哪个文件做什么”。

## 一、核心脚本

- `new_mian.py`
  - 作用：主训练脚本（LoRA 微调入口）。
  - 典型用途：启动训练、调参（学习率、batch、max_length 等）、产出训练目录。

- `prepare_data.py`
  - 作用：训练数据预处理脚本。
  - 典型用途：将原始数据整理为模型训练可用格式（如指令+内容+标签结构）。

- `serve_qwen35_full_http.py`
  - 作用：全模型（merged）HTTP 推理服务（FP16 路径）。
  - 典型用途：提供 `/health` 与 `/generate` 接口，做结构化抽取推理。
  - 特点：偏“全精度服务”，格式控制能力比最简版脚本更强。

- `serve_qwen35_full_http_4bit.py`
  - 作用：全模型 HTTP 推理服务的 4bit/量化版本。
  - 典型用途：显存受限环境下运行推理服务（如 P100 机器）。
  - 特点：更省显存，但输出质量/稳定性可能弱于全精度。
  - 备注：脚本名含「4bit」，但在 **非 P100** GPU 上默认走 **BitsAndBytes 8bit**；**P100** 上会 **自动回退 NF4 4bit**（避免部分环境下 int8 不稳定）。

- `serve_qwen35_full_http_8bit.py`
  - 作用：全模型 HTTP 推理服务，**优先 BitsAndBytes INT8**（`load_in_8bit`），与 vLLM **无关**。
  - 典型用途：本机或云上 **Transformers + bnb** 快速试 8bit 显存占用；接口仍为 `/health`、`/generate`（默认端口 **8014**，避免与 `serve_qwen35_full_http_4bit.py` 默认 8013 冲突）。
  - 特点：**P100** 上同样会 **回退 4bit**；若线上统一用 **vLLM OpenAI 接口**，请使用合并后的权重目录 + **vLLM** 启动，而不是本脚本。

- `quantize_awq_4bit.py`
  - 作用：将 **已 merge 的 FP16/BF16** HuggingFace 目录 **离线量化为 AWQ 4-bit**，产出可直接给 **vLLM `--quantization awq`** 加载的目录。
  - 典型用途：云上节省显存、与 `bench_train_v2_vllm.py` 做吞吐评测。
  - 依赖：`autoawq`；校准文本来自 JSONL 的 `instruction`+`content`（与 bench 拼接方式一致）。

- `quantize_gptq_8bit.py`
  - 作用：将 **已 merge 的 FP16/BF16** 目录 **离线量化为 GPTQ（默认 8-bit）**，产出给 **vLLM `--quantization gptq`**（或日志提示的 `gptq_marlin` 等）使用。
  - 典型用途：需要 **静态 8bit 权重 + vLLM** 时（与 BitsAndBytes 运行时 8bit 不是同一路）。
  - 后端：`--backend auto` 时 **优先 `gptqmodel`**，否则 **`auto_gptq`**（云上若 `auto-gptq` 编译失败，通常只装 `gptqmodel` 即可）。
  - 校准：同样使用 JSONL 的 `instruction`+`content`。

- `bench_train_v2_vllm.py`
  - 作用：遍历 JSONL（默认 `data/val.jsonl`），按 **OpenAI Chat Completions** 格式调用 **vLLM**（如 `/v1/chat/completions`），把 **耗时、post_id、第三列数据集 `output`、第四列模型正文** 写入 TSV 日志。
  - 典型用途：对比 **FP16 / AWQ / GPTQ** 等不同部署在同一验证集上的稳定性与耗时。
  - 常用参数：`--url`、`--model`（与 `--served-model-name` 一致）、`--log`、`--skip-health`（跳过 GET `/health`）。

- `merge_lora_weights.py`
  - 作用：将 LoRA 适配器 **merge** 进基座，得到 **可直接推理或再做 AWQ/GPTQ** 的全量权重目录。
  - 典型用途：量化脚本与 vLLM 的 `--model` 一般需要 **merged FP16** 目录而非仅 LoRA。

- 直接启动命令（替代启动脚本）
  - 作用：不依赖 `Qwen3.5-4b-lora-start.sh`，手工启动服务并做健康检查。
  - 典型用途：脚本丢失或跨机器迁移时快速拉起服务。
  - 示例（4bit 服务，端口 8012）：
    - `cd ~/Qwen3.5-4B-Distillation-github-smoke`
    - `source ~/Qwen3.5-4B-Distillation/.venv/bin/activate`
    - `lsof -ti tcp:8012 | xargs -r kill -9`
    - `nohup env PYTHONUNBUFFERED=1 python3 -u serve_qwen35_full_http_4bit.py --host 0.0.0.0 --port 8012 --model_path training_runs/best_B_full_20260425_184108_merged --max_new_tokens 180 > qwen35_quant_http_8012.log 2>&1 < /dev/null &`
    - `curl -sS http://127.0.0.1:8012/health`

## 二、数据与模板相关文件

- `training_data_template.json`
  - 作用：训练数据结构模板示例。
  - 典型用途：约束/参考训练样本字段格式。

- `prompt_template.txt`
  - 作用：提示词模板。
  - 典型用途：训练或推理时复用统一提示风格。

- `data/`
  - 作用：训练/评估数据目录。
  - 典型用途：存放 `train.jsonl` 等核心样本。

## 三、日志与记录文件

- `train.log`
  - 作用：训练过程日志（主日志）。
  - 典型用途：排查训练报错、观察 loss、步数、保存点。

- `train_f32_full_20260424.log`
  - 作用：某次特定训练实验日志（f32_full 方案）。
  - 典型用途：复盘历史实验参数与结果。

- `TRAINING_DEBUG_LOG.md`
  - 作用：调试过程记录文档。
  - 典型用途：沉淀问题-原因-修复方案。

- `TRAINING_PLAN_AND_RESULTS.md`
  - 作用：训练计划与结果总结文档。
  - 典型用途：作业/汇报时展示实验过程、参数、指标。

- `val_awq_bench.log` / `val_gptq8_bench.log`（示例文件名）
  - 作用：**`bench_train_v2_vllm.py`** 跑验证集得到的 **TSV 结果**（首列耗时秒、第二列 `post_id`、第三列标注 `output`、第四列模型输出 JSON 字符串）。
  - 典型用途：量化模型（AWQ / GPTQ）与基线对比；可与对应的 `*_nohup.log` 对照进程输出。
  - 说明：行数通常与 `data/val.jsonl` 中 **同时具备 instruction 与 content** 的样本条数一致（例如 111 条）。

## 四、其他文件

- `APITest.text`
  - 作用：接口测试相关命令/样例记录。
  - 典型用途：复现实验请求、人工测试接口返回。

- `training_data_4B.xlsx`
  - 作用：原始或中间训练数据表格。
  - 典型用途：人工审核与数据整理。

## 五、建议使用顺序（实操）

1. 先看 `data/` 与 `training_data_template.json`，确认数据结构。
2. 用 `prepare_data.py` 做数据准备。
3. 用 `new_mian.py` 启动训练。
4. 训练完成后优先用 `serve_qwen35_full_http.py`（或资源紧张时 `serve_qwen35_full_http_4bit.py`）部署。
5. 用“直接启动命令”拉起服务并通过 `/health` 做健康检查。
6. 用 `APITest.text` 做接口验证，用 `train.log`/`TRAINING_*.md` 做复盘与提交材料。

### 合并权重 → 离线量化 → vLLM → bench（可选管线）

1. 使用 `merge_lora_weights.py` 得到 **merged FP16** 目录。
2. 任选：`quantize_awq_4bit.py`（AWQ 4bit）或 `quantize_gptq_8bit.py`（GPTQ 8bit），在云 GPU 上生成量化目录。
3. 使用 **`python3 -m vllm.entrypoints.openai.api_server`**（参数见各量化脚本文件头注释）启动服务。
4. 使用 `bench_train_v2_vllm.py` 指向 `http://<主机>:<端口>/v1/chat/completions` 与 `--model <served-model-name>`，生成 `*.bench.log` 便于对比。

### 仅需 Transformers + 省显存（不做 vLLM）

- 使用 `serve_qwen35_full_http_8bit.py`（或 `serve_qwen35_full_http_4bit.py`），按脚本默认端口与 `--model_path` 指向 merged 目录。

