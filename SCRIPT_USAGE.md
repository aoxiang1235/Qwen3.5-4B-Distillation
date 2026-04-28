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

