"""
用途:
- 对训练好的 LoRA 模型做一个快速结构化输出健康检查（不是最终精度评估）。

评估逻辑:
1) 从 `data/train.jsonl` 随机抽样 N 条（默认 20）。
2) 逐条推理，统计:
   - json_parse_ok: 输出能被 json.loads 解析的比例
   - schema_ok: 输出是否满足目标 schema 的比例
3) 打印前 3 条样例，便于人工检查输出风格与错误模式。

运行方式:
- 云端: `python remote_eval_20.py`
"""

import json
import random

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "distilled-qwen-f32_full_20260424"
DATA = "data/train.jsonl"
N = 20
SEED = 42


def load_data(path):
    """读取 JSONL 样本，每行一个 JSON 对象。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def valid_output_schema(obj):
    """检查模型输出是否满足目标 schema（字段和类型）。"""
    if not isinstance(obj, dict):
        return False
    for k in ("is_beauty", "reasoning", "relationships"):
        if k not in obj:
            return False
    if not isinstance(obj["is_beauty"], bool):
        return False
    if not isinstance(obj["reasoning"], str):
        return False
    if not isinstance(obj["relationships"], list):
        return False
    for rel in obj["relationships"]:
        if not isinstance(rel, dict):
            return False
        for k in ("brand_text", "start", "end"):
            if k not in rel or not isinstance(rel[k], str):
                return False
    return True


def main():
    # 固定随机种子，保证抽样可复现，方便横向比较不同模型版本。
    random.seed(SEED)
    rows = load_data(DATA)
    samples = random.sample(rows, min(N, len(rows)))

    print("loading model...")
    # 加载基座模型 + LoRA 适配器。
    tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    ok_json = 0
    ok_schema = 0
    outputs = []

    for i, row in enumerate(samples, 1):
        messages = [
            {"role": "system", "content": "你是一个严谨的结构化信息抽取助手。"},
            {"role": "user", "content": f"指令：{row['instruction']}\n\n文本：{row['content']}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        txt = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        parsed = None
        try:
            # 一级检查: 输出是否可解析为 JSON。
            parsed = json.loads(txt)
            ok_json += 1
            # 二级检查: JSON 字段是否符合预期 schema。
            if valid_output_schema(parsed):
                ok_schema += 1
        except Exception:
            pass

        if i <= 3:
            # 仅保留少量样例，方便快速人工排查。
            outputs.append(
                {
                    "idx": i,
                    "input_preview": row["content"][:80],
                    "raw_output": txt[:400],
                    "json_ok": parsed is not None,
                    "schema_ok": valid_output_schema(parsed) if parsed is not None else False,
                }
            )

    print("=== EVAL RESULT ===")
    print(f"samples={len(samples)}")
    print(f"json_parse_ok={ok_json}/{len(samples)}")
    print(f"schema_ok={ok_schema}/{len(samples)}")
    print("=== EXAMPLES ===")
    for x in outputs:
        print(json.dumps(x, ensure_ascii=False))


if __name__ == "__main__":
    main()
