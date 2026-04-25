"""
用途:
- 对 `remote_eval_20.py` 的失败样本做定向诊断，输出“失败类型 + 失败原因 + 输出片段”。

评估逻辑:
1) 从 `data/train.jsonl` 抽样 N 条（默认 20，seed 固定为 42）。
2) 逐条推理后先做 JSON 解析:
   - 解析失败 => `json_parse_fail`
3) 若可解析，再做 schema 检查:
   - 不符合 => `schema_fail` + 具体原因（如缺字段/类型不匹配）
4) 最终只输出失败样本摘要，便于快速定位问题模式。

运行方式:
- 云端: `python remote_eval_failures.py`
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
    """读取 JSONL 数据，每行一个样本。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def valid_output_schema(obj):
    """返回 (是否通过, 失败原因) 以便定位 schema 失败类型。"""
    if not isinstance(obj, dict):
        return False, "top_level_not_object"
    for k in ("is_beauty", "reasoning", "relationships"):
        if k not in obj:
            return False, f"missing_{k}"
    if not isinstance(obj["is_beauty"], bool):
        return False, "is_beauty_not_bool"
    if not isinstance(obj["reasoning"], str):
        return False, "reasoning_not_str"
    if not isinstance(obj["relationships"], list):
        return False, "relationships_not_list"
    for i, rel in enumerate(obj["relationships"]):
        if not isinstance(rel, dict):
            return False, f"rel_{i}_not_object"
        for k in ("brand_text", "start", "end"):
            if k not in rel:
                return False, f"rel_{i}_missing_{k}"
            if not isinstance(rel[k], str):
                return False, f"rel_{i}_{k}_not_str"
    return True, "ok"


def main():
    # 固定抽样，便于不同版本模型做同集对比。
    random.seed(SEED)
    rows = load_data(DATA)
    samples = random.sample(rows, min(N, len(rows)))

    tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    fails = []
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

        try:
            parsed = json.loads(txt)
        except Exception as exc:
            # 第一类失败: 输出不是合法 JSON。
            fails.append(
                {
                    "idx": i,
                    "type": "json_parse_fail",
                    "error": str(exc),
                    "input_preview": row["content"][:100],
                    "output_preview": txt[:500],
                }
            )
            continue

        ok, reason = valid_output_schema(parsed)
        if not ok:
            # 第二类失败: JSON 可解析，但不符合目标 schema。
            fails.append(
                {
                    "idx": i,
                    "type": "schema_fail",
                    "reason": reason,
                    "input_preview": row["content"][:100],
                    "output_preview": txt[:500],
                }
            )

    print("=== FAIL SUMMARY ===")
    print(f"failed={len(fails)}/{len(samples)}")
    for item in fails:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
