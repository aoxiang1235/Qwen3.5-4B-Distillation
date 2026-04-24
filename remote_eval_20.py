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
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def valid_output_schema(obj):
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
    random.seed(SEED)
    rows = load_data(DATA)
    samples = random.sample(rows, min(N, len(rows)))

    print("loading model...")
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
            parsed = json.loads(txt)
            ok_json += 1
            if valid_output_schema(parsed):
                ok_schema += 1
        except Exception:
            pass

        if i <= 3:
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
