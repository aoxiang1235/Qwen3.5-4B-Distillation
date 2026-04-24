#!/usr/bin/env python3
"""
Qwen3.5-4B 响应蒸馏训练脚本。

仅支持 JSONL 数据入口：
1) 每行一个样本，字段为 instruction/content/output
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

@dataclass
class DistillExample:
    instruction: str
    input_text: str
    output_text: str


def load_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败: {path}:{line_no} ({exc})") from exc
    return records


def _validate_output_schema(output: object, row_idx: int) -> None:
    if not isinstance(output, dict):
        raise ValueError(f"第 {row_idx} 条样本 output 必须是 JSON object。")

    required_keys = {"is_beauty", "reasoning", "relationships"}
    missing = required_keys - set(output.keys())
    if missing:
        raise ValueError(
            f"第 {row_idx} 条样本 output 缺少字段: {sorted(missing)}。"
        )

    if not isinstance(output["is_beauty"], bool):
        raise ValueError(f"第 {row_idx} 条样本 output.is_beauty 必须是 boolean。")
    if not isinstance(output["reasoning"], str):
        raise ValueError(f"第 {row_idx} 条样本 output.reasoning 必须是 string。")
    if not isinstance(output["relationships"], list):
        raise ValueError(f"第 {row_idx} 条样本 output.relationships 必须是 array。")

    for rel_idx, rel in enumerate(output["relationships"]):
        if not isinstance(rel, dict):
            raise ValueError(
                f"第 {row_idx} 条样本 output.relationships[{rel_idx}] 必须是 object。"
            )
        for k in ("brand_text", "start", "end"):
            if k not in rel:
                raise ValueError(
                    f"第 {row_idx} 条样本 output.relationships[{rel_idx}] 缺少字段 {k}。"
                )
            if not isinstance(rel[k], str):
                raise ValueError(
                    f"第 {row_idx} 条样本 output.relationships[{rel_idx}].{k} 必须是 string。"
                )


def build_examples_from_json(records: Iterable[Dict]) -> List[DistillExample]:
    examples: List[DistillExample] = []
    for row_idx, row in enumerate(records, start=1):
        instruction = str(row.get("instruction") or "").strip()
        content = str(row.get("content") or "").strip()
        output = row.get("output")
        if not instruction or not content or output is None:
            continue
        _validate_output_schema(output, row_idx)
        examples.append(
            DistillExample(
                instruction=instruction,
                input_text=content,
                output_text=json.dumps(output, ensure_ascii=False),
            )
        )
    return examples


def format_chat(example: DistillExample, tokenizer) -> str:
    # Qwen chat 模板，训练目标放在 assistant 回复里。
    messages = [
        {"role": "system", "content": "你是一个严谨的结构化信息抽取助手。"},
        {
            "role": "user",
            "content": f"指令：{example.instruction}\n\n文本：{example.input_text}",
        },
        {"role": "assistant", "content": example.output_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


@dataclass
class DataCollatorForCausalLMCustom:
    """batch 内 padding；仅将 padding（attention_mask==0）标为 -100，避免 pad==eos 时误 mask。"""

    tokenizer: Any

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        input_ids_t = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
        attention_mask_t = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels_t = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        labels_t = labels_t.masked_fill(attention_mask_t == 0, -100)
        return {
            "input_ids": input_ids_t,
            "attention_mask": attention_mask_t,
            "labels": labels_t,
        }


def _lcp_token_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def split_dataset(
    examples: List[DistillExample],
    val_ratio: float = 0.02,
    seed: int = 42,
) -> DatasetDict:
    random.seed(seed)
    random.shuffle(examples)
    val_size = max(1, int(len(examples) * val_ratio))
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    train_ds = Dataset.from_list([e.__dict__ for e in train_examples])
    val_ds = Dataset.from_list([e.__dict__ for e in val_examples])
    return DatasetDict({"train": train_ds, "validation": val_ds})


def tokenize_fn(tokenizer, max_length: int):
    system_content = "你是一个严谨的结构化信息抽取助手。"

    def _tokenize(batch):
        input_ids_list: List[List[int]] = []
        attention_mask_list: List[List[int]] = []
        labels_list: List[List[int]] = []

        for ins, inp, out in zip(
            batch["instruction"],
            batch["input_text"],
            batch["output_text"],
        ):
            ex = DistillExample(ins, inp, out)
            messages_full = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": f"指令：{ex.instruction}\n\n文本：{ex.input_text}",
                },
                {"role": "assistant", "content": ex.output_text},
            ]
            messages_prompt = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": f"指令：{ex.instruction}\n\n文本：{ex.input_text}",
                },
            ]
            full_text = tokenizer.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_ids = tokenizer.apply_chat_template(
                messages_prompt,
                tokenize=True,
                add_generation_prompt=True,
            )
            full_no_trunc = tokenizer(full_text, truncation=False)["input_ids"]
            enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            ids = enc["input_ids"]
            attn = enc["attention_mask"]
            if len(full_no_trunc) >= len(prompt_ids) and full_no_trunc[
                : len(prompt_ids)
            ] == prompt_ids:
                assistant_start_full = len(prompt_ids)
            else:
                assistant_start_full = _lcp_token_len(prompt_ids, full_no_trunc)
            cut = max(0, len(full_no_trunc) - len(ids))
            assistant_start_in_seq = max(0, assistant_start_full - cut)

            labels = list(ids)
            for i in range(min(assistant_start_in_seq, len(labels))):
                labels[i] = -100

            input_ids_list.append(ids)
            attention_mask_list.append(attn)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    return _tokenize


def create_model(model_name: str, use_4bit: bool, model_dtype: str):
    quant_config: Optional[BitsAndBytesConfig] = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if model_dtype not in dtype_map:
        raise ValueError(f"不支持的 --model_dtype: {model_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_map[model_dtype],
        quantization_config=quant_config,
        device_map="auto",
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model


def _aligned_logging_steps(requested: int, grad_accum: int) -> int:
    """与 gradient_accumulation 对齐，减少 Trainer 在累积边界上打出 loss=0 的误导日志。"""
    if requested > 0:
        base = max(1, requested)
    else:
        base = 20
    ga = max(1, grad_accum)
    aligned = ((base + ga - 1) // ga) * ga
    return max(ga, aligned)


def _filter_weak_supervision(ds, min_tokens: int, split_name: str):
    """去掉 labels 全为 -100 或有效 token 过少的样本。"""

    def _ok(example: Dict[str, List[int]]) -> bool:
        return sum(1 for t in example["labels"] if t != -100) >= min_tokens

    n_before = len(ds)
    out = ds.filter(_ok)
    n_after = len(out)
    print(f"  - {split_name}: 过滤弱监督样本 {n_before} -> {n_after} (min_supervised_tokens={min_tokens})")
    if n_after < 10:
        raise RuntimeError(
            f"{split_name} 过滤后样本不足 10 条，请降低 --min_supervised_tokens 或检查截断/模板。"
        )
    return out


def _print_supervision_stats(ds, split_name: str):
    counts = [sum(1 for t in row["labels"] if t != -100) for row in ds]
    if not counts:
        raise RuntimeError(f"{split_name} 没有可用样本。")
    counts_sorted = sorted(counts)
    n = len(counts_sorted)
    p50 = counts_sorted[n // 2]
    p90 = counts_sorted[min(n - 1, int(n * 0.9))]
    min_v = counts_sorted[0]
    max_v = counts_sorted[-1]
    mean_v = sum(counts_sorted) / n
    print(
        f"  - {split_name} 监督 token 统计: min={min_v}, p50={p50}, p90={p90}, max={max_v}, mean={mean_v:.1f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.5-4B 蒸馏训练")
    parser.add_argument(
        "--train_jsonl",
        type=str,
        default="",
        help="训练集 JSONL 路径（必填）",
    )
    parser.add_argument(
        "--val_jsonl",
        type=str,
        default="",
        help="验证集 JSONL 路径（可选，不传则按 val_ratio 从 train 切分）",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Student 模型名（建议换成你的 Qwen3.5-4B 路径）",
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        choices=["float16", "float32"],
        default="float32",
        help="模型权重加载精度；P100 上为避免 loss 数值爆炸，默认用 float32",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./distilled-qwen",
        help="输出目录",
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="仅训练若干 step 即停（云端冒烟用）；-1 表示按 num_train_epochs 跑满",
    )
    parser.add_argument(
        "--min_supervised_tokens",
        type=int,
        default=2,
        help="每条样本至少要有多少个非 -100 的 label token，否则丢弃（避免 loss=0 / 反传异常）",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=0,
        help="日志步长；0 表示自动取不小于 20 且为 grad_accum 倍数的值，与梯度累积对齐",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="梯度裁剪阈值；默认 1.0 以提升数值稳定性",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[1/5] 读取数据")
    if not args.train_jsonl:
        raise ValueError("已禁用 Excel 入口，请传 --train_jsonl。")
    if not os.path.exists(args.train_jsonl):
        raise FileNotFoundError(f"找不到训练集 JSONL: {args.train_jsonl}")
    train_records = load_jsonl(args.train_jsonl)
    print(f"  - train records: {len(train_records)}")
    train_examples = build_examples_from_json(train_records)

    if args.val_jsonl:
        if not os.path.exists(args.val_jsonl):
            raise FileNotFoundError(f"找不到验证集 JSONL: {args.val_jsonl}")
        val_records = load_jsonl(args.val_jsonl)
        print(f"  - val records: {len(val_records)}")
        val_examples = build_examples_from_json(val_records)
        dataset = DatasetDict(
            {
                "train": Dataset.from_list([e.__dict__ for e in train_examples]),
                "validation": Dataset.from_list([e.__dict__ for e in val_examples]),
            }
        )
    else:
        dataset = split_dataset(
            examples=train_examples,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    total_examples = len(dataset["train"]) + len(dataset["validation"])
    if total_examples < 100:
        raise RuntimeError("可用样本过少，建议先检查字段映射或数据质量。")

    print("[2/5] 数据集准备完成")
    print(
        f"  - train: {len(dataset['train'])}, validation: {len(dataset['validation'])}"
    )

    print(f"[3/5] 加载 tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    print(f"  - model_dtype: {args.model_dtype}")
    model = create_model(
        args.model_name,
        use_4bit=args.use_4bit,
        model_dtype=args.model_dtype,
    )
    model.print_trainable_parameters()

    print("[4/5] Tokenize")
    tokenized = dataset.map(
        tokenize_fn(tokenizer=tokenizer, max_length=args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    tokenized["train"] = _filter_weak_supervision(
        tokenized["train"], args.min_supervised_tokens, "train"
    )
    tokenized["validation"] = _filter_weak_supervision(
        tokenized["validation"], args.min_supervised_tokens, "validation"
    )
    _print_supervision_stats(tokenized["train"], "train")
    _print_supervision_stats(tokenized["validation"], "validation")

    log_steps = _aligned_logging_steps(args.logging_steps, args.grad_accum)
    if args.logging_steps > 0 and log_steps != args.logging_steps:
        print(
            f"  - logging_steps 已从 {args.logging_steps} 调整为 {log_steps}（与 grad_accum={args.grad_accum} 对齐）"
        )

    print("[5/5] 开始训练")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=log_steps,
        warmup_ratio=0.03,
        bf16=False,
        fp16=False,
        max_grad_norm=args.max_grad_norm,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForCausalLMCustom(tokenizer=tokenizer),
    )
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"训练完成，模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
