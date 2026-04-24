#!/usr/bin/env python3
"""
Qwen3.5-4B 响应蒸馏训练脚本。

支持两种数据入口：
1) JSONL（推荐）：每行一个样本，字段为 instruction/content/output
2) Excel（兼容）：读取历史 training_data_4B.xlsx
"""

import argparse
import json
import os
import random
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import xml.etree.ElementTree as ET

import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


@dataclass
class DistillExample:
    instruction: str
    input_text: str
    output_text: str


def _read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    shared = []
    for si in root.findall(f"{XML_NS}si"):
        texts = [t.text or "" for t in si.iter(f"{XML_NS}t")]
        shared.append("".join(texts))
    return shared


def _read_cell_value(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t")
    v = cell.find(f"{XML_NS}v")
    if v is None or v.text is None:
        return ""
    if cell_type == "s":
        idx = int(v.text)
        return shared_strings[idx] if idx < len(shared_strings) else ""
    return v.text


def load_excel_rows(xlsx_path: str) -> List[Dict[str, str]]:
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = _read_shared_strings(zf)
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows = sheet.find(f"{XML_NS}sheetData").findall(f"{XML_NS}row")
    if not rows:
        return []

    headers = [
        _read_cell_value(c, shared_strings).strip()
        for c in rows[0].findall(f"{XML_NS}c")
    ]
    output = []
    for row in rows[1:]:
        values = [_read_cell_value(c, shared_strings) for c in row.findall(f"{XML_NS}c")]
        if not any(values):
            continue
        row_dict = {}
        for i, key in enumerate(headers):
            row_dict[key] = values[i] if i < len(values) else ""
        output.append(row_dict)
    return output


def build_examples(rows: Iterable[Dict[str, str]]) -> List[DistillExample]:
    instruction = (
        "你是美妆品牌识别助手。请从文本中抽出所有的品牌 mention，"
        "并返回 JSON：brand, brand_text, start, end, is_beauty, reasoning。"
    )

    examples: List[DistillExample] = []
    for row in rows:
        # 优先使用 content，新老字段兼容（content -> clean_text）。
        content = (row.get("content") or row.get("clean_text") or "").strip()
        brand = (row.get("brand") or "").strip()
        brand_text = (row.get("brand_text") or "").strip()
        reasoning = (row.get("reasoning") or "").strip()
        start = str(row.get("start") or "").strip()
        end = str(row.get("end") or "").strip()
        is_beauty = str(row.get("is_beauty") or "").strip()

        if not content or not brand:
            continue

        target = {
            "brand": brand,
            "brand_text": brand_text,
            "start": start,
            "end": end,
            "is_beauty": is_beauty,
            "reasoning": reasoning,
        }
        examples.append(
            DistillExample(
                instruction=instruction,
                input_text=content,
                output_text=json.dumps(target, ensure_ascii=False),
            )
        )
    return examples


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


def build_examples_from_json(records: Iterable[Dict]) -> List[DistillExample]:
    examples: List[DistillExample] = []
    for row in records:
        instruction = str(row.get("instruction") or "").strip()
        content = str(row.get("content") or "").strip()
        output = row.get("output")
        if not instruction or not content or output is None:
            continue
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
    def _tokenize(batch):
        texts = []
        for ins, inp, out in zip(
            batch["instruction"],
            batch["input_text"],
            batch["output_text"],
        ):
            chat_text = format_chat(
                DistillExample(ins, inp, out),
                tokenizer=tokenizer,
            )
            texts.append(chat_text)
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return tokenized

    return _tokenize


def create_model(model_name: str, use_4bit: bool):
    quant_config: Optional[BitsAndBytesConfig] = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.5-4B 蒸馏训练")
    parser.add_argument(
        "--xlsx_path",
        type=str,
        default="training_data_4B.xlsx",
        help="Excel 训练数据路径",
    )
    parser.add_argument(
        "--train_jsonl",
        type=str,
        default="",
        help="训练集 JSONL 路径（推荐，设置后优先于 xlsx）",
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
        "--output_dir",
        type=str,
        default="./distilled-qwen",
        help="输出目录",
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[1/5] 读取数据")
    if args.train_jsonl:
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
    else:
        if not os.path.exists(args.xlsx_path):
            raise FileNotFoundError(f"找不到数据文件: {args.xlsx_path}")
        rows = load_excel_rows(args.xlsx_path)
        print(f"  - excel rows: {len(rows)}")
        examples = build_examples(rows)
        dataset = split_dataset(
            examples=examples,
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
    model = create_model(args.model_name, use_4bit=args.use_4bit)
    model.print_trainable_parameters()

    print("[4/5] Tokenize")
    tokenized = dataset.map(
        tokenize_fn(tokenizer=tokenizer, max_length=args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print("[5/5] 开始训练")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=20,
        warmup_ratio=0.03,
        bf16=False,
        fp16=False,
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
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"训练完成，模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
