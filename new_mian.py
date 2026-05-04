#!/usr/bin/env python3
"""
Qwen3.5-4B 监督微调（SFT）训练脚本，本质是「用 JSONL 里的目标输出」做因果语言模型训练。

训练方法简述：
- 基座：Qwen3.5-4B，默认从魔搭 ModelScope 拉取/复用本地缓存，也可用 --model_source huggingface 或本机目录。
- 只训练 LoRA 适配器（r=16, alpha=32），全参冻结，省显存、不易灾难性遗忘。
- 将 instruction/content 拼成 prompt，只在「助手回复」段计算 loss（其余 token 的 labels 为 -100）；可选梯度检查点降显存。
- HuggingFace Trainer 做反向与日志；nohup 到文件时建议: PYTHONUNBUFFERED=1 python3 -u new_mian.py ...

仅支持 JSONL 数据入口：每行 instruction / content / output 字段。
若本机已有完整基座，把目录传给 --model_name 可跳过下载。
"""

import argparse
import inspect
import json
import os
import random
import sys
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


DEFAULT_SYSTEM_PROMPT = "You are a strict structured information extraction assistant."


def _build_user_content(instruction: str, content: str) -> str:
    return f"Instruction:\n{instruction}\n\nContent:\n{content}"


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
        if "brand_text" not in rel:
            raise ValueError(
                f"第 {row_idx} 条样本 output.relationships[{rel_idx}] 缺少字段 brand_text。"
            )
        if not isinstance(rel["brand_text"], str):
            raise ValueError(
                f"第 {row_idx} 条样本 output.relationships[{rel_idx}].brand_text 必须是 string。"
            )


def build_examples_from_json(
    records: Iterable[Dict], instruction_override: str = ""
) -> List[DistillExample]:
    examples: List[DistillExample] = []
    override = instruction_override.strip()
    for row_idx, row in enumerate(records, start=1):
        instruction = str(row.get("instruction") or "").strip()
        content = str(row.get("content") or "").strip()
        output = row.get("output")
        if not instruction or not content or output is None:
            continue
        if override:
            instruction = override
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
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_user_content(example.instruction, example.input_text),
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
    system_content = DEFAULT_SYSTEM_PROMPT

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
                    "content": _build_user_content(ex.instruction, ex.input_text),
                },
                {"role": "assistant", "content": ex.output_text},
            ]
            messages_prompt = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": _build_user_content(ex.instruction, ex.input_text),
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


def create_model(
    model_name: str,
    use_4bit: bool,
    model_dtype: str,
    gradient_checkpointing: bool = True,
):
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
    if gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        print("  - gradient_checkpointing: 已开启（降低显存）")
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


def _filter_weak_supervision(
    ds, min_tokens: int, split_name: str, min_samples_after_filter: int
):
    """去掉 labels 全为 -100 或有效 token 过少的样本。"""

    def _ok(example: Dict[str, List[int]]) -> bool:
        return sum(1 for t in example["labels"] if t != -100) >= min_tokens

    n_before = len(ds)
    out = ds.filter(_ok)
    n_after = len(out)
    print(f"  - {split_name}: 过滤弱监督样本 {n_before} -> {n_after} (min_supervised_tokens={min_tokens})")
    if n_after < min_samples_after_filter:
        raise RuntimeError(
            f"{split_name} 过滤后样本不足 {min_samples_after_filter} 条，请降低 "
            "--min_supervised_tokens / --min_split_samples_after_filter，或检查截断/模板。"
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
        default="data/train.jsonl",
        help="训练集 JSONL 路径（默认 data/train.jsonl）",
    )
    parser.add_argument(
        "--val_jsonl",
        type=str,
        default="data/val_v2.jsonl",
        help="验证集 JSONL 路径（默认 data/val_v2.jsonl；传空字符串则按 val_ratio 从 train 切分）",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="魔搭 / HF 的模型 id，或本机已下载目录（见 --model_source）",
    )
    parser.add_argument(
        "--model_source",
        type=str,
        choices=["modelscope", "huggingface"],
        default="modelscope",
        help="modelscope=从魔搭 ModelScope 下载/复用缓存；huggingface=从 Hugging Face Hub 拉取",
    )
    parser.add_argument(
        "--modelscope_cache",
        type=str,
        default="",
        help="ModelScope 缓存根目录；留空则使用 <cwd>/.modelscope 或环境变量 MODELSCOPE_CACHE",
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
        "--eval_steps",
        type=int,
        default=0,
        help="评估间隔步数；0 表示自动（max_steps>0 时用 max_steps 以便短跑结束前有 eval_loss）",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=0,
        help="保存 checkpoint 间隔；0 表示自动（max_steps>0 时与 eval_steps 一致）",
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
    parser.add_argument(
        "--min_split_samples_after_filter",
        type=int,
        default=10,
        help="每个数据集 split 在弱监督过滤后至少保留的样本数；小样本调试时可设为 1~2",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="关闭梯度检查点（默认开启以降低显存；关闭可略快但易 OOM）",
    )
    parser.add_argument(
        "--instruction_override",
        type=str,
        default="",
        help="若非空，则覆盖 JSONL 中每条样本的 instruction（用于快速替换提示词）",
    )
    return parser.parse_args()


def resolve_pretrained_path(args) -> str:
    """
    将 --model_name 解析为可传给 from_pretrained 的路径或 id。
    - 若本机目录下已有 config.json，则直接使用（不走魔搭 / HF 下载逻辑）。
    - 否则按 --model_source 从魔搭或 HF 取权重。
    """
    name = (args.model_name or "").strip()
    if not name:
        raise ValueError("请设置 --model_name。")
    if os.path.isfile(os.path.join(name, "config.json")):
        return os.path.abspath(name)
    if args.model_source == "huggingface":
        return name
    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise ImportError(
            "已选择 --model_source modelscope，但未安装 modelscope。请执行: pip install modelscope"
        ) from e
    cache = (args.modelscope_cache or "").strip() or os.environ.get(
        "MODELSCOPE_CACHE", os.path.join(os.getcwd(), ".modelscope")
    )
    os.makedirs(cache, exist_ok=True)
    print(f"  - ModelScope: {name}（缓存根目录: {cache}）")
    return snapshot_download(name, cache_dir=cache)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[1/5] 读取数据")
    if not args.train_jsonl:
        raise ValueError("已禁用 Excel 入口，请传 --train_jsonl（或使用默认 data/train.jsonl）。")
    if not os.path.exists(args.train_jsonl):
        raise FileNotFoundError(f"找不到训练集 JSONL: {args.train_jsonl}")
    train_records = load_jsonl(args.train_jsonl)
    print(f"  - train records: {len(train_records)}")
    train_examples = build_examples_from_json(
        train_records, instruction_override=args.instruction_override
    )

    if args.val_jsonl:
        if not os.path.exists(args.val_jsonl):
            raise FileNotFoundError(f"找不到验证集 JSONL: {args.val_jsonl}")
        val_records = load_jsonl(args.val_jsonl)
        print(f"  - val records: {len(val_records)}")
        val_examples = build_examples_from_json(
            val_records, instruction_override=args.instruction_override
        )
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

    pretrained = resolve_pretrained_path(args)
    print(f"[3/5] 加载 tokenizer/model: {args.model_name} -> {pretrained}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    print(f"  - model_dtype: {args.model_dtype}")
    model = create_model(
        pretrained,
        use_4bit=args.use_4bit,
        model_dtype=args.model_dtype,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    model.print_trainable_parameters()

    print("[4/5] Tokenize")
    tokenized = dataset.map(
        tokenize_fn(tokenizer=tokenizer, max_length=args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    tokenized["train"] = _filter_weak_supervision(
        tokenized["train"],
        args.min_supervised_tokens,
        "train",
        args.min_split_samples_after_filter,
    )
    tokenized["validation"] = _filter_weak_supervision(
        tokenized["validation"],
        args.min_supervised_tokens,
        "validation",
        args.min_split_samples_after_filter,
    )
    _print_supervision_stats(tokenized["train"], "train")
    _print_supervision_stats(tokenized["validation"], "validation")

    log_steps = _aligned_logging_steps(args.logging_steps, args.grad_accum)
    if args.logging_steps > 0 and log_steps != args.logging_steps:
        print(
            f"  - logging_steps 已从 {args.logging_steps} 调整为 {log_steps}（与 grad_accum={args.grad_accum} 对齐）"
        )

    if args.eval_steps > 0:
        eval_steps = args.eval_steps
    elif args.max_steps > 0:
        eval_steps = max(1, args.max_steps)
    else:
        eval_steps = 100
    if args.save_steps > 0:
        save_steps = args.save_steps
    elif args.max_steps > 0:
        save_steps = max(1, args.max_steps)
    else:
        save_steps = 100
    print(
        f"  - eval_steps={eval_steps}, save_steps={save_steps}"
        + ("（短跑自动对齐 max_steps）" if args.max_steps > 0 and args.eval_steps <= 0 else "")
    )

    print("[5/5] 开始训练")
    # 便于 nohup 日志：行缓冲 + 关闭 tqdm，避免 loss 行被 \r 进度条盖掉
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
        except (OSError, ValueError, AttributeError):
            pass
    # Transformers 5.x: evaluation_strategy was renamed to eval_strategy.
    _ta = inspect.signature(TrainingArguments.__init__).parameters
    _eval_kw = (
        {"eval_strategy": "steps"}
        if "eval_strategy" in _ta
        else {"evaluation_strategy": "steps"}
    )
    _log_extra: Dict[str, Any] = {"disable_tqdm": True}
    if "logging_first_step" in _ta:
        _log_extra["logging_first_step"] = True
    if "log_level" in _ta:
        _log_extra["log_level"] = "info"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        **_eval_kw,
        eval_steps=eval_steps,
        save_steps=save_steps,
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
        **_log_extra,
    )

    # Transformers 5.x: Trainer 使用 processing_class 替代 tokenizer 参数名。
    _trp = inspect.signature(Trainer.__init__).parameters
    _tok_kw = (
        {"processing_class": tokenizer}
        if "processing_class" in _trp
        else {"tokenizer": tokenizer}
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        **_tok_kw,
        data_collator=DataCollatorForCausalLMCustom(tokenizer=tokenizer),
    )
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"训练完成，模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
