#!/usr/bin/env python3
"""
Qwen3.5-4B + LoRA 的 JSONL 监督微调：instruction/content → prompt，output(JSON)→ assistant，
仅在回复段计 loss。默认魔搭拉基座。短跑/调参用命令行；nohup 建议: PYTHONUNBUFFERED=1 python3 -u new_mian.py
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

SYSTEM = "你是一个严谨的结构化信息抽取助手。"
LORA_TARGET = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32}
# Transformers 4/5 参数名差异，启动时各查一次
_SIG_TA = inspect.signature(TrainingArguments.__init__).parameters
_SIG_TR = inspect.signature(Trainer.__init__).parameters


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
    def _err(msg: str) -> None:
        raise ValueError(f"第 {row_idx} 条: {msg}")

    if not isinstance(output, dict):
        _err("output 须为 JSON object。")
    miss = {"is_beauty", "reasoning", "relationships"} - set(output)
    if miss:
        _err(f"output 缺字段: {sorted(miss)}")
    if not isinstance(output["is_beauty"], bool):
        _err("is_beauty 须为 bool")
    if not isinstance(output["reasoning"], str):
        _err("reasoning 须为 str")
    rels = output["relationships"]
    if not isinstance(rels, list):
        _err("relationships 须为 array")
    for j, rel in enumerate(rels):
        if not isinstance(rel, dict):
            _err(f"relationships[{j}] 须为 object")
        for k in ("brand_text", "start", "end"):
            if k not in rel or not isinstance(rel[k], str):
                _err(f"relationships[{j}].{k} 须为 string")


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


def _collate_causal(tokenizer) -> Any:
    pad = tokenizer.pad_token_id or 0

    def _batch(features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        def col(k: str) -> List[torch.Tensor]:
            return [torch.tensor(f[k], dtype=torch.long) for f in features]

        am = torch.nn.utils.rnn.pad_sequence(
            col("attention_mask"), batch_first=True, padding_value=0
        )
        lab = torch.nn.utils.rnn.pad_sequence(
            col("labels"), batch_first=True, padding_value=-100
        ).masked_fill(am == 0, -100)
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                col("input_ids"), batch_first=True, padding_value=pad
            ),
            "attention_mask": am,
            "labels": lab,
        }

    return _batch


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
        out_ids, out_m, out_l = [], [], []
        for ins, inp, otxt in zip(
            batch["instruction"], batch["input_text"], batch["output_text"]
        ):
            u = f"指令：{ins}\n\n文本：{inp}"
            sys_usr = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": u},
            ]
            full_text = tokenizer.apply_chat_template(
                sys_usr + [{"role": "assistant", "content": otxt}],
                tokenize=False,
                add_generation_prompt=False,
            )
            p_ids = tokenizer.apply_chat_template(
                sys_usr, tokenize=True, add_generation_prompt=True
            )
            full_u = tokenizer(full_text, truncation=False)["input_ids"]
            enc = tokenizer(
                full_text, truncation=True, max_length=max_length, padding=False
            )
            ids, attn = enc["input_ids"], enc["attention_mask"]
            if len(full_u) >= len(p_ids) and full_u[: len(p_ids)] == p_ids:
                a0 = len(p_ids)
            else:
                a0, n = 0, min(len(p_ids), len(full_u))
                while a0 < n and p_ids[a0] == full_u[a0]:
                    a0 += 1
            c = max(0, len(full_u) - len(ids))
            a_in = max(0, a0 - c)
            labels = list(ids)
            for i in range(min(a_in, len(labels))):
                labels[i] = -100
            out_ids.append(ids)
            out_m.append(attn)
            out_l.append(labels)
        return {"input_ids": out_ids, "attention_mask": out_m, "labels": out_l}

    return _tokenize


def create_model(
    model_name: str,
    use_4bit: bool,
    model_dtype: str,
    gradient_checkpointing: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    quant_config: Optional[BitsAndBytesConfig] = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    if model_dtype not in DTYPE_MAP:
        raise ValueError(f"不支持的 --model_dtype: {model_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE_MAP[model_dtype],
        quantization_config=quant_config,
        device_map="auto",
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(LORA_TARGET),
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
    base = max(1, requested) if requested > 0 else 20
    ga = max(1, grad_accum)
    return max(ga, ((base + ga - 1) // ga) * ga)


def _filter_and_stats(ds, min_tok: int, name: str):
    n0 = len(ds)

    def _ok(ex: Dict[str, List[int]]) -> bool:
        return sum(1 for t in ex["labels"] if t != -100) >= min_tok

    ds = ds.filter(_ok)
    n = len(ds)
    print(f"  - {name}: 弱监督 {n0}->{n} (min_tokens={min_tok})")
    if n < 10:
        raise RuntimeError(
            f"{name} 有效样本<10，降 --min_supervised_tokens 或查数据/模板"
        )
    cnts = sorted(sum(1 for t in r["labels"] if t != -100) for r in ds)
    m = len(cnts)
    print(
        f"  - {name} 监督 token: min={cnts[0]} p50={cnts[m // 2]} "
        f"p90={cnts[min(m - 1, int(m * 0.9))]} max={cnts[-1]} mean={sum(cnts) / m:.1f}"
    )
    return ds


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Qwen3.5-4B 蒸馏训练")
    for name, spec in (
        ("--train_jsonl", {"default": "", "help": "训练 JSONL（必填）"}),
        ("--val_jsonl", {"default": "", "help": "验证 JSONL；空则按 val_ratio 切分"}),
        (
            "--model_name",
            {"default": "Qwen/Qwen3.5-4B", "help": "魔搭/HF id 或本地含 config.json 的目录"},
        ),
        (
            "--model_source",
            {
                "choices": ["modelscope", "huggingface"],
                "default": "modelscope",
                "help": "基座来源",
            },
        ),
        ("--modelscope_cache", {"default": "", "help": "魔搭缓存目录，默认同目录 .modelscope"}),
        (
            "--model_dtype",
            {"choices": ["float16", "float32"], "default": "float32", "help": "权重精度"},
        ),
        ("--output_dir", {"default": "./distilled-qwen", "help": "输出目录"}),
        ("--max_length", {"type": int, "default": 1024}),
        ("--epochs", {"type": float, "default": 1.0}),
        ("--lr", {"type": float, "default": 1e-6}),
        ("--batch_size", {"type": int, "default": 2}),
        ("--grad_accum", {"type": int, "default": 8}),
        ("--val_ratio", {"type": float, "default": 0.02}),
        ("--seed", {"type": int, "default": 42}),
        ("--use_4bit", {"action": "store_true"}),
        ("--max_steps", {"type": int, "default": -1, "help": "限制步数，-1=按 epoch"}),
        (
            "--min_supervised_tokens",
            {"type": int, "default": 2, "help": "非 -100 的 label 数下限"},
        ),
        (
            "--logging_steps",
            {"type": int, "default": 0, "help": "0=自动与 grad_accum 对齐"},
        ),
        ("--max_grad_norm", {"type": float, "default": 1.0}),
        ("--no_gradient_checkpointing", {"action": "store_true"}),
        ("--lora_r", {"type": int, "default": 16}),
        ("--lora_alpha", {"type": int, "default": 32}),
        ("--lora_dropout", {"type": float, "default": 0.05}),
        ("--eval_steps", {"type": int, "default": 100, "help": "评估间隔（optimizer step）"}),
        ("--save_steps", {"type": int, "default": 0, "help": "0=同 eval_steps"}),
        ("--warmup_ratio", {"type": float, "default": 0.03}),
        ("--no_load_best_at_end", {"action": "store_true"}),
    ):
        kwargs = dict(spec)
        if kwargs.get("action") != "store_true" and "type" not in kwargs:
            kwargs["type"] = str
        ap.add_argument(name, **kwargs)
    return ap.parse_args()


def resolve_pretrained_path(args) -> str:
    """本地目录 / HuggingFace id / 魔搭 snapshot 路径之一。"""
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


def _estimate_max_opt_steps(args: argparse.Namespace, n_train: int) -> int:
    if args.max_steps and args.max_steps > 0:
        return int(args.max_steps)
    ga = max(1, args.grad_accum)
    per_ep = (n_train + args.batch_size * ga - 1) // max(1, args.batch_size * ga)
    return int(per_ep * args.epochs)


def _build_training_args(
    args: argparse.Namespace, log_steps: int, load_best: bool
) -> TrainingArguments:
    ev = (
        {"eval_strategy": "steps"}
        if "eval_strategy" in _SIG_TA
        else {"evaluation_strategy": "steps"}
    )
    extra: Dict[str, Any] = {"disable_tqdm": True}
    if "logging_first_step" in _SIG_TA:
        extra["logging_first_step"] = True
    if "log_level" in _SIG_TA:
        extra["log_level"] = "info"
    save_steps = args.save_steps if args.save_steps > 0 else args.eval_steps
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        **ev,
        eval_steps=args.eval_steps,
        save_steps=save_steps,
        logging_steps=log_steps,
        warmup_ratio=args.warmup_ratio,
        bf16=False,
        fp16=False,
        max_grad_norm=args.max_grad_norm,
        save_total_limit=3,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        **extra,
    )


def main():
    args = parse_args()
    if args.lora_r < 1:
        raise ValueError("--lora_r 须 >= 1")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora_dropout 须在 [0, 1) 内")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[1/5] 读取数据")
    if not args.train_jsonl:
        raise ValueError("请传 --train_jsonl。")
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
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.print_trainable_parameters()

    print("[4/5] Tokenize")
    tokenized = dataset.map(
        tokenize_fn(tokenizer=tokenizer, max_length=args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    tokenized["train"] = _filter_and_stats(
        tokenized["train"], args.min_supervised_tokens, "train"
    )
    tokenized["validation"] = _filter_and_stats(
        tokenized["validation"], args.min_supervised_tokens, "validation"
    )

    log_steps = _aligned_logging_steps(args.logging_steps, args.grad_accum)
    if args.logging_steps > 0 and log_steps != args.logging_steps:
        print(
            f"  - logging_steps 已从 {args.logging_steps} 调整为 {log_steps}（与 grad_accum={args.grad_accum} 对齐）"
        )

    print("[5/5] 开始训练")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
        except (OSError, ValueError, AttributeError):
            pass
    n_tr = len(tokenized["train"])
    max_opt = _estimate_max_opt_steps(args, n_tr)
    load_best = not args.no_load_best_at_end
    if load_best and max_opt > 0 and args.eval_steps > max_opt:
        print(
            f"  - 总步数约 {max_opt} < eval_steps={args.eval_steps}，已关闭 load_best_model_at_end"
        )
        load_best = False
    training_args = _build_training_args(args, log_steps, load_best)
    _tok = (
        {"processing_class": tokenizer}
        if "processing_class" in _SIG_TR
        else {"tokenizer": tokenizer}
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=_collate_causal(tokenizer),
        **_tok,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"训练完成，模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
