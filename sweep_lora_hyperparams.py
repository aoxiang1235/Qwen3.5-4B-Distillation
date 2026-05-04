#!/usr/bin/env python3
"""
按仓库内 **TRAINING_PLAN_AND_RESULTS.md** 的两阶段策略，对 ``new_mian.py`` LoRA 微调做 **短跑扫参**，
默认基座为 **Qwen3.5-4B**（与昨日 2.5-3B 流程对齐：先 LR、再 grad_accum×max_length 网格）。

阶段一（LR sweep）：``lr ∈ {5e-7, 1e-6, 2e-6}``，固定 ``grad_accum=4``, ``max_length=384``, ``max_steps=8``。
阶段二（网格）：在阶段一 **eval_loss 最低** 的 lr 上（可用 ``--phase2_lr`` 强行指定），跑 A/B/C：
  A: grad_accum=4, max_length=384
  B: grad_accum=8, max_length=384
  C: grad_accum=4, max_length=512

依赖：与 ``new_mian.py`` 相同（torch、transformers、peft、datasets；魔搭则 modelscope）。

示例（仓库根目录）：

  python3 sweep_lora_hyperparams.py \\
    --train_jsonl data/train.jsonl \\
    --val_jsonl data/val.jsonl \\
    --model_name Qwen/Qwen3.5-4B \\
    --model_dtype float32 \\
    --batch_size 1 \\
    --output_root training_runs/sweep_qwen35_4b

扫参结束后查看 ``<output_root>/sweep_results.json`` 与控制台汇总表。
全流程较长时需 GPU；单机顺序执行，避免多进程抢显存。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


EVAL_LOSS_RE = re.compile(
    r"eval_loss(?:\s*=|:)\s*([\d.]+(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)


@dataclass
class RunResult:
    name: str
    phase: str
    ok: bool
    lr: float
    grad_accum: int
    max_length: int
    max_steps: int
    batch_size: int
    output_dir: str
    eval_loss: Optional[float]
    stderr_tail: str
    duration_sec: float


def _parse_eval_loss(text: str) -> Optional[float]:
    """从 Trainer 日志中提取最后一次 eval_loss。"""
    matches = list(EVAL_LOSS_RE.finditer(text))
    if not matches:
        # transformers 有时打印 'eval_loss': 2.164
        alt = re.findall(r"['\"]eval_loss['\"]\s*:\s*([\d.]+(?:[eE][+-]?\d+)?)", text)
        if alt:
            return float(alt[-1])
        return None
    return float(matches[-1].group(1))


def _eval_loss_from_output_dir(output_dir: Path) -> Optional[float]:
    """从 HF Trainer 写出的 ``trainer_state.json``（通常在 checkpoint-* 下）读取最后一次 eval_loss。"""
    for ts in sorted(output_dir.rglob("trainer_state.json"), key=lambda p: str(p)):
        try:
            data = json.loads(ts.read_text(encoding="utf-8"))
            for entry in reversed(data.get("log_history", [])):
                if "eval_loss" in entry:
                    return float(entry["eval_loss"])
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return None


def _run_one(
    repo_root: Path,
    name: str,
    phase: str,
    output_dir: Path,
    extra_env: Dict[str, str],
    cmd_base: List[str],
    timeout_sec: int,
) -> RunResult:
    t0 = time.perf_counter()
    env = os.environ.copy()
    env.update(extra_env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(
        cmd_base + ["--output_dir", str(output_dir)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env=env,
    )
    log_blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    err_tail = (proc.stderr or "")[-4000:]
    loss: Optional[float] = None
    if proc.returncode == 0:
        loss = _eval_loss_from_output_dir(output_dir)
        if loss is None:
            loss = _parse_eval_loss(log_blob)
    return RunResult(
        name=name,
        phase=phase,
        ok=proc.returncode == 0,
        lr=float(cmd_base[cmd_base.index("--lr") + 1]) if "--lr" in cmd_base else 0.0,
        grad_accum=int(cmd_base[cmd_base.index("--grad_accum") + 1]),
        max_length=int(cmd_base[cmd_base.index("--max_length") + 1]),
        max_steps=int(cmd_base[cmd_base.index("--max_steps") + 1]),
        batch_size=int(cmd_base[cmd_base.index("--batch_size") + 1]),
        output_dir=str(output_dir),
        eval_loss=loss,
        stderr_tail=err_tail,
        duration_sec=time.perf_counter() - t0,
    )


def _fix_cmd_lr(cmd: List[str], lr: float) -> List[str]:
    out = cmd.copy()
    i = out.index("--lr")
    out[i + 1] = str(lr)
    return out


def _fix_cmd_grid(cmd: List[str], grad_accum: int, max_length: int) -> List[str]:
    out = cmd.copy()
    gi = out.index("--grad_accum")
    out[gi + 1] = str(grad_accum)
    mi = out.index("--max_length")
    out[mi + 1] = str(max_length)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3.5-4B LoRA 短跑扫参（对齐 TRAINING_PLAN）")
    p.add_argument("--train_jsonl", type=str, default="data/train.jsonl")
    p.add_argument(
        "--val_jsonl",
        type=str,
        default="data/val.jsonl",
        help="验证集；需与训练集字段一致",
    )
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-4B")
    p.add_argument("--model_source", type=str, choices=["modelscope", "huggingface"], default="modelscope")
    p.add_argument("--modelscope_cache", type=str, default="")
    p.add_argument("--model_dtype", type=str, choices=["float16", "float32"], default="float32")
    p.add_argument("--batch_size", type=int, default=1, help="扫参短跑建议 1（与文档 P100 一致）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument(
        "--output_root",
        type=str,
        default="",
        help="扫参输出根目录；默认 training_runs/sweep_qwen35_<timestamp>",
    )
    p.add_argument(
        "--phase",
        type=str,
        choices=("all", "1", "2"),
        default="all",
        help="只跑阶段一 / 只跑阶段二（需配合 --phase2_lr）/ 全流程",
    )
    p.add_argument(
        "--phase2_lr",
        type=float,
        default=0.0,
        help="阶段二固定学习率；默认 0 表示用阶段一 eval_loss 最优 lr",
    )
    p.add_argument("--timeout_sec", type=int, default=7200, help="单次训练子进程超时秒数")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root or f"training_runs/sweep_qwen35_{ts}")
    output_root.mkdir(parents=True, exist_ok=True)

    cmd_template: List[str] = [
        sys.executable,
        str(repo_root / "new_mian.py"),
        "--train_jsonl",
        args.train_jsonl,
        "--val_jsonl",
        args.val_jsonl,
        "--model_name",
        args.model_name,
        "--model_source",
        args.model_source,
        "--model_dtype",
        args.model_dtype,
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--lr",
        "2e-6",
        "--grad_accum",
        "4",
        "--max_length",
        "384",
        "--max_steps",
        "8",
        "--epochs",
        "1",
        "--logging_steps",
        "1",
        "--min_split_samples_after_filter",
        "10",
    ]
    if args.modelscope_cache.strip():
        cmd_template.extend(["--modelscope_cache", args.modelscope_cache.strip()])
    if args.use_4bit:
        cmd_template.append("--use_4bit")

    extra_env: Dict[str, str] = {}

    results: List[RunResult] = []
    phase1_lrs = [5e-7, 1e-6, 2e-6]
    best_lr = 2e-6
    phase1_rows: List[RunResult] = []

    if args.phase in ("all", "1"):
        print("=== 阶段一：LR sweep ===", flush=True)
        for i, lr in enumerate(phase1_lrs):
            name = f"p1_lr{i + 1}_{lr:g}"
            od = output_root / name
            cmd = _fix_cmd_lr(cmd_template, lr)
            print(f"\n>>> {name}  lr={lr:g}", flush=True)
            r = _run_one(repo_root, name, "1", od, extra_env, cmd, args.timeout_sec)
            phase1_rows.append(r)
            results.append(r)
            print(
                f"    ok={r.ok} eval_loss={r.eval_loss} time={r.duration_sec:.1f}s",
                flush=True,
            )
            if not r.ok:
                print(r.stderr_tail[-800:], flush=True)

        ok_loss = [(lr, rr.eval_loss) for lr, rr in zip(phase1_lrs, phase1_rows) if rr.ok and rr.eval_loss is not None]
        if ok_loss:
            best_lr = min(ok_loss, key=lambda x: x[1])[0]
            print(f"\n阶段一最优 lr = {best_lr:g}（按 eval_loss 最小）", flush=True)
        else:
            print("\n阶段一无有效 eval_loss，阶段二默认 lr=2e-6", flush=True)
            best_lr = 2e-6

    if args.phase in ("all", "2"):
        lr2 = args.phase2_lr if args.phase2_lr > 0 else best_lr
        if args.phase == "2" and args.phase2_lr <= 0:
            print("error: 仅跑阶段二时请指定 --phase2_lr", file=sys.stderr)
            sys.exit(1)
        print(f"\n=== 阶段二：网格 A/B/C，lr={lr2:g} ===", flush=True)
        grid = [
            ("A", 4, 384),
            ("B", 8, 384),
            ("C", 4, 512),
        ]
        cmd_p2_base = _fix_cmd_lr(cmd_template, lr2)
        for tag, ga, ml in grid:
            name = f"p2_{tag}_ga{ga}_ml{ml}"
            od = output_root / name
            cmd = _fix_cmd_grid(cmd_p2_base, ga, ml)
            print(f"\n>>> {name}", flush=True)
            r = _run_one(repo_root, name, "2", od, extra_env, cmd, args.timeout_sec)
            results.append(r)
            print(
                f"    ok={r.ok} eval_loss={r.eval_loss} time={r.duration_sec:.1f}s",
                flush=True,
            )
            if not r.ok:
                print(r.stderr_tail[-800:], flush=True)

    out_json = output_root / "sweep_results.json"
    payload = {
        "model_name": args.model_name,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "best_lr_phase1": best_lr if args.phase in ("all", "1") else None,
        "runs": [asdict(r) for r in results],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out_json}", flush=True)

    print("\n汇总（eval_loss 越低越好）:", flush=True)
    for r in results:
        if r.ok and r.eval_loss is not None:
            print(f"  {r.name:30s}  loss={r.eval_loss:.6f}  ga={r.grad_accum}  ml={r.max_length}  lr={r.lr:g}")
    best_runs = [r for r in results if r.ok and r.eval_loss is not None]
    if best_runs:
        best = min(best_runs, key=lambda x: x.eval_loss if x.eval_loss is not None else 1e9)
        print(
            f"\n建议长跑配置（仅供参考）: --lr {best.lr:g} --grad_accum {best.grad_accum} "
            f"--max_length {best.max_length} --max_steps -1"
        )


if __name__ == "__main__":
    main()
