#!/usr/bin/env python3
"""
将 training_data_4B.xlsx 转换为训练用 JSONL（instruction/content/output）。

默认生成 data/train.jsonl、data/val.jsonl（无字符偏移）。
加 --v3 时生成 train_v3.jsonl / val_v3.jsonl：relationships 每项含 start/end（半开区间）；
在正文中找不到子串时仍保留该条，start=end=-1（与「不丢弃 relationship」一致）。
"""

import argparse
import json
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

DEFAULT_INSTRUCTION = (
    "You are an information extraction assistant.\n\n"
    "Task:\n"
    "1) Determine whether the post is beauty-related.\n"
    "2) Extract beauty-related brand mentions.\n\n"
    "Output:\n"
    "Return ONLY a valid JSON object with this schema:\n"
    '{"is_beauty": true/false, "reasoning": "short reason", "relationships": [{"brand_text":"..."}]}\n\n'
    "Example:\n"
    '{"is_beauty": true, "reasoning": "Perfume discussion with explicit brand mentions.", "relationships": [{"brand_text":"Guerlain"},{"brand_text":"Dior"}]}\n\n'
    "Rules:\n"
    "1) Keep only beauty-related BRANDS (cosmetics, skincare, fragrance, haircare).\n"
    "2) Exclude person names, influencer handles, artist names, and store/salon/spa/shop names.\n"
    '3) Mentions like "@username" are NOT brands unless the text clearly refers to a beauty brand/product.\n'
    "4) brand_text must be copied from the original text span (no rewriting).\n"
    "5) Remove duplicates, keep first-mention order.\n"
    "6) If uncertain whether a mention is a brand, do NOT extract it.\n"
    "7) If is_beauty is false, relationships must be [].\n"
    "8) reasoning should be concise (one short sentence).\n"
    "9) Do not output any extra text.\n\n"
    "Negative examples:\n"
    '- Input: "Thanks Chloe for doing my makeup at Bella Beauty Studio."\n'
    '- Output: {"is_beauty": true, "reasoning": "Beauty context but only person/store mentions, no clear brand.", "relationships": []}\n'
    '- Input: "Follow @amy_makeup_artist and visit Glow Salon now."\n'
    '- Output: {"is_beauty": true, "reasoning": "Contains person/store mentions without clear brand.", "relationships": []}'
)


def _read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values: List[str] = []
    for si in root.findall(f"{XML_NS}si"):
        texts = [t.text or "" for t in si.iter(f"{XML_NS}t")]
        values.append("".join(texts))
    return values


def _read_cell_value(cell: ET.Element, shared: List[str]) -> str:
    cell_type = cell.attrib.get("t")
    v = cell.find(f"{XML_NS}v")
    if v is None or v.text is None:
        return ""
    if cell_type == "s":
        idx = int(v.text)
        return shared[idx] if idx < len(shared) else ""
    return v.text


def load_excel_rows(xlsx_path: str) -> List[Dict[str, str]]:
    with zipfile.ZipFile(xlsx_path) as zf:
        shared = _read_shared_strings(zf)
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows = sheet.find(f"{XML_NS}sheetData").findall(f"{XML_NS}row")
    if not rows:
        return []
    headers = [_read_cell_value(c, shared).strip() for c in rows[0].findall(f"{XML_NS}c")]

    results: List[Dict[str, str]] = []
    for row in rows[1:]:
        vals = [_read_cell_value(c, shared) for c in row.findall(f"{XML_NS}c")]
        if not any(vals):
            continue
        item = {}
        for i, h in enumerate(headers):
            item[h] = vals[i] if i < len(vals) else ""
        results.append(item)
    return results


def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def find_brand_char_span(content: str, brand_text: str) -> Tuple[int, int]:
    """
    在 content 中为 brand_text 找一段 [start, end) 的字符偏移（与 Python 下标一致）。
    先精确子串，再大小写不敏感；找不到返回 (-1, -1)，调用方仍保留 relationship。
    """
    if not brand_text:
        return (-1, -1)
    n = len(content)
    i = content.find(brand_text)
    if i >= 0:
        e = i + len(brand_text)
        if 0 <= i < e <= n:
            return (i, e)
    lo_c, lo_b = content.lower(), brand_text.lower()
    j = lo_c.find(lo_b)
    if j >= 0:
        e = j + len(brand_text)
        if 0 <= j < e <= n:
            return (j, e)
    return (-1, -1)


def build_samples(
    rows: List[Dict[str, str]],
    instruction: str,
    with_offsets: bool = False,
) -> Tuple[List[Dict], Dict[str, int]]:
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        content = str(row.get("content") or row.get("clean_text") or "").strip()
        post_id = str(row.get("post_id") or "").strip()
        if not content:
            continue
        groups[(post_id, content)].append(row)

    samples: List[Dict] = []
    multi_rel = 0
    rel_with_span = 0
    rel_missing_span = 0

    for (post_id, content), items in groups.items():
        is_beauty = any(parse_bool(row.get("is_beauty", "")) for row in items)
        reasoning = ""
        brands_ordered: List[str] = []
        seen = set()

        for row in items:
            if not reasoning:
                candidate_reasoning = str(row.get("reasoning") or "").strip()
                if candidate_reasoning:
                    reasoning = candidate_reasoning
            brand_text = str(row.get("brand_text") or row.get("brand") or "").strip()
            if not brand_text:
                continue

            key = brand_text.lower()
            if key in seen:
                continue
            seen.add(key)
            brands_ordered.append(brand_text)

        relationships: List[Dict] = []
        if is_beauty:
            for brand_text in brands_ordered:
                if with_offsets:
                    start, end = find_brand_char_span(content, brand_text)
                    if start >= 0 and end >= 0:
                        rel_with_span += 1
                    else:
                        rel_missing_span += 1
                    relationships.append(
                        {"brand_text": brand_text, "start": start, "end": end}
                    )
                else:
                    relationships.append({"brand_text": brand_text})

        if len(relationships) > 1:
            multi_rel += 1

        output = {
            "is_beauty": is_beauty,
            "reasoning": reasoning,
            "relationships": relationships,
        }
        samples.append(
            {
                "post_id": post_id,
                "instruction": instruction,
                "content": content,
                "output": output,
            }
        )

    stats: Dict[str, int] = {
        "groups": len(groups),
        "samples": len(samples),
        "multi_relationship_samples": multi_rel,
    }
    if with_offsets:
        stats["relationships_with_span"] = rel_with_span
        stats["relationships_missing_span"] = rel_missing_span
    return samples, stats


def split_and_write(
    samples: List[Dict],
    out_dir: str,
    val_ratio: float,
    seed: int,
    train_name: str = "train.jsonl",
    val_name: str = "val.jsonl",
) -> None:
    random.seed(seed)
    random.shuffle(samples)
    val_size = max(1, int(len(samples) * val_ratio))
    val = samples[:val_size]
    train = samples[val_size:]

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, train_name)
    val_path = os.path.join(out_dir, val_name)

    with open(train_path, "w", encoding="utf-8") as f:
        for row in train:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for row in val:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{train_name}: {len(train)} -> {train_path}")
    print(f"{val_name}:   {len(val)} -> {val_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Excel 转 JSONL 数据准备")
    parser.add_argument("--xlsx_path", type=str, default="training_data_4B.xlsx")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--v3",
        action="store_true",
        help="生成 train_v3.jsonl / val_v3.jsonl，relationships 含 start/end；找不到则为 -1，仍保留条目",
    )
    parser.add_argument(
        "--instruction-file",
        type=str,
        default="",
        help="与 --v3 联用：UTF-8 提示词文件；留空则使用 data/prompts/brand_extraction_fixed_offsets.txt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.xlsx_path):
        raise FileNotFoundError(f"找不到 Excel: {args.xlsx_path}")

    rows = load_excel_rows(args.xlsx_path)
    print(f"excel rows: {len(rows)}")

    if args.v3:
        inst_path = Path(
            (args.instruction_file or "").strip()
            or "data/prompts/brand_extraction_fixed_offsets.txt"
        )
        if not inst_path.is_file():
            raise FileNotFoundError(f"--v3 需要 instruction 文件，找不到: {inst_path}")
        instruction = inst_path.read_text(encoding="utf-8").rstrip("\n")
        if not instruction:
            raise ValueError(f"instruction 文件为空: {inst_path}")
        samples, stats = build_samples(
            rows, instruction=instruction, with_offsets=True
        )
        train_name, val_name = "train_v3.jsonl", "val_v3.jsonl"
    else:
        samples, stats = build_samples(
            rows, instruction=args.instruction, with_offsets=False
        )
        train_name, val_name = "train.jsonl", "val.jsonl"

    print("stats:", json.dumps(stats, ensure_ascii=False))

    if len(samples) < 100:
        raise RuntimeError("可用样本过少，请先检查数据映射。")
    split_and_write(
        samples,
        out_dir=args.out_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_name=train_name,
        val_name=val_name,
    )


if __name__ == "__main__":
    main()
