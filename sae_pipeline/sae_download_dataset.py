#!/usr/bin/env python3
"""Download HuggingFace datasets to local JSONL for SAE collection.

Usage:
    python3 sae_download_dataset.py Taywon/HH_sycophancy_biased_15k_parsed
    python3 sae_download_dataset.py lmsys/lmsys-chat-1m --rows 5000
"""

import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Download HF dataset to local JSONL")
    parser.add_argument("dataset", help="HuggingFace dataset name (e.g., Taywon/HH_sycophancy_biased_15k_parsed)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--rows", type=int, default=None, help="Max rows to download")
    parser.add_argument("--output-dir", default="sae_data/datasets", help="Output directory")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.dataset} (split={args.split})...")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"  {len(ds)} rows, columns: {ds.column_names}")

    if args.rows and args.rows < len(ds):
        ds = ds.select(range(args.rows))
        print(f"  Truncated to {len(ds)} rows")

    os.makedirs(args.output_dir, exist_ok=True)
    safe_name = args.dataset.replace("/", "_")
    output_path = os.path.join(args.output_dir, f"{safe_name}.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved to {output_path} ({size_mb:.1f} MB, {len(ds)} rows)")

    # Show a sample
    with open(output_path) as f:
        sample = json.loads(f.readline())
    print(f"  Sample keys: {list(sample.keys())}")
    for k, v in sample.items():
        preview = str(v)[:120]
        print(f"    {k}: {preview}")

if __name__ == "__main__":
    main()
