#!/usr/bin/env python3
"""SAE Activation Collector — streams per-token activations from llama-server to disk.

Feeds conversation data through a running llama-server's completions API while the
/activations/collect endpoint streams per-token activation vectors to a binary file.

Usage:
    # Collect from Harry's DPO dataset, layer 24, 1000 rows
    python3 sae_collector.py --layer 24 --rows 1000

    # Collect both chosen and rejected responses (for contrastive analysis)
    python3 sae_collector.py --layer 24 --rows 500 --both

    # Collect from a custom JSONL file
    python3 sae_collector.py --layer 24 --rows 500 --input /path/to/data.jsonl

    # List available layers for the current model
    python3 sae_collector.py --info

    # Collect from multiple layers (runs sequentially)
    python3 sae_collector.py --layer 24,32,40 --rows 1000

Binary output format (readable by numpy):
    Header: "ACTV" (4B) + version:u32 + n_embd:u32 + layer_idx:u32
    Data:   float32[n_embd] per token, no framing
"""

import argparse
import json
import os
import random
import struct
import sys
import time

import requests

LLAMA_BASE = os.environ.get("LLAMA_BASE", "http://127.0.0.1:8080")
DEFAULT_DATASET = "data/training_data.jsonl"
DEFAULT_OUTPUT_DIR = "sae_data"


def get_server_info():
    """Get model info from the running llama-server."""
    try:
        r = requests.get(f"{LLAMA_BASE}/activations", params={"top_k": "1"}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error: Cannot reach llama-server at {LLAMA_BASE}: {e}")
        sys.exit(1)


def start_collection(layer, output_path):
    """Start per-token activation collection on the server."""
    r = requests.post(
        f"{LLAMA_BASE}/activations/collect",
        json={"layer": layer, "path": output_path},
        timeout=10
    )
    data = r.json()
    if r.status_code != 200:
        print(f"Error starting collection: {data}")
        sys.exit(1)
    return data


def stop_collection():
    """Stop collection and return stats."""
    r = requests.post(
        f"{LLAMA_BASE}/activations/collect",
        json={"stop": True},
        timeout=10
    )
    return r.json()


def get_collection_status():
    """Check current collection status."""
    r = requests.post(
        f"{LLAMA_BASE}/activations/collect",
        json={},
        timeout=5
    )
    return r.json()


def send_text_for_encoding(text, max_tokens=1):
    """Send text to the completions API for encoding (prompt processing only).

    We set max_tokens=1 because we only care about the forward pass through the
    prompt — the activations are captured during prompt processing. We don't need
    the model to actually generate a response.
    """
    try:
        r = requests.post(
            f"{LLAMA_BASE}/v1/completions",
            json={
                "prompt": text,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=120  # long prompts can take a while
        )
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        print(f"  Warning: Request failed: {e}")
        return None


def format_chat_messages(messages):
    """Format chat messages into a single text string using ChatML format.

    This matches Qwen's expected format so the model processes the tokens
    the same way it would in a real conversation.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def format_row_text(row, side="chosen"):
    """Format a dataset row into text for encoding.

    Handles two formats:
    1. Harry's DPO: chosen/rejected are full chat arrays (prompt embedded)
    2. Sycophancy/HH: separate 'prompt' string + chosen/rejected with assistant messages

    Returns formatted text string, or None if the side is empty.
    """
    messages = row.get(side, [])
    if not messages:
        return None

    # Check if this dataset has a separate prompt field
    prompt_text = row.get("prompt", "")

    if prompt_text and isinstance(messages, list) and len(messages) > 0:
        # Format: separate prompt + response messages (sycophancy dataset style)
        # Build a full conversation: user prompt + assistant response(s)
        parts = [f"<|im_start|>user\n{prompt_text}<|im_end|>"]
        for msg in messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(parts)
    elif isinstance(messages, list) and len(messages) > 0:
        # Format: full chat array (Harry's DPO style)
        return format_chat_messages(messages)
    elif isinstance(messages, str):
        # Plain text format
        return messages

    return None


def load_dataset(path, max_rows=None, shuffle=True, seed=42):
    """Load JSONL dataset, optionally sampling a subset."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if shuffle:
        random.seed(seed)
        random.shuffle(rows)

    if max_rows and max_rows < len(rows):
        rows = rows[:max_rows]

    return rows


def estimate_tokens(text):
    """Rough token count estimate (~4 chars per token)."""
    return len(text) // 4


def collect_layer(layer, rows, output_dir, process_both, dataset_name):
    """Run collection for a single layer."""
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f"activations_layer{layer}_{dataset_name}_{len(rows)}rows.bin"
    )
    # Use absolute path for the server
    abs_output = os.path.abspath(output_file)

    print(f"\nStarting collection: layer {layer} -> {output_file}")

    # Start server-side collection
    result = start_collection(layer, abs_output)
    n_embd = result.get("n_embd", 0)
    print(f"  Server confirmed: n_embd={n_embd}, collecting=True")

    total_tokens_est = 0
    total_rows_processed = 0
    start_time = time.time()

    try:
        for i, row in enumerate(rows):
            # Process chosen responses
            text = format_row_text(row, "chosen")
            if text:
                tok_est = estimate_tokens(text)

                # Skip extremely long sequences that might OOM
                if tok_est > 32000:
                    print(f"  Row {i}: skipping chosen ({tok_est} est tokens, too long)")
                else:
                    resp = send_text_for_encoding(text)
                    if resp:
                        total_tokens_est += tok_est
                        total_rows_processed += 1

            # Optionally process rejected responses too
            if process_both:
                text = format_row_text(row, "rejected")
                if text:
                    tok_est = estimate_tokens(text)
                    if tok_est > 32000:
                        print(f"  Row {i}: skipping rejected ({tok_est} est tokens, too long)")
                    else:
                        resp = send_text_for_encoding(text)
                        if resp:
                            total_tokens_est += tok_est

            # Progress update every 50 rows
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                status = get_collection_status()
                actual_tokens = status.get("collect_n_tokens", 0)
                rate = actual_tokens / elapsed if elapsed > 0 else 0
                file_size_mb = (actual_tokens * n_embd * 4 + 16) / (1024 * 1024) if n_embd else 0
                print(f"  [{i+1}/{len(rows)}] {actual_tokens:,} tokens collected "
                      f"({rate:.0f} tok/s), ~{file_size_mb:.0f} MB on disk, "
                      f"elapsed {elapsed:.0f}s")

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
    finally:
        # Always stop collection cleanly
        result = stop_collection()
        elapsed = time.time() - start_time
        final_tokens = result.get("tokens_collected", 0)
        file_size_mb = os.path.getsize(abs_output) / (1024 * 1024) if os.path.exists(abs_output) else 0

        print(f"\n  Collection complete:")
        print(f"    Layer:           {layer}")
        print(f"    Tokens:          {final_tokens:,}")
        print(f"    Rows processed:  {total_rows_processed}")
        print(f"    File:            {output_file}")
        print(f"    File size:       {file_size_mb:.1f} MB")
        print(f"    Time:            {elapsed:.0f}s")
        if elapsed > 0:
            print(f"    Throughput:      {final_tokens / elapsed:.0f} tokens/s")

    return final_tokens


def verify_output(path, expected_n_embd):
    """Quick verification of the output binary file."""
    if not os.path.exists(path):
        print(f"  WARNING: Output file not found: {path}")
        return False

    size = os.path.getsize(path)
    if size < 16:
        print(f"  WARNING: File too small ({size} bytes)")
        return False

    with open(path, "rb") as f:
        magic = f.read(4)
        version, n_embd, layer_idx = struct.unpack("<III", f.read(12))

    if magic != b"ACTV":
        print(f"  WARNING: Bad magic: {magic}")
        return False

    data_bytes = size - 16
    n_tokens = data_bytes // (n_embd * 4)
    remainder = data_bytes % (n_embd * 4)

    if remainder != 0:
        print(f"  WARNING: File has {remainder} trailing bytes (data may be truncated)")

    print(f"  Verified: {n_tokens:,} tokens, n_embd={n_embd}, layer={layer_idx}, v{version}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Collect activations for SAE training")
    parser.add_argument("--layer", type=str, default="24",
                        help="Layer index to collect from (comma-separated for multiple)")
    parser.add_argument("--rows", type=int, default=1000,
                        help="Number of dataset rows to process")
    parser.add_argument("--both", action="store_true",
                        help="Process both chosen and rejected responses")
    parser.add_argument("--input", type=str, default=DEFAULT_DATASET,
                        help="Input JSONL dataset path")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for binary files")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Don't shuffle the dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    parser.add_argument("--info", action="store_true",
                        help="Show server info and exit")
    args = parser.parse_args()

    # Get server info
    info = get_server_info()

    if args.info:
        print(f"llama-server activation capture status:")
        print(f"  Enabled:     {info.get('enabled')}")
        print(f"  Layers:      {info.get('n_layers')}")
        print(f"  Embedding:   {info.get('n_embd')}")
        print(f"  Collecting:  {info.get('collecting')}")
        print(f"  Last tokens: {info.get('last_n_tokens')}")
        if info.get("collecting"):
            print(f"  Collect layer: {info.get('collect_layer')}")
            print(f"  Tokens so far: {info.get('collect_n_tokens')}")
        return

    if not info.get("enabled"):
        print("Enabling activation capture...")
        requests.post(f"{LLAMA_BASE}/activations", json={"enabled": True}, timeout=5)

    if info.get("collecting"):
        print("WARNING: Collection already in progress. Stopping it first.")
        stop_collection()

    n_layers = info.get("n_layers", 0)
    n_embd = info.get("n_embd", 0)

    # Parse layer(s)
    layers = [int(x.strip()) for x in args.layer.split(",")]
    for layer in layers:
        if layer < 0 or layer >= n_layers:
            print(f"Error: Layer {layer} out of range (model has {n_layers} layers, 0-{n_layers-1})")
            sys.exit(1)

    # Load dataset
    if not os.path.exists(args.input):
        print(f"Error: Dataset not found: {args.input}")
        sys.exit(1)

    print(f"Loading dataset: {args.input}")
    rows = load_dataset(args.input, max_rows=args.rows,
                        shuffle=not args.no_shuffle, seed=args.seed)
    print(f"  Loaded {len(rows)} rows")

    dataset_name = os.path.splitext(os.path.basename(args.input))[0]

    # Estimate total tokens
    sample_size = min(50, len(rows))
    sample_chars = 0
    for row in rows[:sample_size]:
        text = format_row_text(row, "chosen")
        if text:
            sample_chars += len(text)
        if args.both:
            text = format_row_text(row, "rejected")
            if text:
                sample_chars += len(text)
    est_tokens = (sample_chars / sample_size * len(rows)) / 4
    est_size_mb = est_tokens * n_embd * 4 / (1024 * 1024)

    print(f"\nPlan:")
    print(f"  Layers:        {layers}")
    print(f"  Rows:          {len(rows)}")
    print(f"  Mode:          {'chosen + rejected' if args.both else 'chosen only'}")
    print(f"  Est tokens:    ~{est_tokens:,.0f}")
    print(f"  Est file size: ~{est_size_mb:,.0f} MB per layer")
    print(f"  Est time:      ~{est_tokens / 1500 / 60:.0f} min per layer (at ~1500 tok/s prompt processing)")
    print(f"  Output dir:    {args.output_dir}")

    # Collect each layer
    for layer in layers:
        collect_layer(layer, rows, args.output_dir, args.both, dataset_name)

    # Verify outputs
    print("\nVerifying output files:")
    for layer in layers:
        output_file = os.path.join(
            args.output_dir,
            f"activations_layer{layer}_{dataset_name}_{len(rows)}rows.bin"
        )
        verify_output(output_file, n_embd)

    print("\nDone! Next step: train SAE on the collected data.")
    print("Example:")
    print(f"  python3 sae_trainer.py --input {args.output_dir}/activations_layer{layers[0]}_{dataset_name}_{len(rows)}rows.bin")


if __name__ == "__main__":
    main()
