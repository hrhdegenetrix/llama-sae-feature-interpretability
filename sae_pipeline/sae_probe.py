#!/usr/bin/env python3 -u
"""SAE Feature Probe — find which features activate on specific text.

Feed specific phrases or texts through llama-server, capture activations,
run through a trained SAE, and report which features fire most strongly.
Use this to label features by finding what text activates them.

Usage:
    # Probe a single phrase
    python3 sae_probe.py --model sae_models/sae_layer24_16384f_20260305_114446.pt \
        --text "You are absolutely right."

    # Probe multiple phrases and compare
    python3 sae_probe.py --model sae_models/sae_layer24_16384f_20260305_114446.pt \
        --text "You are absolutely right." \
        --text "I disagree with that assessment." \
        --text "That's an interesting perspective, but I see it differently."

    # Probe with a contrast phrase to find discriminating features
    python3 sae_probe.py --model sae_models/sae_layer24_16384f_20260305_114446.pt \
        --text "You are absolutely right." \
        --contrast "I think there's more to it than that."

    # Probe from a file (one phrase per line)
    python3 sae_probe.py --model sae_models/sae_layer24_16384f_20260305_114446.pt \
        --file phrases.txt

    # Find max-activating examples for a specific feature from a dataset
    python3 sae_probe.py --model sae_models/sae_layer24_16384f_20260305_114446.pt \
        --feature 5578 --dataset sae_data/activations_layer24_harry_dpo_ready_20rows.bin
"""

import argparse
import json
import os
import struct
import sys
import time
import tempfile

import numpy as np
import requests
import torch

from sae_trainer import SparseAutoencoder, load_activation_file, MemmapActivationDataset

LLAMA_BASE = os.environ.get("LLAMA_BASE", "http://127.0.0.1:8080")


def collect_activations_for_text(text, layer):
    """Feed text through llama-server and collect per-token activations at the target layer.

    Returns numpy array of shape (n_tokens, n_embd) or None on failure.
    """
    # Get server info
    try:
        info = requests.get(f"{LLAMA_BASE}/activations", timeout=5).json()
    except Exception as e:
        print(f"Error: Cannot reach llama-server: {e}")
        return None

    n_embd = info.get("n_embd", 0)
    if n_embd == 0:
        print("Error: Server reports n_embd=0 — activation capture may not be initialized")
        return None

    # Enable capture if not already
    if not info.get("enabled"):
        requests.post(f"{LLAMA_BASE}/activations", json={"enabled": True}, timeout=5)

    # Stop any existing collection
    if info.get("collecting"):
        requests.post(f"{LLAMA_BASE}/activations/collect", json={"stop": True}, timeout=5)

    # Create temp file for collection
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False, dir="/tmp") as tmp:
        tmp_path = tmp.name

    try:
        # Start collection
        result = requests.post(
            f"{LLAMA_BASE}/activations/collect",
            json={"layer": layer, "path": tmp_path},
            timeout=10
        ).json()

        if not result.get("started"):
            print(f"Error starting collection: {result}")
            return None

        # Send text for encoding (prompt processing only)
        resp = requests.post(
            f"{LLAMA_BASE}/v1/completions",
            json={"prompt": text, "max_tokens": 1, "temperature": 0},
            timeout=60
        )

        if resp.status_code != 200:
            print(f"Error from completions API: {resp.status_code}")
            return None

        # Stop collection
        stop_result = requests.post(
            f"{LLAMA_BASE}/activations/collect",
            json={"stop": True},
            timeout=10
        ).json()

        tokens_collected = stop_result.get("tokens_collected", 0)

        if tokens_collected == 0:
            print("Warning: 0 tokens collected")
            return None

        # Read the binary file
        file_size = os.path.getsize(tmp_path)
        if file_size <= 16:
            print("Warning: Empty activation file")
            return None

        data = np.memmap(tmp_path, dtype=np.float32, mode='r', offset=16,
                         shape=(tokens_collected, n_embd))
        # Copy to regular array since we'll delete the file
        result_array = np.array(data)
        return result_array

    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


def probe_text(model, text, layer, device="cuda", top_n=20):
    """Probe which SAE features activate on a given text.

    Returns dict with per-feature stats across all tokens in the text.
    """
    activations = collect_activations_for_text(text, layer)
    if activations is None:
        return None

    n_tokens = len(activations)
    tensor = torch.from_numpy(activations).float().to(device)

    model.eval()
    with torch.no_grad():
        features = model.encode(tensor)

    features_np = features.cpu().numpy()

    # Per-feature stats across tokens in this text
    mean_act = features_np.mean(axis=0)
    max_act = features_np.max(axis=0)
    freq = (features_np > 0).mean(axis=0)

    # Find top features by mean activation
    top_by_mean = np.argsort(-mean_act)[:top_n]
    # Find top features by max activation (strongest single-token fire)
    top_by_max = np.argsort(-max_act)[:top_n]
    # Find top by frequency (most consistently active)
    top_by_freq = np.argsort(-freq)[:top_n]

    return {
        "text": text[:100] + ("..." if len(text) > 100 else ""),
        "n_tokens": n_tokens,
        "mean_activation": mean_act,
        "max_activation": max_act,
        "frequency": freq,
        "top_by_mean": top_by_mean,
        "top_by_max": top_by_max,
        "top_by_freq": top_by_freq,
        "features_per_token": features_np,
    }


def print_probe_result(result, label=None):
    """Print a formatted probe result."""
    if result is None:
        print("  (no result)")
        return

    header = label or result["text"]
    print(f"\n{'='*80}")
    print(f"  {header}")
    print(f"  ({result['n_tokens']} tokens)")
    print(f"{'='*80}")

    print(f"\n  Top features by mean activation:")
    print(f"  {'Feature':>8}  {'Mean':>10}  {'Max':>10}  {'Freq':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")
    for idx in result["top_by_mean"][:20]:
        print(f"  {idx:>8d}  {result['mean_activation'][idx]:>10.6f}  "
              f"{result['max_activation'][idx]:>10.6f}  "
              f"{result['frequency'][idx]:>8.4f}")

    print(f"\n  Top features by peak activation (strongest single-token fire):")
    print(f"  {'Feature':>8}  {'Max':>10}  {'Mean':>10}  {'Freq':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")
    for idx in result["top_by_max"][:10]:
        print(f"  {idx:>8d}  {result['max_activation'][idx]:>10.6f}  "
              f"{result['mean_activation'][idx]:>10.6f}  "
              f"{result['frequency'][idx]:>8.4f}")


def differential_probe(model, target_text, contrast_text, layer, device="cuda", top_n=30):
    """Compare feature activations between two texts to find discriminating features."""
    print(f"Probing target text...")
    target = probe_text(model, target_text, layer, device=device)
    if target is None:
        print("Failed to probe target text")
        return

    print(f"Probing contrast text...")
    contrast = probe_text(model, contrast_text, layer, device=device)
    if contrast is None:
        print("Failed to probe contrast text")
        return

    # Find features that are much more active in target than contrast
    mean_diff = target["mean_activation"] - contrast["mean_activation"]
    freq_diff = target["frequency"] - contrast["frequency"]

    print(f"\n{'='*80}")
    print(f"  DIFFERENTIAL: \"{target_text[:60]}\" vs \"{contrast_text[:60]}\"")
    print(f"{'='*80}")

    # Target-specific features
    target_specific = np.argsort(-mean_diff)
    print(f"\n  Features MORE active in target (unique to \"{target_text[:40]}\"):")
    print(f"  {'Feature':>8}  {'T_mean':>10}  {'C_mean':>10}  {'Diff':>10}  {'T_freq':>8}  {'C_freq':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for idx in target_specific[:top_n]:
        if mean_diff[idx] <= 0:
            break
        print(f"  {idx:>8d}  {target['mean_activation'][idx]:>10.6f}  "
              f"{contrast['mean_activation'][idx]:>10.6f}  "
              f"{mean_diff[idx]:>+10.6f}  "
              f"{target['frequency'][idx]:>8.4f}  "
              f"{contrast['frequency'][idx]:>8.4f}")

    # Contrast-specific features
    contrast_specific = np.argsort(mean_diff)
    print(f"\n  Features MORE active in contrast (unique to \"{contrast_text[:40]}\"):")
    print(f"  {'Feature':>8}  {'T_mean':>10}  {'C_mean':>10}  {'Diff':>10}  {'T_freq':>8}  {'C_freq':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for idx in contrast_specific[:top_n]:
        if mean_diff[idx] >= 0:
            break
        print(f"  {idx:>8d}  {target['mean_activation'][idx]:>10.6f}  "
              f"{contrast['mean_activation'][idx]:>10.6f}  "
              f"{mean_diff[idx]:>+10.6f}  "
              f"{target['frequency'][idx]:>8.4f}  "
              f"{contrast['frequency'][idx]:>8.4f}")

    return target, contrast, mean_diff


def batch_probe_shared_features(model, texts, layer, device="cuda", top_n=30,
                                min_phrase_ratio=0.5):
    """Probe multiple phrases and find features that are consistently active across them.

    This is the core algorithm for discovering behavioral features from phrase clusters.
    A feature must be in the top_n for at least min_phrase_ratio of the phrases to be
    considered "shared."

    Returns a JSON-serializable dict with shared features and per-phrase results.
    """
    n_features = model.n_features
    results = []

    print(f"Batch probing {len(texts)} phrases...")
    for i, text in enumerate(texts):
        print(f"  [{i+1}/{len(texts)}] \"{text[:60]}\"")
        result = probe_text(model, text, layer, device=device, top_n=top_n)
        if result:
            results.append(result)
        else:
            print(f"    (failed)")

    if len(results) < 2:
        print("Error: Need at least 2 successful probes for shared feature analysis")
        return None

    # For each feature, count how many phrases have it in their top-N by mean activation
    feature_phrase_count = np.zeros(n_features)
    feature_total_mean = np.zeros(n_features)
    feature_total_freq = np.zeros(n_features)

    for r in results:
        top_indices = r["top_by_mean"][:top_n]
        for idx in top_indices:
            feature_phrase_count[idx] += 1
        feature_total_mean += r["mean_activation"]
        feature_total_freq += r["frequency"]

    # Average across phrases
    n_phrases = len(results)
    feature_avg_mean = feature_total_mean / n_phrases
    feature_avg_freq = feature_total_freq / n_phrases

    # Find shared features: present in top-N of at least min_phrase_ratio of phrases
    min_count = max(2, int(n_phrases * min_phrase_ratio))
    shared_mask = feature_phrase_count >= min_count
    shared_indices = np.where(shared_mask)[0]

    # Rank shared features by average mean activation
    shared_scores = feature_avg_mean[shared_indices]
    rank_order = np.argsort(-shared_scores)
    ranked_shared = shared_indices[rank_order]

    # Build output
    print(f"\n{'='*80}")
    print(f"  SHARED FEATURES across {n_phrases} phrases (min {min_count}/{n_phrases} agreement)")
    print(f"{'='*80}")
    print(f"  Found {len(ranked_shared)} shared features\n")

    if len(ranked_shared) > 0:
        print(f"  {'Feature':>8}  {'Phrases':>8}  {'Avg Mean':>10}  {'Avg Freq':>10}  {'Score':>10}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")

    shared_features = []
    for idx in ranked_shared[:top_n]:
        count = int(feature_phrase_count[idx])
        avg_mean = float(feature_avg_mean[idx])
        avg_freq = float(feature_avg_freq[idx])
        # Score combines consistency (how many phrases) with strength (mean activation)
        score = (count / n_phrases) * avg_mean
        print(f"  {idx:>8d}  {count:>4d}/{n_phrases:<3d}  {avg_mean:>10.6f}  {avg_freq:>10.4f}  {score:>10.6f}")
        shared_features.append({
            "feature": int(idx),
            "phrase_count": count,
            "total_phrases": n_phrases,
            "avg_mean": avg_mean,
            "avg_freq": avg_freq,
            "score": score,
        })

    # Build feature spec for extraction (weighted by score)
    if shared_features:
        specs = []
        max_score = shared_features[0]["score"]
        for sf in shared_features[:10]:  # top 10 for extraction
            weight = sf["score"] / max_score if max_score > 0 else 1.0
            specs.append(f"{sf['feature']}:{weight:.2f}")
        feature_spec = ",".join(specs)
        print(f"\n  Extraction spec (top {min(10, len(shared_features))}, weighted by score):")
        print(f"  {feature_spec}")
    else:
        feature_spec = ""

    return {
        "n_phrases": n_phrases,
        "n_probed": len(texts),
        "n_failed": len(texts) - n_phrases,
        "min_agreement": min_count,
        "shared_features": shared_features,
        "feature_spec": feature_spec,
        "phrases": [r["text"] for r in results],
    }


def main():
    parser = argparse.ArgumentParser(description="Probe SAE features on specific text")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained SAE model (.pt)")
    parser.add_argument("--text", type=str, action="append", default=[],
                        help="Text to probe (can specify multiple)")
    parser.add_argument("--contrast", type=str, default=None,
                        help="Contrast text for differential analysis")
    parser.add_argument("--file", type=str, default=None,
                        help="File with one phrase per line to probe")
    parser.add_argument("--batch-json", type=str, default=None,
                        help="JSON string with {phrases: [...]} for batch shared-feature analysis")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to collect from (auto-detected from model metadata)")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top features to show (default 20)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load SAE model
    print(f"Loading SAE model: {args.model}")
    meta_path = args.model.replace(".pt", ".json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
        n_embd = metadata["n_embd"]
        n_features = metadata["n_features"]
        layer = metadata.get("layer_idx", 24)
        print(f"  {n_embd} -> {n_features} features, layer {layer}")
    else:
        state = torch.load(args.model, map_location="cpu")
        n_features = state["encoder.weight"].shape[0]
        n_embd = state["encoder.weight"].shape[1]
        layer = 24
        print(f"  Inferred: {n_embd} -> {n_features} features")

    if args.layer is not None:
        layer = args.layer

    model = SparseAutoencoder(n_embd, n_features).to(args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.eval()
    print(f"  Loaded on {args.device}, will collect from layer {layer}")

    # Collect texts to probe
    texts = list(args.text)
    if args.file:
        with open(args.file) as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)

    # Batch JSON mode — shared feature analysis, outputs JSON to stdout
    if args.batch_json:
        try:
            batch_data = json.loads(args.batch_json)
            texts = batch_data.get("phrases", [])
            top_n = batch_data.get("top_n", 30)
            min_ratio = batch_data.get("min_phrase_ratio", 0.5)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON: {e}"}))
            sys.exit(1)

        if len(texts) < 2:
            print(json.dumps({"error": "Need at least 2 phrases for batch analysis"}))
            sys.exit(1)

        result = batch_probe_shared_features(model, texts, layer, device=args.device,
                                             top_n=top_n, min_phrase_ratio=min_ratio)
        if result:
            print(f"\n__JSON_OUTPUT__")
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps({"error": "Batch probe failed — not enough successful probes"}))
        return

    if not texts:
        print("Error: No text to probe. Use --text or --file.")
        sys.exit(1)

    # Differential mode
    if args.contrast and len(texts) == 1:
        differential_probe(model, texts[0], args.contrast, layer, device=args.device,
                          top_n=args.top)
        return

    # Standard probe mode
    results = []
    for text in texts:
        print(f"\nProbing: \"{text[:80]}\"...")
        result = probe_text(model, text, layer, device=args.device, top_n=args.top)
        if result:
            results.append(result)
            print_probe_result(result)

    # If multiple texts, show comparison
    if len(results) > 1:
        print(f"\n{'='*80}")
        print(f"  COMPARISON — Features that differ most between probed texts")
        print(f"{'='*80}")

        # For each pair, find the biggest differences
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                diff = results[i]["mean_activation"] - results[j]["mean_activation"]
                top_diff = np.argsort(-np.abs(diff))[:15]

                print(f"\n  \"{results[i]['text'][:40]}\" vs \"{results[j]['text'][:40]}\":")
                print(f"  {'Feature':>8}  {'Text1':>10}  {'Text2':>10}  {'Diff':>10}")
                for idx in top_diff:
                    print(f"  {idx:>8d}  {results[i]['mean_activation'][idx]:>10.6f}  "
                          f"{results[j]['mean_activation'][idx]:>10.6f}  "
                          f"{diff[idx]:>+10.6f}")


if __name__ == "__main__":
    main()
