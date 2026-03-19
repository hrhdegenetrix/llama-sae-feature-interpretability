#!/usr/bin/env python3 -u
"""Batch re-probe all saved phrase clusters against a trained SAE.

Reads each cluster from sae_data/clusters/, probes every phrase through
llama-server to collect activations, encodes them with the SAE, and finds
shared features. Outputs updated cluster files with new feature mappings
and generates a sae_steering.json for the persona.

Usage:
    python3 sae_batch_reprobe.py \
        --model sae_models/sae_layer20_16384f_20260306_011034.pt \
        --output sae_steering.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from sae_trainer import SparseAutoencoder
from sae_probe import collect_activations_for_text


def load_clusters(cluster_dir="sae_data/clusters"):
    """Load all cluster JSON files."""
    clusters = {}
    for fname in sorted(os.listdir(cluster_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(cluster_dir, fname)
        with open(path) as f:
            data = json.load(f)
        label = data.get("label", fname.replace(".json", ""))
        clusters[label] = data
    return clusters


def probe_cluster(model, phrases, layer, device="cuda", top_n=30, min_phrase_ratio=0.5,
                   delay=0.5):
    """Probe all phrases in a cluster and find shared features.

    Returns dict with shared features ranked by consistency * strength.
    """
    n_features = model.n_features
    results = []

    for i, text in enumerate(phrases):
        if i > 0 and delay > 0:
            time.sleep(delay)

        # Retry with backoff on failure
        activations = None
        for attempt in range(3):
            try:
                activations = collect_activations_for_text(text, layer)
                if activations is not None:
                    break
            except Exception as e:
                print(f"      attempt {attempt+1} error: {e}")
            time.sleep(2 * (attempt + 1))

        if activations is None:
            print(f"      (failed after retries)")
            continue

        tensor = torch.from_numpy(activations).float().to(device)
        with torch.no_grad():
            features = model.encode(tensor)
        features_np = features.cpu().numpy()

        mean_act = features_np.mean(axis=0)
        max_act = features_np.max(axis=0)
        freq = (features_np > 0).mean(axis=0)

        top_by_mean = np.argsort(-mean_act)[:top_n]
        results.append({
            "mean_activation": mean_act,
            "max_activation": max_act,
            "frequency": freq,
            "top_by_mean": top_by_mean,
            "n_tokens": len(activations),
        })

    if len(results) < 2:
        return None

    # Count how many phrases have each feature in their top-N
    feature_phrase_count = np.zeros(n_features)
    feature_total_mean = np.zeros(n_features)

    for r in results:
        for idx in r["top_by_mean"][:top_n]:
            feature_phrase_count[idx] += 1
        feature_total_mean += r["mean_activation"]

    n_phrases = len(results)
    feature_avg_mean = feature_total_mean / n_phrases

    min_count = max(2, int(n_phrases * min_phrase_ratio))
    shared_mask = feature_phrase_count >= min_count
    shared_indices = np.where(shared_mask)[0]

    shared_scores = feature_avg_mean[shared_indices]
    rank_order = np.argsort(-shared_scores)
    ranked_shared = shared_indices[rank_order]

    shared_features = []
    for idx in ranked_shared[:top_n]:
        count = int(feature_phrase_count[idx])
        avg_mean = float(feature_avg_mean[idx])
        score = (count / n_phrases) * avg_mean
        shared_features.append({
            "feature": int(idx),
            "phrase_count": count,
            "total_phrases": n_phrases,
            "avg_mean": avg_mean,
            "score": score,
        })

    return {
        "n_probed": len(phrases),
        "n_successful": n_phrases,
        "min_agreement": min_count,
        "shared_features": shared_features,
        "feature_avg_mean": feature_avg_mean,  # full vector for differential analysis
    }


def main():
    parser = argparse.ArgumentParser(description="Batch re-probe clusters against new SAE")
    parser.add_argument("--model", type=str, required=True, help="SAE model path (.pt)")
    parser.add_argument("--output", type=str, default="sae_steering.json",
                        help="Output steering config path")
    parser.add_argument("--cluster-dir", type=str, default="sae_data/clusters",
                        help="Directory with cluster JSON files")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-features", type=int, default=5,
                        help="Number of top features per cluster for steering (default 5)")
    parser.add_argument("--default-scale", type=float, default=0.02,
                        help="Default steering scale for features (default 0.02)")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="Only probe these clusters (by label)")
    parser.add_argument("--merge", type=str, default=None,
                        help="Merge results into existing steering JSON instead of overwriting")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay in seconds between probes (default 0.5)")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Global baseline .npz file for differential scoring")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load SAE
    print(f"Loading SAE: {args.model}")
    meta_path = args.model.replace(".pt", ".json")
    with open(meta_path) as f:
        metadata = json.load(f)
    n_embd = metadata["n_embd"]
    n_features = metadata["n_features"]
    layer = metadata["layer_idx"]

    model = SparseAutoencoder(n_embd, n_features).to(args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.eval()
    print(f"  {n_embd} -> {n_features} features, layer {layer}, on {args.device}")

    # Load baseline for differential scoring
    global_baseline = None
    if args.baseline and os.path.exists(args.baseline):
        baseline_data = np.load(args.baseline)
        global_baseline = baseline_data["mean"]
        print(f"  Loaded global baseline: {args.baseline} (differential scoring enabled)")

    # Load clusters
    clusters = load_clusters(args.cluster_dir)
    if args.only:
        clusters = {k: v for k, v in clusters.items() if k in args.only}
    print(f"\nFound {len(clusters)} clusters: {', '.join(clusters.keys())}")

    # Probe each cluster
    all_results = {}

    # Load existing config if merging
    if args.merge and os.path.exists(args.merge):
        with open(args.merge) as f:
            steering_config = json.load(f)
        print(f"  Merging into existing config: {args.merge} ({len(steering_config.get('clusters', {}))} clusters)")
    else:
        steering_config = {
            "sae_model": args.model,
            "layer": layer,
            "n_embd": n_embd,
            "n_features": n_features,
            "clusters": {},
        }

    for label, data in clusters.items():
        phrases = data.get("phrases", [])
        print(f"\n{'='*60}")
        print(f"  Cluster: {label} ({len(phrases)} phrases)")
        print(f"{'='*60}")

        result = probe_cluster(model, phrases, layer, device=args.device, delay=args.delay)

        if result is None:
            print(f"  FAILED — not enough successful probes")
            all_results[label] = {"status": "failed"}
            continue

        shared = result["shared_features"]
        print(f"  Found {len(shared)} shared features "
              f"({result['n_successful']}/{result['n_probed']} phrases succeeded)")

        if shared:
            print(f"  Top features:")
            for sf in shared[:10]:
                print(f"    #{sf['feature']:>5d}  score={sf['score']:.6f}  "
                      f"({sf['phrase_count']}/{sf['total_phrases']} phrases)")

        # Add to steering config — use differential scoring if baseline available
        if global_baseline is not None and "feature_avg_mean" in result:
            diff = result["feature_avg_mean"] - global_baseline
            # Only consider features that are shared AND above baseline
            diff_features = []
            for sf in shared:
                fidx = sf["feature"]
                d = float(diff[fidx])
                if d > 0:
                    diff_features.append({
                        "feature": fidx,
                        "diff": d,
                        "abs_mean": sf["avg_mean"],
                        "phrase_count": sf["phrase_count"],
                        "total_phrases": sf["total_phrases"],
                    })
            # Sort by differential (how much MORE this cluster activates vs baseline)
            diff_features.sort(key=lambda x: -x["diff"])
            top = diff_features[:args.top_features]

            if top:
                print(f"  Differential top features (vs baseline):")
                for df in top[:10]:
                    print(f"    #{df['feature']:>5d}  diff=+{df['diff']:.6f}  "
                          f"abs={df['abs_mean']:.6f}  "
                          f"({df['phrase_count']}/{df['total_phrases']} phrases)")

            steering_config["clusters"][label] = {
                "features": [df["feature"] for df in top],
                "weights": [df["diff"] for df in top],
                "scale": args.default_scale,
                "n_shared": len(shared),
                "n_probed": result["n_successful"],
                "scoring": "differential",
            }
        else:
            top = shared[:args.top_features]
            steering_config["clusters"][label] = {
                "features": [sf["feature"] for sf in top],
                "weights": [sf["score"] for sf in top],
                "scale": args.default_scale,
                "n_shared": len(shared),
                "n_probed": result["n_successful"],
            }

        all_results[label] = result

    # Inter-cluster differential: find features unique to each cluster
    # by comparing each cluster's mean to the average of all other clusters
    cluster_means = {}
    for label, result in all_results.items():
        if isinstance(result, dict) and "feature_avg_mean" in result:
            cluster_means[label] = result["feature_avg_mean"]

    if len(cluster_means) >= 3:
        print(f"\n{'='*60}")
        print(f"  INTER-CLUSTER DIFFERENTIAL ANALYSIS")
        print(f"  ({len(cluster_means)} clusters)")
        print(f"{'='*60}")

        all_means = np.stack(list(cluster_means.values()))
        grand_mean = all_means.mean(axis=0)

        for label in cluster_means:
            cluster_mean = cluster_means[label]
            # Diff vs grand mean of all clusters
            diff = cluster_mean - grand_mean
            # Only features with positive diff (more active in this cluster)
            positive_mask = diff > 0
            if not positive_mask.any():
                continue

            # Rank by diff magnitude
            top_diff_indices = np.argsort(-diff)
            top_diff = []
            for idx in top_diff_indices[:args.top_features * 2]:
                d = float(diff[idx])
                if d <= 0:
                    break
                top_diff.append({
                    "feature": int(idx),
                    "diff_vs_others": d,
                    "abs_mean": float(cluster_mean[idx]),
                    "grand_mean": float(grand_mean[idx]),
                })

            if top_diff:
                print(f"\n  {label}:")
                for df in top_diff[:5]:
                    print(f"    #{df['feature']:>5d}  diff=+{df['diff_vs_others']:.6f}  "
                          f"cluster={df['abs_mean']:.6f}  others={df['grand_mean']:.6f}")

            # Update steering config with inter-cluster differential features
            top = top_diff[:args.top_features]
            steering_config["clusters"][label] = {
                "features": [df["feature"] for df in top],
                "weights": [df["diff_vs_others"] for df in top],
                "scale": args.default_scale,
                "n_shared": steering_config["clusters"].get(label, {}).get("n_shared", 0),
                "n_probed": steering_config["clusters"].get(label, {}).get("n_probed", 0),
                "scoring": "inter_cluster_differential",
            }

    # Save steering config
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(steering_config, f, indent=2)
    print(f"\nSteering config saved: {args.output}")

    # Save detailed results
    results_path = args.model.replace(".pt", "_cluster_results.json")
    serializable = {}
    for label, result in all_results.items():
        if isinstance(result, dict) and "shared_features" in result:
            serializable[label] = {
                "n_probed": result["n_probed"],
                "n_successful": result["n_successful"],
                "shared_features": result["shared_features"][:20],
            }
        else:
            serializable[label] = result
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Detailed results saved: {results_path}")

    # Summary
    n_ok = sum(1 for r in all_results.values()
               if isinstance(r, dict) and r.get("shared_features"))
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {n_ok}/{len(clusters)} clusters successfully probed")
    print(f"  Steering config: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
