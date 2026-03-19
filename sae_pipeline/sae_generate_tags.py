#!/usr/bin/env python3 -u
"""Generate enriched feature-to-cluster tags from batch reprobe results.

Reads the steering config (with inter-cluster differential features) and
cluster results (with shared feature details) to produce a comprehensive
feature_cluster_tags.json with confidence scoring.

Usage:
    # Generate from the full steering config
    python3 sae_generate_tags.py \
        --steering sae_steering.json \
        --results sae_models/sae_layer18_16384f_20260307_080746_cluster_results.json

    # Override max features per cluster (default: all shared features)
    python3 sae_generate_tags.py \
        --steering sae_steering.json \
        --results sae_models/sae_layer18_16384f_20260307_080746_cluster_results.json \
        --max-per-cluster 15

Output:
    sae_models/feature_cluster_tags.json
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Generate enriched SAE feature tags")
    parser.add_argument("--steering", type=str, required=True,
                        help="Path to full steering config JSON (with clusters.features/weights)")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to cluster results JSON (with shared_features details)")
    parser.add_argument("--output", type=str, default="sae_models/feature_cluster_tags.json",
                        help="Output path (default: sae_models/feature_cluster_tags.json)")
    parser.add_argument("--max-per-cluster", type=int, default=20,
                        help="Max features to include per cluster (default: 20)")
    args = parser.parse_args()

    # Load steering config
    with open(args.steering) as f:
        steering = json.load(f)

    clusters = steering.get("clusters", {})
    if not clusters:
        print("Error: no clusters found in steering config")
        sys.exit(1)

    sae_model = steering.get("sae_model", "")
    layer = steering.get("layer", 0)
    n_embd = steering.get("n_embd", 0)
    n_features = steering.get("n_features", 0)

    # Load cluster results if provided (gives us phrase_count data for consistency scoring)
    results = {}
    if args.results and os.path.exists(args.results):
        with open(args.results) as f:
            results = json.load(f)
        print(f"Loaded cluster results: {len(results)} clusters")

    # Build feature -> cluster mapping
    # A feature can appear in multiple clusters; we track all of them
    feature_map = {}  # feature_id_str -> list of cluster entries

    for label, cluster_data in clusters.items():
        steering_features = set(cluster_data.get("features", []))
        steering_weights = cluster_data.get("weights", [])
        steering_feat_list = cluster_data.get("features", [])
        n_probed = cluster_data.get("n_probed", 0)
        scoring = cluster_data.get("scoring", "unknown")

        # Get shared feature details from results if available
        result_shared = {}
        if label in results and "shared_features" in results[label]:
            for sf in results[label]["shared_features"]:
                result_shared[sf["feature"]] = sf

        # Merge: start with steering features (have inter-cluster differential weights),
        # then add any additional features from cluster results
        all_feat_ids = list(steering_feat_list[:args.max_per_cluster])
        if result_shared:
            for sf in results[label]["shared_features"][:args.max_per_cluster]:
                if sf["feature"] not in steering_features:
                    all_feat_ids.append(sf["feature"])
            all_feat_ids = all_feat_ids[:args.max_per_cluster]

        for feat_id in all_feat_ids:
            key = str(feat_id)

            # Get weight from steering (inter-cluster differential)
            if feat_id in steering_features:
                idx = steering_feat_list.index(feat_id)
                weight = steering_weights[idx] if idx < len(steering_weights) else 0
            else:
                weight = 0

            # Get consistency from cluster results (if this feature appears there)
            sf_detail = result_shared.get(feat_id, {})
            phrase_count = sf_detail.get("phrase_count", 0)
            total_phrases = sf_detail.get("total_phrases", n_probed)
            consistency = phrase_count / total_phrases if total_phrases > 0 else 0

            # For inter-cluster differential, weight IS the diff_vs_others
            diff_vs_others = weight if scoring == "inter_cluster_differential" else 0

            # Use score from cluster results as fallback diff estimate
            if diff_vs_others == 0 and sf_detail:
                diff_vs_others = sf_detail.get("score", 0)

            # Get cluster mean from results
            cluster_mean = sf_detail.get("avg_mean", 0)
            others_mean = max(0, cluster_mean - diff_vs_others) if cluster_mean > 0 else 0

            # Specificity: how unique is this feature to this cluster?
            specificity = diff_vs_others / others_mean if others_mean > 0 else (1.0 if diff_vs_others > 0 else 0)

            # Confidence scoring:
            # - Features from inter-cluster differential ARE validated by the differential
            #   analysis itself (they fire more in this cluster than in any other). If we
            #   also have phrase-level consistency data, great — use it. If not, the
            #   differential weight alone is meaningful evidence.
            # - Features only from cluster results use consistency * specificity.
            is_differential = feat_id in steering_features and scoring == "inter_cluster_differential"
            if consistency > 0:
                # Have both differential and phrase data — combine them
                confidence = specificity * consistency
            elif is_differential and diff_vs_others > 0:
                # Differential-only feature: use specificity directly as confidence
                # (these passed the inter-cluster filter, so they ARE unique to this cluster)
                confidence = specificity
            else:
                confidence = 0

            entry = {
                "label": label,
                "diff_vs_others": round(diff_vs_others, 6),
                "cluster_mean": round(cluster_mean, 6),
                "others_mean": round(others_mean, 6),
                "consistency": round(consistency, 4),
                "specificity": round(specificity, 4),
                "confidence": round(confidence, 4),
            }

            if key not in feature_map:
                feature_map[key] = []
            feature_map[key].append(entry)

    # Build final tags structure
    features_out = {}
    for fid, cluster_entries in feature_map.items():
        # Primary cluster = the one with highest confidence (or diff if no consistency data)
        best = max(cluster_entries, key=lambda e: e["confidence"] if e["confidence"] > 0 else e["diff_vs_others"])
        features_out[fid] = {
            "clusters": cluster_entries,
            "primary_cluster": best["label"],
            "max_diff": max(e["diff_vs_others"] for e in cluster_entries),
            "max_confidence": max(e["confidence"] for e in cluster_entries),
        }

    output = {
        "sae_model": sae_model,
        "layer": layer,
        "n_embd": n_embd,
        "n_features": n_features,
        "n_features_mapped": len(features_out),
        "generated_from": "sae_generate_tags.py with inter-cluster differential + confidence scoring",
        "features": features_out,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    n_high = sum(1 for f in features_out.values() if f["max_confidence"] >= 0.10)
    n_med = sum(1 for f in features_out.values() if 0.05 <= f["max_confidence"] < 0.10)
    n_low = sum(1 for f in features_out.values() if f["max_confidence"] < 0.05)
    n_multi = sum(1 for f in features_out.values() if len(f["clusters"]) > 1)

    print(f"\nFeature tags generated: {args.output}")
    print(f"  Total features mapped: {len(features_out)}")
    print(f"  Clusters: {len(clusters)}")
    print(f"  Multi-cluster features: {n_multi}")
    print(f"  Confidence: high(>=0.10)={n_high}  med(0.05-0.10)={n_med}  low(<0.05)={n_low}")

    # Show per-cluster breakdown
    cluster_counts = {}
    for f in features_out.values():
        c = f["primary_cluster"]
        cluster_counts[c] = cluster_counts.get(c, 0) + 1
    print(f"\n  Per-cluster feature counts:")
    for c in sorted(cluster_counts.keys()):
        print(f"    {c}: {cluster_counts[c]}")


if __name__ == "__main__":
    main()
