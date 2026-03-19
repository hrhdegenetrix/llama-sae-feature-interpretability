#!/usr/bin/env python3 -u
"""Extract SAE feature directions as llama.cpp-compatible control vector GGUFs.

Each SAE feature corresponds to a column in the decoder weight matrix — a direction
in activation space. This tool extracts those directions and writes them as GGUF
control vectors that llama-server can load via --control-vector-scaled.

Usage:
    # Extract a single sycophancy feature as a control vector
    python3 sae_extract_vectors.py \
        --model sae_models/sae_layer24_16384f_20260305_114446.pt \
        --features 12846 \
        --name sycophancy_deference \
        --layer 24

    # Extract multiple features combined (e.g., a sycophancy cluster)
    python3 sae_extract_vectors.py \
        --model sae_models/sae_layer24_16384f_20260305_114446.pt \
        --features 12846,14126,4199,132,11294,13238 \
        --name sycophancy_cluster \
        --layer 24

    # Extract with custom weights per feature (e.g., stronger suppression of top features)
    python3 sae_extract_vectors.py \
        --model sae_models/sae_layer24_16384f_20260305_114446.pt \
        --features 12846:2.0,14126:1.5,4199:1.0 \
        --name sycophancy_weighted \
        --layer 24

    # List the top sycophancy/persona features from a differential analysis
    python3 sae_extract_vectors.py \
        --model sae_models/sae_layer24_16384f_20260305_114446.pt \
        --from-analysis sae_data/differential_analysis_layer24.json \
        --top-sycophancy 10 \
        --name anti_sycophancy \
        --layer 24

Apply the vector:
    # In start_llama.sh or manually:
    llama-server ... --control-vector-scaled /path/to/anti_sycophancy.gguf:-0.3 \
                     --control-vector-layer-range 24 24

    Negative scale = suppress the feature direction
    Positive scale = amplify the feature direction
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import gguf

from sae_trainer import SparseAutoencoder

DEFAULT_OUTPUT_DIR = "sae_vectors"


def extract_feature_direction(model, feature_idx):
    """Extract the decoder direction for a single SAE feature.

    Returns a numpy float32 array of shape (n_embd,).
    """
    with torch.no_grad():
        # Decoder weight shape: (n_input, n_features)
        # Column feature_idx is the direction in activation space
        direction = model.decoder.weight[:, feature_idx].cpu().numpy().astype(np.float32)
    return direction


def combine_directions(model, features_and_weights):
    """Combine multiple feature directions with weights.

    Args:
        features_and_weights: list of (feature_idx, weight) tuples

    Returns normalized combined direction vector.
    """
    n_input = model.decoder.weight.shape[0]
    combined = np.zeros(n_input, dtype=np.float32)

    for feat_idx, weight in features_and_weights:
        direction = extract_feature_direction(model, feat_idx)
        combined += weight * direction

    # Normalize to unit length
    norm = np.linalg.norm(combined)
    if norm > 1e-8:
        combined /= norm

    return combined


def write_control_vector_gguf(direction, layer, n_layers, output_path, model_hint=""):
    """Write a direction vector as a llama.cpp-compatible control vector GGUF.

    Creates a GGUF with direction tensors for each layer 1..n_layers.
    Only the target layer has actual data; others are zero vectors.
    """
    n_embd = len(direction)

    writer = gguf.GGUFWriter(output_path, "controlvector")
    writer.add_string("controlvector.model_hint", model_hint)
    writer.add_int32("controlvector.layer_count", n_layers)

    for il in range(1, n_layers + 1):
        if il == layer:
            writer.add_tensor(f"direction.{il}", direction)
        else:
            writer.add_tensor(f"direction.{il}", np.zeros(n_embd, dtype=np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def parse_feature_spec(spec_str):
    """Parse feature specification string.

    Formats:
        "12846" -> [(12846, 1.0)]
        "12846,14126,4199" -> [(12846, 1.0), (14126, 1.0), (4199, 1.0)]
        "12846:2.0,14126:1.5" -> [(12846, 2.0), (14126, 1.5)]
    """
    result = []
    for part in spec_str.split(","):
        part = part.strip()
        if ":" in part:
            feat_str, weight_str = part.split(":", 1)
            result.append((int(feat_str), float(weight_str)))
        else:
            result.append((int(part), 1.0))
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract SAE features as control vectors")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained SAE model (.pt)")
    parser.add_argument("--features", type=str, default=None,
                        help="Feature indices (comma-separated, optional :weight suffix)")
    parser.add_argument("--from-analysis", type=str, default=None,
                        help="Load features from differential analysis JSON")
    parser.add_argument("--top-sycophancy", type=int, default=0,
                        help="Number of top sycophancy features from analysis")
    parser.add_argument("--top-persona", type=int, default=0,
                        help="Number of top persona features from analysis")
    parser.add_argument("--name", type=str, default="sae_feature",
                        help="Output filename (without .gguf)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Target layer (auto-detected from model metadata)")
    parser.add_argument("--n-layers", type=int, default=None,
                        help="Total model layers (auto-detected from server or metadata)")
    parser.add_argument("--model-hint", type=str, default="qwen3",
                        help="Model hint for GGUF metadata")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for loading model (cpu is fine, no GPU needed)")
    args = parser.parse_args()

    # Load SAE model
    print(f"Loading SAE model: {args.model}")
    meta_path = args.model.replace(".pt", ".json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
        n_embd = metadata["n_embd"]
        n_features = metadata["n_features"]
        layer = metadata.get("layer_idx", 24)
    else:
        state = torch.load(args.model, map_location="cpu")
        n_features = state["encoder.weight"].shape[0]
        n_embd = state["encoder.weight"].shape[1]
        layer = 24

    if args.layer is not None:
        layer = args.layer

    model = SparseAutoencoder(n_embd, n_features).to(args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.eval()
    print(f"  {n_embd} -> {n_features} features, target layer {layer}")

    # Determine total model layers
    n_layers = args.n_layers
    if n_layers is None:
        try:
            import requests
            llama_base = os.environ.get("LLAMA_BASE", "http://127.0.0.1:8080")
            info = requests.get(f"{llama_base}/activations", timeout=3).json()
            n_layers = info.get("n_layers", 48)
            print(f"  Model layers from server: {n_layers}")
        except:
            n_layers = 48
            print(f"  Using default n_layers={n_layers} (server not reachable)")

    # Collect feature specs
    features_and_weights = []

    if args.features:
        features_and_weights = parse_feature_spec(args.features)

    if args.from_analysis:
        print(f"Loading analysis: {args.from_analysis}")
        with open(args.from_analysis) as f:
            analysis = json.load(f)

        if args.top_sycophancy > 0:
            syco_feats = analysis.get("sycophancy_candidates", [])[:args.top_sycophancy]
            for entry in syco_feats:
                feat_idx = entry["feature"]
                # Weight by frequency difference (stronger diff = higher weight)
                weight = abs(entry.get("freq_diff", 1.0))
                features_and_weights.append((feat_idx, weight))
            print(f"  Added {len(syco_feats)} sycophancy features")

        if args.top_persona > 0:
            persona_feats = analysis.get("persona_candidates", [])[:args.top_persona]
            for entry in persona_feats:
                feat_idx = entry["feature"]
                weight = abs(entry.get("freq_diff", 1.0))
                features_and_weights.append((feat_idx, weight))
            print(f"  Added {len(persona_feats)} persona features")

    if not features_and_weights:
        print("Error: No features specified. Use --features or --from-analysis + --top-sycophancy/--top-persona")
        sys.exit(1)

    # Print feature summary
    print(f"\nFeatures to extract:")
    for feat_idx, weight in features_and_weights:
        direction = extract_feature_direction(model, feat_idx)
        print(f"  Feature {feat_idx:>6d}: weight={weight:.3f}, "
              f"direction norm={np.linalg.norm(direction):.4f}")

    # Combine directions
    if len(features_and_weights) == 1:
        feat_idx, _ = features_and_weights[0]
        direction = extract_feature_direction(model, feat_idx)
        # Normalize to unit length
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction /= norm
        print(f"\nSingle feature direction extracted (normalized)")
    else:
        direction = combine_directions(model, features_and_weights)
        print(f"\nCombined {len(features_and_weights)} feature directions (normalized)")

    print(f"  Direction norm: {np.linalg.norm(direction):.6f}")
    print(f"  Direction shape: {direction.shape}")

    # Write GGUF
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.name}.gguf")

    print(f"\nWriting control vector GGUF: {output_path}")
    write_control_vector_gguf(direction, layer, n_layers, output_path,
                              model_hint=args.model_hint)

    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"  File size: {file_size_kb:.0f} KB")

    # Print usage instructions
    print(f"\nUsage:")
    print(f"  # Suppress (negative scale = push away from this direction):")
    print(f"  llama-server ... --control-vector-scaled {output_path}:-0.3 \\")
    print(f"                   --control-vector-layer-range {layer} {layer}")
    print(f"")
    print(f"  # Amplify (positive scale = push toward this direction):")
    print(f"  llama-server ... --control-vector-scaled {output_path}:0.3 \\")
    print(f"                   --control-vector-layer-range {layer} {layer}")
    print(f"")
    print(f"  # Add to trait_scales.json for automatic loading:")
    print(f"  Edit your steering config to include \"{args.name}\": -0.3")
    print(f"  Then copy the GGUF to your control vector directory to the cvectors/output dir")
    print(f"")
    print(f"  Scale guide: 0.1-0.3 = subtle, 0.3-0.6 = noticeable, 0.6+ = strong")
    print(f"  Start with small negative scales for suppression to keep output natural")


if __name__ == "__main__":
    main()
