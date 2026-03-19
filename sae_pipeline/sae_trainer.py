#!/usr/bin/env python3 -u
"""Train a Sparse Autoencoder (SAE) on collected activation data.

Learns a dictionary of interpretable features from activation vectors collected
by sae_collector.py. The trained SAE decomposes activation vectors into sparse
combinations of learned features, enabling fine-grained analysis of model behavior.

Usage:
    # Train on a single collection file
    python3 sae_trainer.py --input sae_data/activations_layer24_dataset_1000rows.bin

    # Train on multiple files (e.g., multiple datasets combined)
    python3 sae_trainer.py --input sae_data/activations_layer24_*.bin

    # Custom expansion factor and sparsity
    python3 sae_trainer.py --input sae_data/*.bin --expansion 8 --l1-coeff 3e-4

    # Resume from a checkpoint
    python3 sae_trainer.py --input sae_data/*.bin --resume sae_models/sae_layer24_latest.pt

Output:
    sae_models/sae_layer{L}_{n_features}f_{timestamp}.pt  — trained model
    sae_models/sae_layer{L}_{n_features}f_{timestamp}.json — metadata + training stats
"""

import argparse
import glob
import json
import os
import struct
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_OUTPUT_DIR = "sae_models"


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder for activation decomposition.

    Architecture: input -> encoder (ReLU) -> decoder
    The encoder expands the input into a higher-dimensional sparse representation.
    The decoder reconstructs the original activation from the sparse features.

    Loss = reconstruction_loss + l1_coeff * sparsity_loss
    """

    def __init__(self, n_input, n_features):
        super().__init__()
        self.n_input = n_input
        self.n_features = n_features

        # Encoder: project to higher-dim sparse space
        self.encoder = nn.Linear(n_input, n_features)

        # Decoder: reconstruct from sparse features
        # Tied weights would be decoder.weight = encoder.weight.T
        # but untied gives better results for interpretability
        self.decoder = nn.Linear(n_features, n_input, bias=False)

        # Pre-encoder bias (subtract mean activation)
        self.pre_bias = nn.Parameter(torch.zeros(n_input))

        self._init_weights()

    def _init_weights(self):
        """Initialize decoder columns as unit vectors."""
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        # Initialize decoder weights as unit-norm columns
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
            # Normalize decoder columns
            norms = self.decoder.weight.norm(dim=0, keepdim=True)
            self.decoder.weight.div_(norms.clamp(min=1e-8))

    def encode(self, x):
        """Encode input to sparse feature activations."""
        return torch.relu(self.encoder(x - self.pre_bias))

    def decode(self, features):
        """Decode sparse features back to activation space."""
        return self.decoder(features) + self.pre_bias

    def forward(self, x):
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features


def read_activation_header(path):
    """Read the header from a binary activation file. Returns (n_embd, layer_idx, n_tokens)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"ACTV":
            raise ValueError(f"Bad magic in {path}: {magic}")
        version, n_embd, layer_idx = struct.unpack("<III", f.read(12))

    file_size = os.path.getsize(path)
    data_bytes = file_size - 16
    n_tokens = data_bytes // (n_embd * 4)
    return n_embd, layer_idx, n_tokens


def load_activation_file(path):
    """Load a binary activation file using memory mapping for efficiency.
    Returns (data, n_embd, layer_idx). Data is a memory-mapped numpy array."""
    n_embd, layer_idx, n_tokens = read_activation_header(path)

    # Memory-map the file — reads from disk on demand, doesn't load into RAM
    data = np.memmap(path, dtype=np.float32, mode='r', offset=16,
                     shape=(n_tokens, n_embd))
    return data, n_embd, layer_idx


def load_all_activations(paths):
    """Load multiple activation files. Returns list of memmap arrays + metadata.
    Does NOT concatenate — keeps files as separate memmaps to avoid RAM usage."""
    arrays = []
    n_embd = None
    layer_idx = None
    total_tokens = 0

    for path in paths:
        file_embd, file_layer, file_tokens = read_activation_header(path)
        print(f"  {path}: {file_tokens:,} tokens, n_embd={file_embd}, layer={file_layer}")

        if n_embd is None:
            n_embd = file_embd
            layer_idx = file_layer
        else:
            if file_embd != n_embd:
                print(f"  WARNING: n_embd mismatch ({file_embd} vs {n_embd}), skipping")
                continue
            if file_layer != layer_idx:
                print(f"  Note: mixing layers ({file_layer} and {layer_idx})")

        data = np.memmap(path, dtype=np.float32, mode='r', offset=16,
                         shape=(file_tokens, n_embd))
        arrays.append(data)
        total_tokens += file_tokens

    print(f"  Total: {total_tokens:,} tokens, n_embd={n_embd}")
    return arrays, n_embd, layer_idx, total_tokens


class MemmapActivationDataset(torch.utils.data.Dataset):
    """Dataset that indexes across multiple memory-mapped activation files.
    Loads data on-demand from disk — no RAM overhead."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.cumulative = []
        total = 0
        for arr in arrays:
            total += len(arr)
            self.cumulative.append(total)
        self.total = total

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Find which array this index falls into
        for i, cum in enumerate(self.cumulative):
            if idx < cum:
                local_idx = idx - (self.cumulative[i-1] if i > 0 else 0)
                # Copy from memmap to regular array, then to tensor
                return torch.from_numpy(self.arrays[i][local_idx].copy()).float()
        raise IndexError(f"Index {idx} out of range")


def train_sae(activation_arrays, n_tokens, n_embd, n_features, l1_coeff=1e-4,
              lr=3e-4, batch_size=4096, epochs=5, device="cuda",
              log_interval=100, checkpoint_path=None):
    """Train a sparse autoencoder on activation data.

    Args:
        activation_arrays: list of numpy memmap arrays
        n_tokens: total number of tokens across all arrays
        n_embd: embedding dimension
        n_features: number of SAE features (expansion factor * n_embd)
        l1_coeff: L1 sparsity coefficient
        lr: learning rate
        batch_size: training batch size
        epochs: number of training epochs
        device: torch device
        log_interval: steps between log messages
        checkpoint_path: optional path to resume from

    Returns:
        (model, stats_dict)
    """
    # Compute pre-bias from a sample (don't load everything for mean)
    print("  Computing mean activation from sample...")
    sample_size = min(50000, n_tokens)
    sample_indices = np.random.choice(n_tokens, sample_size, replace=False)
    sample_indices.sort()

    # Gather samples from memmap arrays
    cumulative = []
    total = 0
    for arr in activation_arrays:
        total += len(arr)
        cumulative.append(total)

    sample_sum = np.zeros(n_embd, dtype=np.float64)
    for idx in sample_indices:
        for i, cum in enumerate(cumulative):
            if idx < cum:
                local_idx = idx - (cumulative[i-1] if i > 0 else 0)
                sample_sum += activation_arrays[i][local_idx].astype(np.float64)
                break
    mean_activation = (sample_sum / sample_size).astype(np.float32)

    # Create dataset and loader
    dataset = MemmapActivationDataset(activation_arrays)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, prefetch_factor=4)

    # Create model
    model = SparseAutoencoder(n_embd, n_features).to(device)
    with torch.no_grad():
        model.pre_bias.copy_(torch.from_numpy(mean_activation))

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Resuming from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))

    # Training loop
    total_steps = 0
    stats = {
        "losses": [],
        "recon_losses": [],
        "l1_losses": [],
        "sparsity": [],  # avg number of active features per token
        "dead_features": [],  # features that never activate
    }

    print(f"\n  Training SAE: {n_embd} -> {n_features} features")
    print(f"  {n_tokens:,} tokens, batch={batch_size}, epochs={epochs}")
    print(f"  L1 coeff={l1_coeff}, lr={lr}")
    print()

    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_l1 = 0.0
        epoch_sparsity = 0.0
        n_batches = 0

        # Track feature activation counts for dead feature detection
        feature_counts = torch.zeros(n_features, device=device)

        for batch in loader:
            batch = batch.to(device)

            reconstruction, features = model(batch)

            # Reconstruction loss (MSE)
            recon_loss = ((reconstruction - batch) ** 2).mean()

            # Sparsity loss (L1 on feature activations)
            l1_loss = features.abs().mean()

            loss = recon_loss + l1_coeff * l1_loss

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Normalize decoder weights to unit norm (important for interpretability)
            with torch.no_grad():
                norms = model.decoder.weight.norm(dim=0, keepdim=True)
                model.decoder.weight.div_(norms.clamp(min=1e-8))

            # Track stats
            active = (features > 0).float()
            feature_counts += active.sum(dim=0)
            batch_sparsity = active.sum(dim=1).mean().item()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_l1 += l1_loss.item()
            epoch_sparsity += batch_sparsity
            n_batches += 1
            total_steps += 1

            if total_steps % log_interval == 0:
                dead = (feature_counts == 0).sum().item()
                elapsed = time.time() - start_time
                print(f"  Step {total_steps:5d} | loss={loss.item():.6f} "
                      f"recon={recon_loss.item():.6f} l1={l1_loss.item():.6f} "
                      f"sparsity={batch_sparsity:.1f}/{n_features} "
                      f"dead={dead} | {elapsed:.0f}s")

        # Epoch summary
        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_l1 = epoch_l1 / n_batches
        avg_sparsity = epoch_sparsity / n_batches
        dead = (feature_counts == 0).sum().item()

        stats["losses"].append(avg_loss)
        stats["recon_losses"].append(avg_recon)
        stats["l1_losses"].append(avg_l1)
        stats["sparsity"].append(avg_sparsity)
        stats["dead_features"].append(dead)

        elapsed = time.time() - start_time
        print(f"\n  Epoch {epoch+1}/{epochs} | loss={avg_loss:.6f} "
              f"recon={avg_recon:.6f} l1={avg_l1:.6f} "
              f"active={avg_sparsity:.1f}/{n_features} "
              f"dead={dead}/{n_features} ({dead/n_features*100:.1f}%) "
              f"| {elapsed:.0f}s\n")

    total_time = time.time() - start_time
    stats["total_time_s"] = total_time
    stats["total_steps"] = total_steps

    return model, stats


def analyze_features(model, activation_arrays, n_tokens, top_n=20, device="cuda"):
    """Quick analysis of learned features after training.
    Uses a sample of activations to avoid loading everything into RAM."""
    model.eval()

    # Sample up to 50K tokens for analysis
    sample_size = min(50000, n_tokens)
    dataset = MemmapActivationDataset(activation_arrays)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=2)

    all_features = []
    collected = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            features = model.encode(batch)
            all_features.append(features.cpu())
            collected += len(batch)
            if collected >= sample_size:
                break

    features_np = torch.cat(all_features, dim=0).numpy()

    # Feature activation statistics
    mean_activation = features_np.mean(axis=0)
    max_activation = features_np.max(axis=0)
    activation_freq = (features_np > 0).mean(axis=0)  # fraction of tokens where feature fires

    # Sort by activation frequency (most common features)
    freq_order = np.argsort(-activation_freq)

    print(f"\nTop {top_n} most frequent features:")
    print(f"  {'Feature':>8}  {'Freq':>8}  {'Mean':>10}  {'Max':>10}")
    for i in range(top_n):
        idx = freq_order[i]
        print(f"  {idx:>8d}  {activation_freq[idx]:>8.4f}  "
              f"{mean_activation[idx]:>10.4f}  {max_activation[idx]:>10.4f}")

    # Dead features (never activate)
    dead = (activation_freq == 0).sum()
    ultra_rare = (activation_freq < 0.001).sum()

    print(f"\nFeature health:")
    print(f"  Total features:     {len(activation_freq):,}")
    print(f"  Dead (0% active):   {dead:,} ({dead/len(activation_freq)*100:.1f}%)")
    print(f"  Ultra-rare (<0.1%): {ultra_rare:,} ({ultra_rare/len(activation_freq)*100:.1f}%)")
    print(f"  Avg active/token:   {(features_np > 0).sum(axis=1).mean():.1f}")

    return {
        "mean_activation": mean_activation.tolist(),
        "max_activation": max_activation.tolist(),
        "activation_freq": activation_freq.tolist(),
        "dead_features": int(dead),
        "ultra_rare_features": int(ultra_rare),
        "avg_active_per_token": float((features_np > 0).sum(axis=1).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Train SAE on collected activations")
    parser.add_argument("--input", type=str, required=True, nargs="+",
                        help="Input binary activation file(s) (supports glob)")
    parser.add_argument("--expansion", type=int, default=8,
                        help="Expansion factor (n_features = expansion * n_embd, default 8)")
    parser.add_argument("--l1-coeff", type=float, default=1e-4,
                        help="L1 sparsity coefficient (default 1e-4)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default 3e-4)")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Batch size (default 4096)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (default 5)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for trained models")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--analyze", action="store_true",
                        help="Run feature analysis after training")
    args = parser.parse_args()

    # Expand globs in input paths
    input_paths = []
    for pattern in args.input:
        expanded = sorted(glob.glob(pattern))
        if not expanded:
            print(f"Warning: no files match '{pattern}'")
        input_paths.extend(expanded)

    if not input_paths:
        print("Error: no input files found")
        sys.exit(1)

    print(f"Loading {len(input_paths)} activation file(s) (memory-mapped):")
    activation_arrays, n_embd, layer_idx, n_tokens = load_all_activations(input_paths)

    n_features = args.expansion * n_embd
    print(f"\nSAE config: {n_embd} -> {n_features} features (expansion={args.expansion}x)")

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.device == "cuda":
        vram_needed_mb = (n_embd * n_features * 4 * 3) / (1024 * 1024)  # rough: weights + grads + optimizer
        print(f"  Estimated VRAM: ~{vram_needed_mb:.0f} MB")

    # Train
    model, stats = train_sae(
        activation_arrays,
        n_tokens=n_tokens,
        n_embd=n_embd,
        n_features=n_features,
        l1_coeff=args.l1_coeff,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        checkpoint_path=args.resume,
    )

    # Save model + metadata
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"sae_layer{layer_idx}_{n_features}f_{timestamp}"

    model_path = os.path.join(args.output_dir, f"{base_name}.pt")
    meta_path = os.path.join(args.output_dir, f"{base_name}.json")

    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")

    # Feature analysis
    feature_stats = None
    if args.analyze:
        print("\nRunning feature analysis...")
        feature_stats = analyze_features(model, activation_arrays, n_tokens, device=args.device)

    # Save metadata
    metadata = {
        "model_path": model_path,
        "layer_idx": int(layer_idx),
        "n_embd": int(n_embd),
        "n_features": n_features,
        "expansion": args.expansion,
        "n_tokens": n_tokens,
        "l1_coeff": args.l1_coeff,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "input_files": input_paths,
        "training_stats": {
            "final_loss": stats["losses"][-1],
            "final_recon_loss": stats["recon_losses"][-1],
            "final_l1_loss": stats["l1_losses"][-1],
            "final_sparsity": stats["sparsity"][-1],
            "final_dead_features": stats["dead_features"][-1],
            "total_steps": stats["total_steps"],
            "total_time_s": stats["total_time_s"],
        },
        "timestamp": timestamp,
    }
    if feature_stats:
        metadata["feature_analysis"] = {
            "dead_features": feature_stats["dead_features"],
            "ultra_rare_features": feature_stats["ultra_rare_features"],
            "avg_active_per_token": feature_stats["avg_active_per_token"],
        }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {meta_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SAE Training Complete")
    print(f"{'='*60}")
    print(f"  Layer:           {layer_idx}")
    print(f"  Features:        {n_features:,} ({args.expansion}x expansion)")
    print(f"  Tokens trained:  {n_tokens:,}")
    print(f"  Final loss:      {stats['losses'][-1]:.6f}")
    print(f"  Final recon:     {stats['recon_losses'][-1]:.6f}")
    print(f"  Dead features:   {stats['dead_features'][-1]:,}/{n_features}")
    print(f"  Training time:   {stats['total_time_s']:.0f}s")
    print(f"  Model size:      {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    print(f"  Output:          {model_path}")


if __name__ == "__main__":
    main()
