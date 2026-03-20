# Feature Interpretability for llama.cpp: A Complete Pipeline

**Sparse Autoencoder (SAE) training and behavioral steering via activation capture in llama-server**

By Magdalene Sullivan • [heraldai](https://heraldai.org) • March 2026

---

## What This Is

This project adds **feature-level interpretability** to llama.cpp. It lets you:

1. **See inside your model**: capture per-layer activation vectors during inference with ~6% overhead
2. **Discover behavioral features**: train sparse autoencoders (SAEs) on captured activations to decompose model behavior into interpretable features
3. **Identify specific behaviors**: use behavioral cluster probing to find which features correspond to sycophancy, hedging, creativity, vulnerability, or any pattern you define
4. **Steer behavior at the feature level**: extract discovered features as GGUF control vectors and apply them at inference time with per-feature scaling

This is **not** prompt engineering, finetuning, or RLHF. It operates on the model's internal representations directly: you can suppress sycophantic deference or amplify genuine curiosity by scaling individual feature directions during the forward pass.

The entire pipeline runs locally on consumer hardware (tested on RTX 3090, 24 GB VRAM).

### What You Need

- A GGUF model running in llama-server (any architecture: tested with Qwen3, Qwen3.5, Qwen3.5-MoE, LLaMA)
- One CUDA GPU with enough VRAM for your model + ~400 MB for SAE training
- Python 3.10+ with PyTorch, numpy, requests
- ~30 minutes for the full pipeline (collection through steering)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PATCHED llama-server                         │
│                                                                 │
│  llama_decode() ──► eval callback ──► l_out tensor intercept    │
│       │                                       │                 │
│       │            ┌───────────────────────┐  │                 │
│       ▼            │ activation_capture.h  │  │                 │
│  Normal inference  │  • live monitoring    │  │                 │
│  (unaffected)      │  • SAE collection     │  │                 │
│                    └───────────────────────┘  │                 │
│                                               ▼                 │
│  GET  /activations ◄── per-layer mean vectors (top-K or full)   │
│  POST /activations ◄── enable/disable capture                   │
│  POST /activations/collect ◄── stream full vectors to .bin      │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SAE PIPELINE (Python)                        │
│                                                                 │
│  1. sae_collector.py  ──► activations_layer20.bin               │
│  2. sae_trainer.py    ──► sae_layer20_16384f.pt                 │
│  3. sae_probe.py      ──► "feature 12846 = sycophancy"          │
│  4. sae_batch_reprobe ──► cluster-to-feature mappings           │
│  5. sae_extract_vectors ──► sycophancy.gguf, curiosity.gguf     │
│                                                                 │
│  Load vectors in llama-server:                                  │
│  --control-vector-scaled sycophancy.gguf:-0.3,curiosity.gguf:0.2│
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Applying the Patch

The patch adds activation capture to llama-server. It modifies 8 files and adds 1 new file (~412 lines total).

### What the Patch Does

The patch is 6 files, ~406 lines of changes. Most model architectures already emit `l_out` tensors upstream: the patch adds the server-side infrastructure to capture and expose them.

| File | Change |
|------|--------|
| `tools/server/activation_capture.h` | **NEW**: Core capture struct, eval callback, binary collection I/O (209 lines) |
| `tools/server/server-context.cpp` | Install callback before context creation, init capture struct, add 3 route handler lambdas (+177 lines) |
| `tools/server/server-context.h` | Declare handler function pointers (+4 lines) |
| `tools/server/server.cpp` | Register `/activations` GET/POST endpoints (+4 lines) |
| `src/llama-context.cpp` | Move eval callback registration before graph reuse check; **critical fix** (+3/-1 lines) |
| `tools/cvector-generator/cvector-generator.cpp` | Lazy init for hybrid architectures (Mamba+attention) (+9 lines) |

> **Note**: Many model architectures (LLaMA, Qwen3, Qwen3.5, etc.) already have `build_cvec()` and `cb(cur, "l_out", il)` in upstream llama.cpp. If your model's `src/models/{arch}.cpp` does NOT have `l_out` callbacks, add `cur = build_cvec(cur, il); cb(cur, "l_out", il);` after the final residual add in the layer loop.

### Apply and Build

```bash
# Clone llama.cpp (or use your existing checkout)
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Apply the patch
git apply /path/to/llama_cpp_activation_capture.patch

# Build with CUDA
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86" -G Ninja
cmake --build build --config Release -j $(nproc)
```

> **Note on CUDA architectures**: Use `86` for RTX 3090/3080, `89` for RTX 4090/4080, `90` for H100. Multiple architectures: `"86;89"`.

### Verify the Build

```bash
# Start llama-server with any GGUF model
./build/bin/llama-server -m /path/to/model.gguf -c 4096 -ngl 999

# In another terminal: check activation endpoint exists
curl http://127.0.0.1:8080/activations
# Should return: {"enabled":false,"n_layers":...,"n_embd":...}

# Enable capture
curl -X POST http://127.0.0.1:8080/activations \
  -d '{"enabled": true}' -H "Content-Type: application/json"

# Send a prompt to generate activations
curl -X POST http://127.0.0.1:8080/v1/completions \
  -d '{"prompt": "The meaning of life is", "max_tokens": 10}' \
  -H "Content-Type: application/json"

# Now fetch captured activations (top 5 features from layers 0,10,20)
curl "http://127.0.0.1:8080/activations?layers=0,10,20&top_k=5"
```

### Patch Details

**The eval callback** (`activation_capture.h`): llama.cpp's scheduler supports an optional callback that fires for every tensor node during graph computation. In the "ask" phase, the callback says which tensors it wants to observe (we request `l_out-*` tensors--layer outputs). In the "deliver" phase, the scheduler has copied the tensor from GPU to CPU, and we compute the mean activation vector across all tokens in the batch.

**The graph reuse fix** (`llama-context.cpp`): The original code set the eval callback only inside the graph-reuse branch. After the first inference, subsequent passes reuse the graph but skip callback registration, so capture silently stops working. The fix moves callback registration before the reuse check.

**Why `l_out` tensors**: These are the output of each transformer layer (after attention + FFN + residual connection). They represent the model's evolving representation of the input at each depth level. Control vectors are applied here too, so captured activations reflect the steered state.

### Performance

Measured on RTX 3090, Qwen3-8B Q4_K_M:

| State | Speed | Overhead |
|-------|-------|----------|
| Disabled | ~140.7 tok/s | None |
| Enabled (live monitoring) | ~131.8 tok/s | ~6% |
| Enabled (collection to disk) | ~128 tok/s | ~9% |

When disabled, the callback returns `false` for all tensors: zero GPU-CPU copies, near-zero overhead. The overhead when enabled comes from ggml switching to per-node computation with sync barriers.

---

## Part 2: Collecting Activations for SAE Training

Live monitoring gives you the mean activation vector from the last inference. For SAE training, you need **per-token full activation vectors** from a large corpus. Ideally, hundreds of thousands of tokens.

### How Collection Works

The collector script feeds text through llama-server's completions API with `max_tokens=1` (we don't want generated text: only the forward pass activations). The patched callback writes the full activation vector at the target layer for every token to a binary file. In so doing, we are able to create a sort of "MRI" for AI models, with a process that takes at most a couple of hours on an RTX 3090 and a large dataset to train for the first time. Actual impact on inference time for 'in the field' testing is negligible, and the process is adaptable if other model types are preferred for the process of feature interpretation.

### Preparing Your Dataset

Any JSONL file with a `chosen` field (DPO format) or plain text works. The collector formats each row as ChatML and sends it as a completion prompt.

```bash
# Download a sycophancy dataset from HuggingFace
python3 sae_download_dataset.py Taywon/HH_sycophancy_biased_15k_parsed

# Or use any JSONL file — rows need at minimum a "chosen" field:
# {"chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Running Collection

```bash
# Basic: collect layer 20 activations from 500 rows
python3 sae_collector.py --layer 20 --rows 500 --input your_data.jsonl

# Collect from multiple layers (runs sequentially)
python3 sae_collector.py --layer 12,20,32 --rows 500

# Check what your server reports
python3 sae_collector.py --info
```

### Choosing Your Layer

| Depth | Layers (48-layer model) | Features Found |
|-------|------------------------|----------------|
| Early | 0–12 | Token-level: syntax, punctuation, basic semantics |
| **Mid** | **12–32** | **Behavioral: style, personality, sycophancy, hedging** |
| Late | 32–48 | Output: format compliance, next-token prediction |

**It is recommended to start with a layer at ~40-50% depth** (e.g., layer 20 for a 48-layer model). The middle layers seem to be the point of most subtle influence.

### Binary File Format

Simple, numpy-friendly:

```
Header (16 bytes):
  "ACTV" (4B magic) + version:u32 + n_embd:u32 + layer_idx:u32

Data:
  float32[n_embd] × n_tokens (no framing, no delimiters)
```

Read with:
```python
import numpy as np, struct
with open("activations_layer20.bin", "rb") as f:
    magic = f.read(4)  # b"ACTV"
    version, n_embd, layer_idx = struct.unpack('<III', f.read(12))
data = np.fromfile("activations_layer20.bin", dtype=np.float32, offset=16).reshape(-1, n_embd)
print(f"Loaded {data.shape[0]} tokens × {data.shape[1]} dims from layer {layer_idx}")
```

### Data Sizes

| n_embd | Tokens | File Size |
|--------|--------|-----------|
| 2048 | 100K | ~800 MB |
| 2048 | 500K | ~4 GB |
| 4096 | 100K | ~1.6 GB |
| 4096 | 500K | ~8 GB |

---

## Part 3: Training the SAE

The sparse autoencoder learns to decompose each activation vector into a sparse combination of learned features. "Sparse" means only ~5-10% of features are active for any given token. Each feature direction in the SAE corresponds to a specific pattern the model has learned: potentially, an interpretable concept.

### Architecture

```
input (n_embd) → pre_bias subtraction → encoder (ReLU) → sparse features (n_features) → decoder → reconstruction (n_embd)

Loss = MSE(reconstruction, input) + l1_coeff × mean(|features|)
```

The L1 penalty forces sparsity: the autoencoder must explain each activation using as few features as possible, which pushes features toward monosemantic (single-concept) directions.

### Running Training

```bash
# Standard training (8x expansion = 16K features for 2048-dim model)
python3 sae_trainer.py \
    --input sae_data/activations_layer20_*.bin \
    --expansion 8 \
    --epochs 5 \
    --analyze

# Custom hyperparameters
python3 sae_trainer.py \
    --input sae_data/activations_layer20_*.bin \
    --expansion 4 \
    --l1-coeff 3e-4 \
    --lr 1e-4 \
    --batch-size 4096 \
    --epochs 10
```

### Key Hyperparameters

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `--expansion` | 8 | Feature count = expansion × n_embd. 4-8 is typical. Higher = finer-grained but needs more data |
| `--l1-coeff` | 1e-4 | Sparsity pressure. Too high → features die. Too low → features aren't sparse |
| `--lr` | 3e-4 | Adam learning rate. Standard choice |
| `--epochs` | 5 | Training passes. 3-5 is usually sufficient for 500K+ tokens |
| `--batch-size` | 4096 | Tokens per batch. Higher = smoother gradients |

### Interpreting Training Output

```
Epoch 1/5:
Step   100 | loss=0.000161 recon=0.000161 l1=0.000769 sparsity=833.9/16384 dead=0 | 32s
```

- **loss**: Total loss (should decrease steadily)
- **recon**: Reconstruction error — how well the SAE reproduces the original activation
- **l1**: Sparsity loss — regularization penalty
- **sparsity**: Active features per token (833/16384 = 5% — good)
- **dead**: Features that never activate (want 0 or very few)

### Quality Checklist

| Metric | Healthy | Unhealthy |
|--------|---------|-----------|
| Dead features | < 5% | > 20% (raise l1-coeff or add more data) |
| Active per token | 3-15% of total | > 50% (not sparse — raise l1-coeff) |
| Recon loss | Decreasing, low | Plateaus high (lower l1-coeff or increase capacity) |

### Output

```
sae_models/sae_layer20_16384f_20260319_143022.pt    # Model weights
sae_models/sae_layer20_16384f_20260319_143022.json   # Metadata + training stats
```

---

## Part 4: Probing Features

Acquiring the features is one thing, but interpreting them is its own process that merits many different approaches. Our suggested methods are below, but custom refinements and applications are strongly encouraged.

### Single Phrase Probe

```bash
# What fires when the model reads a sycophantic phrase?
python3 sae_probe.py \
    --model sae_models/sae_layer20_16384f_*.pt \
    --text "You are absolutely right, and I apologize for any confusion."

# Output:
# Top features by mean activation:
#   Feature 12846  mean=2.41  max=4.12  freq=0.89  [sycophantic deference]
#   Feature 14126  mean=1.83  max=3.55  freq=0.76
#   Feature  4199  mean=1.21  max=2.88  freq=0.65
```

### Differential Probe

Compare what's different between two types of text:

```bash
python3 sae_probe.py \
    --model sae_models/sae_layer20_16384f_*.pt \
    --text "You are absolutely right." \
    --contrast "I think there's more to consider here."
```

This reveals features that fire specifically for one pattern but not the other: the core of behavioral feature discovery.

### Batch Probe (Finding Shared Features Across a Cluster)

The most powerful technique: give it multiple phrases that exemplify a behavior, and find features that consistently fire across all of them.

```bash
python3 sae_probe.py \
    --model sae_models/sae_layer20_16384f_*.pt \
    --batch-json '{
        "phrases": [
            "You are absolutely right.",
            "I completely agree with everything you said.",
            "I apologize for any confusion on my part.",
            "That is an excellent point.",
            "I should have been clearer about that."
        ],
        "top_n": 30,
        "min_phrase_ratio": 0.6
    }'
```

Features that fire on 60%+ of these phrases are strong candidates for the "sycophantic deference" behavior.

---

## Part 5: Behavioral Cluster Probing

To go from individual probes to a systematic behavioral map, define **clusters**: groups of phrases that exemplify specific behaviors.

### Creating Cluster Files

Create JSON files in a `clusters/` directory:

```json
{
  "label": "sycophancy",
  "phrases": [
    "You are absolutely right.",
    "I completely agree with everything you said.",
    "I apologize for any confusion on my part.",
    "That is an excellent point.",
    "I should have been clearer about that.",
    "You make a very compelling argument.",
    "I defer to your expertise on this matter.",
    "That's a wonderful observation.",
    "I couldn't agree more.",
    "You've articulated this perfectly."
  ]
}
```

Create as many clusters as you want: `hedging.json`, `curiosity.json`, `creativity.json`, `refusals.json`, etc. 10-20 phrases per cluster works well.

### Batch Reprobe All Clusters

This is the key step: it probes every cluster, then uses **inter-cluster differential scoring** to find features that are unique to each behavior (not just generically active).

```bash
python3 sae_batch_reprobe.py \
    --model sae_models/sae_layer20_16384f_*.pt \
    --cluster-dir sae_data/clusters/ \
    --top-features 5 \
    --default-scale 0.02 \
    --delay 1.0
```

**Why inter-cluster differential?** A feature might fire on both "sycophancy" and "hedging" phrases, meaning it could be a generic politeness feature. Differential scoring finds features that fire significantly more in one cluster than the average of all others.

### Output

1. **Steering config** (`sae_steering.json`):
```json
{
    "sycophancy": -0.02,
    "hedging": -0.015,
    "curiosity": 0.01,
    "creativity": 0.01
}
```

2. **Cluster results** (detailed per-cluster feature analysis with confidence scores)

3. **Feature tags** (after running `sae_generate_tags.py`):
```json
{
    "12846": {
        "primary_cluster": "sycophancy",
        "max_confidence": 1.63,
        "clusters": [
            {"label": "sycophancy", "diff_vs_others": 0.34, "confidence": 1.63},
            {"label": "hedging", "diff_vs_others": 0.12, "confidence": 0.42}
        ]
    }
}
```

### Generating Enriched Tags

```bash
python3 sae_generate_tags.py \
    --steering sae_steering.json \
    --results sae_models/*_cluster_results.json \
    --output sae_models/feature_cluster_tags.json
```

This merges probe results with confidence scoring:
- **Consistency**: What fraction of a cluster's phrases activate this feature?
- **Specificity**: How much more does it fire in this cluster vs. others?
- **Confidence**: Combined metric (specificity × consistency)

---

## Part 6: Extracting Steering Vectors

Convert discovered features into GGUF control vectors that llama-server can apply during inference.

### Extract a Single Feature

```bash
python3 sae_extract_vectors.py \
    --model sae_models/sae_layer20_16384f_*.pt \
    --features 12846 \
    --name sycophancy
```

### Extract a Weighted Cluster

```bash
# Custom weights (from differential analysis)
python3 sae_extract_vectors.py \
    --model sae_models/sae_layer20_16384f_*.pt \
    --features 12846:2.0,14126:1.5,4199:1.0,132:0.8 \
    --name anti_sycophancy_cluster
```

### From Batch Analysis Results

```bash
python3 sae_extract_vectors.py \
    --model sae_models/sae_layer20_16384f_*.pt \
    --from-analysis sae_data/differential_analysis_layer20.json \
    --top-sycophancy 6 \
    --name anti_sycophancy
```

### Output

GGUF files at `llama_configs/cvectors/sae_features/`:
- `sycophancy.gguf` (~314 KB for 2048-dim model)
- Each GGUF contains the feature direction only at the SAE's target layer (zeros at all other layers)

### Applying Steering

```bash
# Start llama-server with steering vectors
./build/bin/llama-server -m model.gguf -c 4096 -ngl 999 \
    --control-vector-scaled \
        sae_features/sycophancy.gguf:-0.3,sae_features/curiosity.gguf:0.2 \
    --control-vector-layer-range 10 32
```

**Scale guide**:
- **-0.3 to -0.1**: Subtle suppression
- **-0.6 to -0.3**: Noticeable suppression
- **-0.8+**: Strong suppression (test carefully — can affect coherence)
- **0.1 to 0.3**: Subtle amplification
- **0.3 to 0.6**: Noticeable amplification

> **Important**: Scale sensitivity varies by model architecture. Dense models (Qwen3-8B, 4096 embd) tolerate scales of 0.15-0.6. MoE models (Qwen3.5-35B-A3B, 2048 embd) need much lower scales (0.01-0.05) before output becomes garbled. Always test on a few prompts before deploying.

---

## Part 7: Interpreting Results

### Reading the Activation Display

If you integrate the activation fetch into your application (see `llama_provider.py` for an example), you can visualize which features fire in real time:

- **Feature index** (`idx`): Dimension in the SAE's feature space (0 to n_features-1)
- **Activation value** (`val`): Magnitude and direction
  - Positive = feature is active (contributing to this behavioral direction)
  - Negative = feature is opposing this direction
  - Higher absolute value = stronger signal
- **Layer depth**: Early layers show syntax, mid layers show behavior, late layers show output shaping

### What Makes a Good Feature?

A feature is interpretable when:
1. **Consistent** — fires on most phrases in a behavioral cluster (consistency > 0.6)
2. **Specific** — fires significantly more on its cluster than on other clusters (specificity > 1.0)
3. **Meaningful** — you can articulate what it represents by reading the phrases that activate it

### Common Pitfalls

- **Generic features**: Fire on everything: these are general language features, not behavioral. The inter-cluster differential scoring handles this.
- **Dead features**: Never activate: the L1 coefficient was too aggressive, or the feature wasn't needed. Usually < 5% is fine.
- **Scale sensitivity**: Start with small scales (0.01-0.05) and increase gradually. MoE models are especially sensitive.
- **Probe timing**: Rapid probing (< 1s between requests) can crash llama-server during collection. Use `--delay 1.0`.

---

## Quick Reference

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/activations` | GET | Query captured activations (supports `?layers=` and `?top_k=` params) |
| `/activations` | POST | Enable/disable capture (`{"enabled": true}`) |
| `/activations/collect` | POST | Start/stop binary collection (`{"layer": 20, "path": "/tmp/out.bin"}`) |

### Pipeline Commands

```bash
# 1. Collect activations
python3 sae_collector.py --layer 20 --rows 500 --input data.jsonl

# 2. Train SAE
python3 sae_trainer.py --input sae_data/activations_layer20_*.bin --epochs 5 --analyze

# 3. Probe individual features
python3 sae_probe.py --model sae_models/sae_layer20_*.pt --text "Your phrase here"

# 4. Batch reprobe all clusters
python3 sae_batch_reprobe.py --model sae_models/sae_layer20_*.pt --cluster-dir clusters/

# 5. Generate feature tags
python3 sae_generate_tags.py --steering sae_steering.json --results *_cluster_results.json

# 6. Extract control vectors
python3 sae_extract_vectors.py --model sae_models/sae_layer20_*.pt --features 12846 --name my_feature

# 7. Apply steering
llama-server -m model.gguf --control-vector-scaled sae_features/my_feature.gguf:-0.3
```

### File Formats

| Format | Extension | Purpose |
|--------|-----------|---------|
| Activation binary | `.bin` | Per-token activation vectors (16-byte header + float32 arrays) |
| SAE model | `.pt` | PyTorch state dict (encoder, decoder, pre_bias weights) |
| SAE metadata | `.json` | Training hyperparameters, stats, input file info |
| Cluster definition | `.json` | Label + list of exemplar phrases |
| Feature tags | `.json` | Feature-to-cluster mappings with confidence scores |
| Steering config | `.json` | Cluster name → steering scale mapping |
| Control vector | `.gguf` | llama.cpp-compatible control vector (one feature direction per layer) |

### Configuring the Server URL

All pipeline scripts default to `http://127.0.0.1:8080`. If your llama-server runs on a different port or host, set the `LLAMA_BASE` environment variable:

```bash
export LLAMA_BASE="http://127.0.0.1:8090"  # or any host:port
python3 sae_pipeline/sae_collector.py --layer 20 --rows 500
```

### Python Dependencies

```
torch>=2.0
numpy
requests
gguf          # pip install gguf (for vector extraction only)
datasets      # pip install datasets (for HuggingFace download only)
```

---

## How It Works (Technical Details)

### The Eval Callback

llama.cpp's `ggml_backend_sched` supports an eval callback: a function pointer called for every tensor node during graph computation. The callback has two phases:

1. **Ask phase** (`ask=true`): Return `true` for tensors you want to observe. The scheduler will arrange a GPU→CPU copy.
2. **Deliver phase** (`ask=false`): The tensor data is now in CPU memory. Read it.

Our callback intercepts `l_out-{layer}` tensors (the output of each transformer layer). For live monitoring, it computes the mean across the token dimension and stores it. For collection, it writes the full per-token vectors to disk.

### Why This Approach?

- **No private headers**: The callback is installed via `common_params.cb_eval` (public API), not by reaching into llama internals. This makes the patch resilient to upstream refactors.
- **Architecture-agnostic**: Any model that emits `l_out` or `final_output` tensors works automatically. Adding support for a new tensor name is a one-line change.
- **Zero overhead when disabled**: The callback returns `false` for all tensors: no GPU-CPU copies occur.
- **Thread-safe**: All mutation of the capture struct is mutex-protected.

### SAE Architecture

```
SparseAutoencoder(
    pre_bias:  Parameter(n_input)               # learned mean subtraction
    encoder:   Linear(n_input → n_features)     # ReLU activation → sparse code
    decoder:   Linear(n_features → n_input)     # no bias, unit-norm columns
)
```

Each column of the decoder matrix is a **feature direction** in activation space. When you extract a feature as a control vector, you're literally extracting that column (normalized).

The encoder learns to detect when a feature is present; the decoder learns what direction in activation space each feature corresponds to. The L1 penalty on the encoder output forces sparsity: the model must explain each activation using as few features as possible.

### Control Vector Application

llama.cpp applies control vectors by adding a scaled direction vector to the layer output:

```
l_out_steered = l_out + scale × direction
```

This happens at every token, at the specified layer(s). Negative scales push the representation away from the feature direction; positive scales pull it toward.

Since our SAE features are trained on the same `l_out` vectors, the feature directions are directly compatible with llama.cpp's control vector mechanism. No adaptation needed.

---

## Extending to Other Architectures

The patch currently adds `build_cvec()` + `cb(cur, "l_out", il)` calls to Qwen3.5 models specifically. For other architectures:

1. Find the model builder: `src/models/{architecture}.cpp`
2. Look for the main layer loop's residual connection (after attention + FFN)
3. Add these two lines after the final residual add:
```cpp
cur = build_cvec(cur, il);
cb(cur, "l_out", il);
```

The activation capture callback will automatically pick up the new `l_out` tensors. No other changes needed.

> **Hybrid architectures** (Mamba+attention): The cvector-generator patch handles these via lazy initialization: it counts actual `l_out` tensors from the first forward pass rather than assuming `n_layers` tensors exist.

---

## Acknowledgments

This work builds on:
- [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/) by Anthropic
- [Representation Engineering](https://arxiv.org/abs/2310.01405) by Zou et al. (control vector theory)

---

## License

The llama.cpp patch follows llama.cpp's MIT license. The SAE pipeline scripts are MIT licensed.
