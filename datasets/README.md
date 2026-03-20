# Datasets for "Unexpected Partial Isomorphisms in Neural Networks"

This directory contains generation scripts, generated datasets, and access guides
for studying the hypothesis that ML models internally use counterintuitive
mappings between concepts for efficient compression.

---

## Directory Structure

```
datasets/
├── .gitignore                         # excludes large .npy/.npz files
├── README.md                          # this file
│
├── synthetic/                         # Toy Models of Superposition data
│   ├── generate_sparse_features.py    # generator script
│   ├── tms_tiny.npz                   # 5 features, 100k samples
│   ├── tms_medium.npz                 # 80 features, 500k samples
│   ├── tms_large_uniform.npz          # 256 features, 1M samples
│   ├── correlated_groups.npz          # 10 groups × 10 correlated features
│   ├── sparsity_sweep_s0p*.npz        # 6 sparsity levels (0.5 → 0.999)
│   └── *_meta.json                    # metadata for each dataset
│
├── group_ops/                         # Group composition datasets
│   ├── generate_group_ops.py          # generator script
│   ├── cyclic_113_triples.npy         # all (a, b, a*b) for Z/113Z
│   ├── cyclic_113_table.npy           # Cayley table
│   ├── symmetric_4_triples.npy        # S4, order 24
│   └── *_meta.json                    # metadata
│
├── othello/                           # OthelloGPT game sequences
│   ├── generate_othello.py            # generator / downloader
│   ├── games.npy                      # (n_games, max_len) move sequences
│   ├── game_lengths.npy               # actual length of each game
│   └── meta.json
│
├── probing/                           # Pythia probing setup
│   └── pythia_probing_guide.py        # utilities + documentation
│
└── benchmarks/                        # Representation similarity metrics
    └── representation_similarity.py   # CKA, RSA, Procrustes, kNN, PI-score
```

---

## 1. Synthetic Sparse Feature Datasets

**Research connection:** Elhage et al. (2022), "Toy Models of Superposition"
https://transformer-circuits.pub/2022/toy_model/index.html

The data-generating process: each feature $i$ of point $x$ is independently
sampled from Uniform(0,1] with probability `(1 - sparsity)`, and set to 0
otherwise.  This mirrors the sparse, independent feature structure assumed
in the superposition hypothesis.

### Available Presets

| Name | Features | Samples | Sparsity | Notes |
|---|---|---|---|---|
| `tms_tiny` | 5 | 100k | 0.99 | Exact match to TMS Figure 1 running example |
| `tms_medium` | 80 | 500k | 0.99 | Geometric importance decay 0.9 |
| `tms_large_uniform` | 256 | 1M | 0.995 | Uniform importance (no hierarchy) |
| `correlated_groups` | 100 | 500k | 0.99 | 10 groups of 10, within-group corr=0.8 |
| `sparsity_sweep_s*` | 40 | 200k | 0.5–0.999 | Phase transition sweep |

### Usage

```python
import numpy as np
data = np.load("synthetic/tms_tiny.npz")
X = data["X"]          # (100000, 5) float32
# For correlated_groups dataset:
data = np.load("synthetic/correlated_groups.npz")
X, group_labels = data["X"], data["group_labels"]
```

### Re-generating

```bash
# Generate all presets (takes ~40s)
python3 synthetic/generate_sparse_features.py

# Single preset
python3 synthetic/generate_sparse_features.py --preset tms_tiny

# Custom one-off
python3 synthetic/generate_sparse_features.py \
    --n_samples 200000 --n_features 512 --sparsity 0.99
```

---

## 2. Group Operation Datasets

**Research connection:** Chughtai et al. (2023), "A Toy Model of Universality"
https://arxiv.org/abs/2302.03025  |  Nanda et al. (2023), "Progress Measures
for Grokking" https://arxiv.org/abs/2301.05217

Each dataset is the complete set of n² triples (a, b, a*b) for a finite group G
of order n.  Networks trained on this task must learn the group table, and
mechanistic analysis reveals Fourier-based internal representations.

### Available Groups

| Name | Order | Triples | Notes |
|---|---|---|---|
| `cyclic_113` | 113 | 12,769 | Exact group used in Nanda et al. grokking paper |
| `cyclic_97` | 97 | 9,409 | Prime-order cyclic group |
| `cyclic_59` | 59 | 3,481 | |
| `symmetric_4` | 24 | 576 | S4, non-abelian |
| `symmetric_3` | 6 | 36 | S3, smallest non-abelian |
| `dihedral_10` | 20 | 400 | D10 |
| `dihedral_6` | 12 | 144 | D6 |
| `klein4` | 4 | 16 | Z/2Z × Z/2Z |
| `quaternion` | 8 | 64 | Q8 |

### Usage

```python
from group_ops.generate_group_ops import load_group_dataset, make_train_test_split

triples, table, meta = load_group_dataset("group_ops/", "cyclic_113")
# triples: (12769, 3) int32 — each row is (a_idx, b_idx, c_idx)
# table:   (113, 113) int32 — Cayley table

train, test = make_train_test_split(triples, test_fraction=0.2)
```

### Re-generating

```bash
# Generate all default groups
python3 group_ops/generate_group_ops.py --all

# Single group
python3 group_ops/generate_group_ops.py --group cyclic_113

# Sweep all cyclic groups up to order 200
python3 group_ops/generate_group_ops.py --all_cyclic --max_order 200
```

---

## 3. Othello Game Sequences (OthelloGPT)

**Research connection:** Li et al. (2022), "Emergent World Representations"
https://arxiv.org/abs/2210.13382  |  Nanda et al. (2023) probing analysis

Game sequences are encoded as integers 0–63 (row-major cell index on the 8×8
board).  The full research dataset contains ~130,000 games from synthetic
self-play; the local copy contains 5,000 synthetic games.

### Download official dataset (recommended)

```bash
pip install datasets
python3 othello/generate_othello.py --out_dir othello/
```

This attempts to download `roamresearch/OthelloGPT` from HuggingFace, falling
back to 50,000 locally generated random games if unavailable.

### Usage

```python
from othello.generate_othello import load_games, get_board_states

games, lengths, meta = load_games("othello/")
# games:   (n_games, max_len) int8, -1 = padding
# lengths: (n_games,) int16

# Get board state sequence for game 0
moves = games[0, :lengths[0]].tolist()
states = get_board_states(moves)
# states: (len(moves)+1, 64) int8, values in {-1=white, 0=empty, 1=black}
```

---

## 4. Pythia Sparse Probing Experiments

**Research connection:** Gurnee et al. (2023), "Finding Neurons in a Haystack"
https://arxiv.org/abs/2305.01610

### Installation

```bash
python3 -m pip install transformers torch datasets spacy accelerate
python3 -m spacy download en_core_web_sm
```

### Model access

```python
from probing.pythia_probing_guide import load_pythia, extract_activations

# Load Pythia-70M (smallest, ~70M params)
model, tokenizer = load_pythia("70m")

# Load a specific training checkpoint (step 50000 of 143000)
model, tokenizer = load_pythia("70m", checkpoint_step=50000)
```

Available sizes: `70m`, `160m`, `410m`, `1b`, `1.4b`, `2.8b`, `6.9b`, `12b`
(and `-d` deduped variants).

### Extracting activations

```python
texts = ["The cat sat on the mat.", "Paris is the capital of France."]
acts = extract_activations(texts, model, tokenizer, layer_indices=[0, 6, 12])
# acts: dict {layer_idx -> (n_texts, seq_len, hidden_dim)}
```

### Feature labelling (for probing)

```python
from probing.pythia_probing_guide import label_tokens_with_spacy, PROBING_FEATURES

labels = label_tokens_with_spacy(texts)
# labels: dict {"part_of_speech/NOUN" -> [bool_array_per_doc, ...], ...}
# Total: 60 features across 5 categories
```

### The Pile (training corpus)

```python
from probing.pythia_probing_guide import load_pile_sample

docs = load_pile_sample(n_docs=10_000)  # streams from HuggingFace
```

---

## 5. Representation Similarity Benchmarks

The `benchmarks/representation_similarity.py` module provides standard and
custom metrics for quantifying cross-model representation alignment.

### Metrics

| Function | Method | Rotation-invariant |
|---|---|---|
| `cka_linear(X, Y)` | Linear CKA (Kornblith 2019) | Yes |
| `cka_rbf(X, Y)` | RBF-kernel CKA | Yes |
| `procrustes_similarity(X, Y)` | Orthogonal Procrustes | Yes |
| `rsa_correlation(X, Y)` | Spearman on RDMs | Yes |
| `mutual_knn_overlap(X, Y, k)` | k-NN neighbourhood overlap | Yes |
| `partial_isomorphism_score(...)` | Custom per-feature alignment | Partial |

### Usage

```python
from benchmarks.representation_similarity import compare_all_metrics, partial_isomorphism_score
import numpy as np

X = np.load(...)  # (n_samples, d1) activations from model A
Y = np.load(...)  # (n_samples, d2) activations from model B

results = compare_all_metrics(X, Y)
# {'cka_linear': 0.87, 'rsa_cosine': 0.73, 'procrustes_similarity': 0.91, ...}

# Detect unexpected cross-feature alignments
score = partial_isomorphism_score(X, Y, feature_labels_X, feature_labels_Y)
# score['aligned_fraction'] = 0.6  (40% of feature classes map unexpectedly)
# score['unexpected_pairs'] = [(label_A, label_B, strength), ...]
```

---

## Generation Summary

| Script | Runtime | Output size |
|---|---|---|
| `synthetic/generate_sparse_features.py` | ~40s | ~50 MB (large datasets gitignored) |
| `group_ops/generate_group_ops.py --all` | <1s | ~0.5 MB |
| `othello/generate_othello.py --n_games 5000` | ~5s | ~0.4 MB |

Large `.npy` and `.npz` files are excluded from git via `.gitignore`.
Regenerate locally with the scripts above.

---

## Key Papers

1. Elhage et al. (2022). **Toy Models of Superposition.**
   https://transformer-circuits.pub/2022/toy_model/index.html

2. Nanda et al. (2023). **Progress Measures for Grokking via Mechanistic
   Interpretability.** https://arxiv.org/abs/2301.05217

3. Chughtai et al. (2023). **A Toy Model of Universality: Reverse Engineering
   How Networks Learn Group Operations.** https://arxiv.org/abs/2302.03025

4. Li et al. (2022). **Emergent World Representations: Exploring a Sequence
   Model Trained on a Synthetic Task.** https://arxiv.org/abs/2210.13382

5. Gurnee et al. (2023). **Finding Neurons in a Haystack: Case Studies with
   Sparse Probing.** https://arxiv.org/abs/2305.01610

6. Kornblith et al. (2019). **Similarity of Neural Network Representations
   Revisited.** https://arxiv.org/abs/1905.00414
