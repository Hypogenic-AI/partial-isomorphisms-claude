# Code Repositories for Studying Unexpected Partial Isomorphisms

All repositories cloned with `--depth 1` for space efficiency. This document summarizes each repo's purpose, key files, and relevance to the partial isomorphisms research project.

---

## 1. TransformerLens

**Source:** https://github.com/neelnanda-io/TransformerLens
**Local path:** `./TransformerLens/`
**Install:** `pip install transformer_lens`

### Purpose
The core mechanistic interpretability library for GPT-2 style transformers. Created by Neel Nanda and now maintained by the TransformerLensOrg. Supports 50+ pretrained models with a hook-based system for caching and modifying internal activations.

### Key Functionality
- `HookedTransformer`: the central model class; supports `run_with_cache()` to capture all intermediate activations
- `ActivationCache`: dict-like container for all cached activations, indexed by hook name
- `patching.py`: activation patching utilities (path patching, causal tracing)
- `HookedTransformerConfig`: configuration dataclass for model hyperparameters
- `SVDInterpreter.py`: SVD-based weight analysis tool

### Key Scripts / Demos
| File | Description |
|------|-------------|
| `demos/Othello_GPT.ipynb` | TransformerLens analysis of OthelloGPT — directly relevant |
| `demos/Exploratory_Analysis_Demo.ipynb` | Standard mech-interp workflow with attention heads and MLPs |
| `demos/Activation_Patching_in_TL_Demo.ipynb` | Causal tracing via activation patching |
| `demos/Attribution_Patching_Demo.ipynb` | Efficient gradient-based patching attribution |
| `demos/Grokking_Demo.ipynb` | Mechanistic study of grokking phenomenon |
| `Main_Demo.ipynb` | Primary library introduction notebook |

### Relevance to Partial Isomorphisms
TransformerLens is the foundational toolkit for any mechanistic interpretability work in this project. Its hook system enables probing whether two representations are isomorphic by comparing cached activations across layers, models, or training checkpoints. The `Othello_GPT.ipynb` demo is a direct precursor to the partial isomorphisms question: Nanda et al. showed that OthelloGPT learns a *linear* world model, but the degree to which that world model is a full versus partial isomorphism to the true board state is precisely the question this project investigates.

---

## 2. toy-models-of-superposition

**Source:** https://github.com/anthropics/toy-models-of-superposition
**Local path:** `./toy-models-of-superposition/`

### Purpose
Official Anthropic notebook accompanying the "Toy Models of Superposition" paper (Elhage et al., 2022). Provides minimal, reproducible implementations of the superposition phenomenon where neural networks represent more features than they have dimensions.

### Key Files
| File | Description |
|------|-------------|
| `toy_models.ipynb` | Single self-contained notebook reproducing all paper figures |

### Key Concepts Implemented
- **Superposition setup:** ReLU networks trained to compress `n` sparse features into `m << n` dimensions and reconstruct them
- **Phase diagrams:** showing when networks prefer pure vs. superposed representations as a function of feature importance and sparsity
- **Antipodal configurations:** optimal packing of feature directions (pentagons, tetrahedra, etc.) in activation space
- **Privileged vs. non-privileged basis:** how the choice of activation function affects which basis directions are meaningful

### Relevance to Partial Isomorphisms
Superposition is the mechanism by which partial isomorphisms arise. When a model superimposes features, the mapping from concept space to activation space is injective but not isometric — a partial isomorphism in the categorical sense. The toy model gives a controlled testbed where we can measure exactly how "partial" the isomorphism is as a function of feature sparsity, feature importance ratios, and network depth.

---

## 3. representation-engineering (RepE)

**Source:** https://github.com/andyzoujm/representation-engineering
**Local path:** `./representation-engineering/`
**Install:** `pip install -e ./representation-engineering`

### Purpose
Official implementation of "Representation Engineering: A Top-Down Approach to AI Transparency" (Zou et al., 2023). Uses *population-level* representations (contrast vectors derived from PCA over sets of activations) rather than individual neurons or circuits.

### Key Package Structure
```
repe/
  rep_readers.py          # RepReader class: extracts representation directions via PCA
  rep_reading_pipeline.py # HuggingFace pipeline for reading representations
  rep_control_pipeline.py # HuggingFace pipeline for controlling via rep vectors
  rep_control_reading_vec.py  # RepControlReadingVecPipeline
  rep_control_contrast_vec.py # RepControlContrastVecPipeline
  pipelines.py            # Pipeline registration
```

### Key Example Notebooks
```
examples/
  honesty/          # Honesty/truthfulness detection and control
  harmless_harmful/ # Harmlessness representation extraction
  memorization/     # Memorization detection
  primary_emotions/ # Emotion representations
  fairness/         # Fairness-relevant representations
  languages/        # Cross-lingual representation analysis
```

### Quickstart Pattern
```python
from repe import repe_pipeline_registry
repe_pipeline_registry()
rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
rep_control_pipeline = pipeline("rep-control", model=model, tokenizer=tokenizer)
```

### Relevance to Partial Isomorphisms
RepE's contrast vectors are a concrete operationalization of "representation direction." When studying partial isomorphisms, these vectors tell us which directions in activation space encode semantically meaningful distinctions. If two models have parallel contrast vectors for the same concept, that is evidence of an isomorphic substructure. If the vectors are misaligned but the downstream behavior is equivalent, that is evidence of a partial isomorphism — same functional structure, different realization.

---

## 4. mech_int_othelloGPT

**Source:** https://github.com/ajyl/mech_int_othelloGPT
**Local path:** `./mech_int_othelloGPT/`
**Install:** `conda env create -f environment.yml && conda activate mech_int_othello`

### Purpose
Code for "Emergent Linear Representations in World Models of Self-Supervised Sequence Models" (Nanda et al., 2023). Demonstrates that OthelloGPT's internal board representation is linear ("my colour" vs. "opponent's colour"), enabling intervention via simple vector arithmetic.

### Key Scripts
| File | Description |
|------|-------------|
| `mech_int/board_probe.py` | Train and evaluate linear probes on board state |
| `mech_int/train_flipped.py` | Train "Flipped" probes (testing alternate representational schemes) |
| `mech_int/intervene.py` | Causal interventions: edit activations and measure board-state change |
| `mech_int/intervene_blank.py` | Intervention experiments on blank squares |
| `mech_int/intervene_flipped.py` | Intervention with flipped colour scheme |
| `mech_int/tl_othello_utils.py` | Utility functions: board state conversion, hook helpers |
| `constants.py` | Shared constants (board size, token mappings) |
| `mech_int/figures/cosine_exp.ipynb` | Cosine similarity experiments between probe directions |
| `mech_int/figures/movefirst.sync.ipynb` | Move-first analysis notebooks |

### Data Requirements
- OthelloGPT model checkpoint (synthetic model) from Google Drive
- Sequence data generated via `othello_world` repo

### Relevance to Partial Isomorphisms
This is the most directly relevant repository. The central finding — that OthelloGPT's world model is *linear* but represents board state from the perspective of the current player rather than an absolute coordinate system — is itself a partial isomorphism: the model's internal map is isomorphic to the true board state only up to a colour-flip transformation. The probe training and intervention scripts provide the exact methodology needed to detect and characterize such partial isomorphisms in other settings.

---

## 5. SAELens

**Source:** https://github.com/jbloomAus/SAELens
**Local path:** `./SAELens/`
**Install:** `pip install sae-lens` or `poetry install` (dev)

### Purpose
Production-grade library for training, loading, and analysing Sparse Autoencoders (SAEs). SAEs decompose neural network activations into sparse combinations of learned feature directions, providing a potential "dictionary" of the features in superposition.

### Key Package Structure
```
sae_lens/
  saes/
    sae.py                 # Base SAE class with encode()/decode()
    standard_sae.py        # Standard L1-regularized SAE
    topk_sae.py            # TopK SAE (fixed sparsity)
    gated_sae.py           # Gated SAE architecture
    jumprelu_sae.py        # JumpReLU SAE
    batchtopk_sae.py       # BatchTopK SAE
  training/
    sae_trainer.py         # Main training loop
    activations_store.py   # Streams activations from a model during training
    optim.py               # Optimizers and LR schedules
  llm_sae_training_runner.py  # Top-level training entry point
  cache_activations_runner.py # Pre-cache activations to disk for fast training
  loading/                 # Pre-trained SAE loading utilities
  pretrained_saes.yaml     # Registry of all available pre-trained SAEs
```

### Key Scripts
| File | Description |
|------|-------------|
| `scripts/training_a_sparse_autoencoder_othelloGPT.py` | Train SAE specifically on OthelloGPT activations |
| `scripts/sweep-gpt2.py` / `sweep-gpt2-blocks.py` | Hyperparameter sweeps for GPT-2 SAEs |
| `scripts/replication_how_train_saes.py` | Replication script for "How to Train Your SAE" paper |
| `tutorials/basic_loading_and_analysing.ipynb` | Load pre-trained SAE and inspect features |
| `tutorials/training_a_sparse_autoencoder.ipynb` | End-to-end SAE training walkthrough |
| `tutorials/training_saes_on_synthetic_data.ipynb` | Train SAEs on toy superposition models |
| `tutorials/logits_lens_with_features.ipynb` | Interpret SAE features via logit lens |

### Quickstart Pattern
```python
from sae_lens import SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
)
# Extract features
feature_activations = sae.encode(residual_stream_activations)
reconstructed = sae.decode(feature_activations)
```

### Relevance to Partial Isomorphisms
SAEs are the primary tool for moving from "there is structure in activation space" to "here are the specific features encoded." In the context of partial isomorphisms, an SAE trained on two different models (or two different layers of the same model) allows direct comparison of feature dictionaries. If the dictionaries share a common subset of features (up to rotation/permutation), that subspace constitutes the isomorphic part. Features present in one dictionary but absent from the other define the "partial" boundary. The `training_a_sparse_autoencoder_othelloGPT.py` script is a direct starting point for applying SAEs to the OthelloGPT setting studied in this project.

---

## Summary Table

| Repository | Primary Use | Key Entry Point | Partial Isomorphism Role |
|------------|-------------|-----------------|--------------------------|
| `TransformerLens` | Load models, cache activations, hook-based intervention | `HookedTransformer.run_with_cache()` | Activation extraction and comparison infrastructure |
| `toy-models-of-superposition` | Understand superposition mechanics in controlled settings | `toy_models.ipynb` | Minimal testbed where isomorphism degree is analytically tractable |
| `representation-engineering` | Extract and manipulate population-level representation directions | `rep_readers.py` `RepReader` class | Operationalize "representation direction" for isomorphism comparison |
| `mech_int_othelloGPT` | Linear probing and causal intervention on OthelloGPT | `mech_int/board_probe.py`, `intervene.py` | Direct example of a known partial isomorphism (colour-relative board representation) |
| `SAELens` | Train/load sparse autoencoders to decompose activations into features | `SAE.from_pretrained()`, `scripts/training_a_sparse_autoencoder_othelloGPT.py` | Feature-level dictionary comparison to identify isomorphic subspaces |

## Suggested Workflow for Partial Isomorphism Studies

1. **Establish baseline representations** using `TransformerLens` `run_with_cache()` to collect activation snapshots across layers and/or models.

2. **Train probes** (following `mech_int_othelloGPT/mech_int/board_probe.py` methodology) to identify which directions encode which concepts.

3. **Extract contrast vectors** via `representation-engineering`'s `RepReader` to get population-level representation directions for each concept.

4. **Train SAEs** using `SAELens` on each model/layer to get full feature dictionaries; compare dictionaries across conditions.

5. **Characterize the isomorphism** by measuring the overlap between feature dictionaries or probe directions: the shared subspace is the isomorphic part; the remainder is what makes it *partial*.

6. **Intervene** using `mech_int_othelloGPT/mech_int/intervene.py` patterns to confirm causal structure — a true (partial) isomorphism should be recoverable under intervention.

7. **Stress-test with toy models** using `toy-models-of-superposition` by varying sparsity and feature importance to map out the boundary between full and partial isomorphism.
