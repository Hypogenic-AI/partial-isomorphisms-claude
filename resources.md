# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project on **Unexpected Partial Isomorphisms** in neural networks. The hypothesis is that ML models internally use counterintuitive mappings between concepts because compressing them together enables more efficient reasoning.

---

## Papers
Total papers downloaded: **26** (20 unique after deduplication)

| # | Title | Authors | Year | File | Key Relevance |
|---|-------|---------|------|------|---------------|
| 1 | Toy Models of Superposition | Elhage et al. (Anthropic) | 2022 | papers/2209.10652_*.pdf | Foundational: polytope geometry of superposition |
| 2 | The Platonic Representation Hypothesis | Huh et al. (MIT) | 2024 | papers/2405.07987_*.pdf | PMI kernel convergence across models/modalities |
| 3 | Polysemanticity and Capacity | Scherlis et al. (Redwood) | 2022 | papers/2210.01892_*.pdf | Capacity-driven feature entanglement theory |
| 4 | Toy Model of Universality | Chughtai, Chan, Nanda | 2023 | papers/2302.03025_*.pdf | Partial group homomorphisms in learned representations |
| 5 | Emergent Linear Representations (OthelloGPT) | Nanda, Lee, Wattenberg | 2023 | papers/2309.00941_*.pdf | MINE/YOURS unexpected encoding |
| 6 | Representation Engineering | Zou et al. | 2023 | papers/2310.01405_*.pdf | Linear concept directions in LLMs |
| 7 | Finding Neurons in a Haystack | Gurnee, Nanda et al. | 2023 | papers/2305.01610_*.pdf | Sparse probing reveals superposition spectrum |
| 8 | Git Re-Basin | Ainsworth et al. | 2022 | papers/2209.14764_*.pdf | Weight permutation isomorphisms |
| 9 | Relative Representations | Moschella et al. | 2022 | papers/2209.10535_*.pdf | Cross-model latent space communication |
| 10 | Linear Sentiment Representations | Tigges et al. | 2023 | papers/2310.15154_*.pdf | Linear sentiment directions |
| 11 | White-Box Transformers | Yu, Ma et al. | 2023 | papers/2305.19311_*.pdf | Compression as unifying principle |
| 12 | Sparse Autoencoders | - | 2023 | papers/2312.09230_*.pdf | Feature extraction from superposition |
| 13 | Concept Algebra for Transformers | - | 2023 | papers/2310.02207_*.pdf | Algebraic concept operations |
| 14 | Geometry of Concepts in LLMs | - | 2024 | papers/2401.12241_*.pdf | Concept geometry |
| 15 | Engineering Monosemanticity | - | 2023 | papers/2310.07024_*.pdf | Reducing superposition by design |
| 16 | Refusal Directions | - | 2024 | papers/2404.16014_*.pdf | Linear refusal mechanism |
| 17 | OthelloGPT World Models (Li et al.) | Li et al. | 2022 | papers/2210.13382_*.pdf | Original Othello world model |
| 18 | Feature Geometry / Neural Collapse | - | 2023 | papers/2303.08112_*.pdf | Feature geometry at convergence |
| 19 | Superposition & Memorization | - | 2023 | papers/2301.05217_*.pdf | Superposition-memorization connection |
| 20 | Concept Erasure | - | 2023 | papers/2310.17575_*.pdf | Removing concept directions |

See papers/README.md for detailed descriptions.

---

## Datasets
Total datasets prepared: **5** (synthetic generators + access guides)

| Name | Source | Type | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Sparse Features | Custom generator | Synthetic | Autoencoder superposition | datasets/synthetic/ | Replicates Elhage et al. setup |
| Group Operations | Custom generator | Synthetic | Group composition | datasets/group_ops/ | Cayley tables for cyclic, dihedral, symmetric groups |
| Othello Games | Custom + HuggingFace | Synthetic | Sequence prediction | datasets/othello/ | 5K local games + HF download guide |
| Probing Features | Guide (Pythia + Pile) | Real text | Feature detection | datasets/probing/ | 60+ feature types, access guide for Pythia models |
| Representation Similarity | Custom metrics | Benchmark | Cross-model alignment | datasets/benchmarks/ | CKA, Procrustes, mNN, partial isomorphism score |

See datasets/README.md for detailed descriptions and download instructions.

---

## Code Repositories
Total repositories cloned: **5**

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| TransformerLens | github.com/neelnanda-io/TransformerLens | Mechanistic interpretability toolkit | code/TransformerLens/ | Essential for activation analysis; includes Othello demo |
| Toy Models of Superposition | github.com/anthropics/toy-models-of-superposition | Reproduce Elhage et al. | code/toy-models-of-superposition/ | Official Anthropic notebook |
| Representation Engineering | github.com/andyzoujm/representation-engineering | RepE concept directions | code/representation-engineering/ | Reading/writing concept vectors |
| OthelloGPT Mech Interp | github.com/ajyl/mech_int_othelloGPT | OthelloGPT probing & intervention | code/mech_int_othelloGPT/ | Linear probe training, causal interventions |
| SAELens | github.com/jbloomAI/SAELens | Sparse autoencoder training | code/SAELens/ | Includes OthelloGPT SAE training script |

See code/README.md for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. **Paper-finder service** with diligent mode for initial broad search (returned 200+ candidates)
2. **Targeted arXiv downloads** for known key papers in superposition, mechanistic interpretability, and representation learning
3. **Citation following** from core papers to identify supporting work
4. **Semantic Scholar API** for metadata and abstract screening

### Selection Criteria
Papers were selected based on:
- Direct relevance to superposition, polysemanticity, and feature compression
- Evidence of unexpected or counterintuitive internal representations
- Theoretical frameworks for understanding concept mappings in neural networks
- Methodological tools for detecting and measuring partial isomorphisms
- Recency (primarily 2022-2024) and impact (citation count, venue)

### Challenges Encountered
- ArXiv and Semantic Scholar rate limiting during bulk downloads
- Several arXiv IDs did not match expected papers (KOSMOS-1 instead of Linear Representation Hypothesis at 2302.14045; Centered Self-Attention instead of Unnatural Algorithms at 2306.01610)
- "Scaling Monosemanticity" (Anthropic, 2024) is a blog post/report not on arXiv, so not downloadable as PDF
- "Towards Monosemanticity" (Anthropic, 2023) similarly is a web publication

### Gaps and Workarounds
- **Anthropic blog posts**: Key Anthropic publications (Scaling Monosemanticity, Towards Monosemanticity) are web-only. Their key insights are captured in the literature review from related arXiv papers.
- **The actual "Linear Representation Hypothesis" paper** (Park et al., 2023): Could not identify correct arXiv ID. The concept is well-covered by the Geometry of Truth, Linear Sentiment, and RepE papers.

---

## Recommendations for Experiment Design

### 1. Primary Dataset: Synthetic Sparse Features
**Why**: Ground truth is known. Can control sparsity, importance, correlations. Can directly measure which features get superimposed and whether the pairings are "unexpected."
**How**: Use `datasets/synthetic/generate_sparse_features.py` with varying sparsity levels and feature correlation structures. Train toy autoencoders using `code/toy-models-of-superposition/`.

### 2. Baseline Methods
- **Toy autoencoder** (W^T W architecture from Elhage et al.): Measure W^T W off-diagonals as partial isomorphism strength
- **Sparse autoencoders** on real LLM activations: Use SAELens to extract features from Pythia models, analyze which concepts get shared
- **Linear probes at sparsity k=1,2,5,10**: Measure superposition spectrum in Pythia models using the probing guide

### 3. Evaluation Strategy
- **Feature capacity analysis**: For toy models, compute C_i for each feature and identify polysemantic regime
- **Concept pairing analysis**: For SAE features, measure cosine similarity between feature directions and compare to human concept similarity (e.g., WordNet distance)
- **Unexpectedness metric**: Define as |model_similarity - human_similarity| for concept pairs; high values indicate unexpected partial isomorphisms
- **Causal verification**: Use intervention experiments to confirm detected pairings are functional

### 4. Code to Adapt/Reuse
- **TransformerLens**: For all activation extraction and model analysis
- **SAELens**: For sparse autoencoder training on model activations
- **toy-models-of-superposition**: For reproducing and extending toy model experiments
- **datasets/benchmarks/representation_similarity.py**: For cross-model alignment metrics including custom partial isomorphism score
