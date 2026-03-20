# Literature Review: Unexpected Partial Isomorphisms in Neural Networks

## Research Area Overview

This review synthesizes literature on how neural networks internally represent and compress concepts, with focus on the hypothesis that models use **unexpected partial isomorphisms**—counterintuitive mappings between seemingly unrelated concepts—because compressing multiple concepts together enables more efficient reasoning.

The research sits at the intersection of three active areas: (1) **superposition and polysemanticity** in neural representations, (2) **representation similarity and convergence** across models and modalities, and (3) **mechanistic interpretability** of learned algorithms. Together, these literatures provide strong evidence that neural networks routinely create partial structure-preserving maps between concepts that humans would not naturally associate.

---

## Key Papers

### 1. Toy Models of Superposition
- **Authors**: Elhage, Hume, Olsson, Schiefer, et al. (Anthropic)
- **Year**: 2022
- **Source**: arXiv:2209.10652 (Transformer Circuits Thread)
- **Key Contribution**: First systematic study showing that neural networks store more features than they have dimensions by exploiting sparsity, creating geometric structures (regular polytopes) in weight space.
- **Methodology**: Toy ReLU autoencoder x' = ReLU(W^T W x + b) with synthetic sparse data. Features have geometric importance decay and controlled sparsity S.
- **Key Findings**:
  - Superposition emerges when features are sparse and a nonlinearity (ReLU) makes negative interference "free"
  - First-order phase transitions between not-represented / dedicated-dimension / superposition states
  - Features in uniform superposition organize into regular polytopes (digons, triangles, tetrahedra, pentagons, square antiprisms) corresponding to solutions of the Thomson problem
  - "Stickiness" at rational dimensionality fractions (1/2, 2/3, 3/4) resembles the fractional quantum Hall effect
  - Correlated features form local orthogonal bases; anti-correlated features prefer antipodal embedding
  - At high sparsity, code-like structures (binary codes across neurons) emerge
  - Gradient descent spontaneously discovers these geometric structures—they are *unexpected* in the sense that they arise from optimization, not design
- **Datasets Used**: Purely synthetic sparse features
- **Code Available**: Yes (Colab notebooks linked in paper)
- **Relevance**: **Foundational**. The W^T W matrix encodes partial isomorphisms between features. Off-diagonal entries are "hallucinated correlations"—the model treats distinct features as partially equivalent. The polytope geometry shows these mappings have precise mathematical structure.

### 2. The Platonic Representation Hypothesis
- **Authors**: Huh, Cheung, Wang, Isola (MIT)
- **Year**: 2024
- **Source**: arXiv:2405.07987, ICML 2024
- **Key Contribution**: Different neural networks (different architectures, objectives, modalities) converge toward a shared statistical model of reality—the "platonic representation"—characterized by the Pointwise Mutual Information (PMI) kernel.
- **Methodology**: Measures alignment between 78 vision models and multiple LLMs using mutual k-nearest-neighbor (mNN) and CKNNA metrics on paired image-caption datasets.
- **Key Findings**:
  - Better models show higher mutual alignment; "all strong models are alike"
  - Cross-modal (vision-language) alignment scales linearly with model quality
  - LLM alignment to vision models predicts downstream language task performance
  - Alignment is primarily **local** (neighborhood-level), not global—a partial isomorphism
  - The PMI kernel K_PMI(x_a, x_b) = log P(x_a|x_b)/P(x_a) is preserved across modalities for bijective observation functions
  - Color similarity structure recovered independently from pixel co-occurrences, language co-occurrences, and human perception are approximately the same
- **Datasets Used**: VTAB (19 vision tasks), WIT (Wikipedia Image Text), DCI (Densely Captioned Images), CIFAR-10
- **Code Available**: Not explicitly released
- **Relevance**: **Critical**. Provides the theoretical framework for partial isomorphisms: representations from different models/modalities are isomorphic up to constant offsets, but only for shared information about underlying reality. The "partiality" is explained by modality-specific information loss.

### 3. Polysemanticity and Capacity in Neural Networks
- **Authors**: Scherlis, Sachan, Jermyn, Benton, Shlegeris (Redwood Research)
- **Year**: 2022
- **Source**: arXiv:2210.01892
- **Key Contribution**: Formalizes how capacity constraints drive polysemanticity. Defines feature capacity C_i and shows polysemanticity is the globally optimal solution under capacity pressure with sparse inputs.
- **Methodology**: Toy regression model y = Σ v_i x_i² with quadratic activations; analytical solutions for optimal capacity allocation.
- **Key Findings**:
  - Features are polysemantic (0 < C_i < 1) only when multiple features simultaneously hit the same marginal loss threshold—capacity is genuinely contested
  - High kurtosis (k > 3) enables superposition by down-weighting "hallucinated correlations"
  - For k ≤ 3 (dense inputs), superposition never occurs; models jump directly from ignoring to fully representing features
  - Efficient embedding matrices have block-semiorthogonal structure: within-block features interfere, cross-block features are orthogonal
  - The Anthropic model produces many small blocks (regular polytopes); the Redwood model produces one large superposed block
- **Datasets Used**: Purely synthetic
- **Code Available**: No
- **Relevance**: **High**. The "hallucinated correlations" mechanism is precisely the unexpected partial isomorphism: the model's output develops spurious statistical relationships between independent features because entangling them is the globally optimal loss-minimizing strategy.

### 4. A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations
- **Authors**: Chughtai, Chan, Nanda
- **Year**: 2023
- **Source**: arXiv:2302.03025, ICML 2023
- **Key Contribution**: All networks trained on group composition implement the same algorithm family (GCR—Group Composition via Representation theory), but choose different subsets of irreducible representations.
- **Methodology**: One-hidden-layer ReLU MLPs and Transformers trained on finite group multiplication tables (cyclic, dihedral, symmetric, alternating groups).
- **Key Findings**:
  - Networks embed group elements as representation matrices ρ(a) from mathematical representation theory
  - The ReLU MLP layer multiplies these matrices; unembedding computes characters χ_ρ
  - **Weak universality** holds: all networks share the GCR algorithmic skeleton but differ in which representation-theoretic subspaces are activated
  - Different seeds learn different subsets of irreducible representations—each is a **partial isomorphism** preserving structure along certain representation-theoretic directions
  - Training dynamics split into memorization, circuit formation, and cleanup (grokking) phases
- **Datasets Used**: Synthetic group Cayley tables (all (a,b) pairs for groups up to |S6| = 720)
- **Code Available**: Yes (referenced in paper)
- **Relevance**: **High**. Directly demonstrates partial isomorphisms: the network implements a partial group homomorphism (preserving structure only along selected irreducible representations). Different models select different partial views, creating a family of partial isomorphisms to the same algebraic structure.

### 5. Emergent Linear Representations in World Models of Self-Supervised Sequence Models
- **Authors**: Nanda, Lee, Wattenberg
- **Year**: 2023
- **Source**: arXiv:2309.00941
- **Key Contribution**: OthelloGPT encodes its world model linearly using a MINE/YOURS/EMPTY frame (relative to current player), not the absolute BLACK/WHITE/EMPTY that humans would use.
- **Methodology**: Linear and nonlinear probes on OthelloGPT activations; causal interventions via vector addition.
- **Key Findings**:
  - Linear probes with MINE/YOURS/EMPTY achieve 99.5% accuracy vs. 74.4% for BLACK/WHITE/EMPTY
  - The model's relational encoding is an **unexpected partial isomorphism**: it conflates "my pieces" regardless of color, which is functionally correct but counterintuitive
  - Causal interventions confirm these directions are not correlational artifacts
  - Multiple circuits exist for the same output—different computational paths appear isomorphic at the output level
- **Datasets Used**: Synthetic Othello game sequences (3.5M training games)
- **Code Available**: Yes (github.com/ajyl/mech_int_othelloGPT)
- **Relevance**: **Very High**. A concrete example of an unexpected partial isomorphism: the model maps BLACK→MINE and WHITE→MINE depending on context, treating two distinct human categories as one internal concept. This is precisely the kind of counterintuitive mapping the research hypothesis predicts.

### 6. Representation Engineering: A Top-Down Approach to AI Transparency
- **Authors**: Zou, Phan, Chen, et al. (CAIS, CMU, Berkeley, Stanford)
- **Year**: 2023
- **Source**: arXiv:2310.01405
- **Key Contribution**: Safety-relevant concepts (honesty, morality, power-seeking) emerge as linearly-separable directions in LLM activation spaces that can be read and written like dials.
- **Methodology**: Linear Artificial Tomography (LAT): PCA on contrastive stimulus pairs to extract concept directions.
- **Key Findings**:
  - A single reading vector extracted from LLaMA-2 outperforms few-shot prompting on multiple QA benchmarks
  - Concept directions generalize out-of-distribution to novel scenarios
  - Directions are composable and can be used for control (steering model behavior)
- **Datasets Used**: TruthfulQA, RACE, CommonsenseQA, ARC
- **Code Available**: Yes (github.com/andyzoujm/representation-engineering)
- **Relevance**: **Moderate**. Demonstrates that high-level concepts have linear structure in LLMs, which is a prerequisite for studying partial isomorphisms between concept directions.

### 7. Finding Neurons in a Haystack: Case Studies with Sparse Probing
- **Authors**: Gurnee, Nanda, Pauly, Harvey, Troitskii, Bertsimas
- **Year**: 2023
- **Source**: arXiv:2305.01610
- **Key Contribution**: Systematic study using k-sparse probes showing that early layers use superposition (polysemantic neurons), middle layers have dedicated neurons for contextual features, and sparsity increases with scale.
- **Methodology**: k-sparse linear classifiers on MLP activations of Pythia models (70M–6.9B params), probing for 100+ features across 10 categories.
- **Key Findings**:
  - First ~25% of MLP layers employ substantially more superposition than the rest
  - Higher-level contextual features (e.g., is_python_code) get monosemantic neurons in middle layers
  - Scale increases sparsity on average but with multiple dynamics: some features emerge, some split, some remain unchanged
  - Polysemantic neurons activate for collections of seemingly unrelated n-grams
- **Datasets Used**: The Pile (for activations), custom feature labels
- **Code Available**: Referenced but not explicitly linked
- **Relevance**: **High**. Provides empirical evidence for the spectrum of superposition in real LLMs—exactly the regime where unexpected partial isomorphisms operate.

### 8. Git Re-Basin: Merging Models modulo Permutation Symmetries
- **Authors**: Ainsworth, Hayase, Srinivasa
- **Year**: 2022
- **Source**: arXiv:2209.14764
- **Key Contribution**: Different trained models converge to the same loss basin up to permutation of neurons, enabling model merging.
- **Relevance**: **Moderate**. Shows that the permutation symmetry group acts on the space of learned representations, creating equivalence classes of solutions. This is a structural isomorphism at the weight level.

### 9. Relative Representations Enable Zero-Shot Latent Space Communication
- **Authors**: Moschella, Maiorca, Fumero, Norelli, Locatello, Rodolà
- **Year**: 2022
- **Source**: arXiv:2209.10535
- **Key Contribution**: Different models trained independently on different data can communicate through their latent spaces using relative representations (cosine similarities to anchor points).
- **Relevance**: **Moderate**. Demonstrates that the similarity structure (kernel) is shared across models—a partial isomorphism of the metric space, not just the linear space.

---

## Common Methodologies

### Toy Model Analysis
Used in: Elhage et al., Scherlis et al., Chughtai et al.
- Small networks with controllable architecture on synthetic data
- Allows ground-truth feature knowledge and analytical solutions
- Primary tool for understanding *why* superposition/partial isomorphisms emerge

### Linear Probing
Used in: Nanda et al., Gurnee et al., Zou et al.
- Train linear classifiers on internal activations to detect concept representations
- k-sparse variants localize features to specific neurons
- PCA on contrastive pairs extracts concept directions

### Representation Similarity Analysis
Used in: Huh et al., Moschella et al.
- Measure alignment between model representation spaces
- Metrics: CKA, mutual k-NN, CKNNA, Procrustes distance
- Can be applied cross-model and cross-modality

### Causal Intervention
Used in: Nanda et al., Zou et al.
- Add/subtract concept direction vectors from activations
- Verify that detected directions causally affect model behavior
- Distinguishes correlational from functional representations

---

## Standard Baselines

- **Random/untrained model representations**: Lower bound for representation alignment
- **PCA/linear dimensionality reduction**: Shows how much structure is linearly accessible
- **Full (non-sparse) probes vs. sparse probes**: Tests whether features are in superposition or dedicated dimensions
- **Monosemantic networks (with dictionary learning/SAEs)**: Attempts to extract individual features from superposition

---

## Evaluation Metrics

- **Feature capacity** C_i: Fraction of a dimension allocated to feature i (Scherlis et al.)
- **Probe accuracy at sparsity k**: How many neurons needed to detect a feature (Gurnee et al.)
- **Mutual k-NN alignment**: Shared local neighborhood structure between representations (Huh et al.)
- **CKA (Centered Kernel Alignment)**: Global representation similarity
- **Causal intervention accuracy**: Whether adding/subtracting directions changes behavior correctly
- **Reconstruction MSE**: For autoencoder-based superposition studies

---

## Datasets in the Literature

- **Synthetic sparse features**: Used in Elhage et al., Scherlis et al. — controllable sparsity, importance, correlations
- **Finite group Cayley tables**: Used in Chughtai et al. — cyclic, dihedral, symmetric groups
- **Othello game sequences**: Used in Nanda et al. — synthetic board game data for studying world models
- **The Pile / OpenWebText**: Used for LLM activation extraction in probing studies
- **VTAB / WIT / CIFAR-10**: Used in Huh et al. for representation alignment measurement
- **TruthfulQA / CommonsenseQA / ARC**: Used in Zou et al. for evaluating concept directions

---

## Gaps and Opportunities

1. **No systematic study of *which* partial isomorphisms models create**: We know models compress features together, but there's no taxonomy of *which* unexpected concept pairings arise and whether they follow predictable patterns.

2. **Limited connection between superposition geometry and model behavior**: The polytope structures found by Elhage et al. are beautiful mathematics, but their functional consequences for downstream tasks remain unclear.

3. **Scaling laws for partial isomorphisms**: How do the specific concept mappings change with model scale? Gurnee et al. show sparsity increases, but the *structure* of the remaining superposition is unstudied.

4. **Cross-task transfer of superposed representations**: If model A superimposes concepts X and Y (creating a partial isomorphism), does this help or hurt when the model encounters tasks requiring *both* X and Y?

5. **Quantifying "unexpectedness"**: No formal metric exists for how surprising a particular concept pairing is. A framework combining human concept similarity judgments with model-internal similarity could fill this gap.

6. **Bridge between toy models and real LLMs**: Most theoretical understanding comes from toy models; validating these insights in full-scale models remains challenging.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **Synthetic sparse feature datasets** (primary): Following Elhage et al., generate features with controlled sparsity, importance, and correlation structure. This allows ground-truth analysis of which partial isomorphisms the model creates.
2. **Finite group operation datasets**: Following Chughtai et al., study which representation-theoretic partial isomorphisms emerge across different groups and architectures.
3. **Othello game sequences**: Use OthelloGPT as a testbed for studying unexpected encodings (MINE/YOURS vs. BLACK/WHITE) in a model with known ground truth.

### Recommended Baselines
1. **Toy autoencoder** (Elhage et al. architecture): ReLU(W^T W x + b) with varying feature-to-dimension ratios
2. **Sparse autoencoders (SAEs)**: Train on real LLM activations to extract features and study which concepts get superimposed
3. **Linear probes at varying sparsity k**: Measure the spectrum of superposition in real models

### Recommended Metrics
1. **Feature capacity C_i and off-diagonal interference** (W_i · W_j)²: Direct measure of partial isomorphism strength
2. **Cosine similarity between feature directions**: Which concept pairs share representational space?
3. **Partial isomorphism score**: Procrustes-aligned similarity restricted to concept subsets
4. **Human-judged concept similarity vs. model-internal similarity**: Measures "unexpectedness" of concept pairings

### Methodological Considerations
- Start with toy models where ground truth is known before scaling to real models
- Use causal interventions to verify that detected partial isomorphisms are functional, not artifacts
- Compare partial isomorphisms across multiple random seeds to distinguish systematic from accidental pairings
- Control for sparsity: many apparent concept pairings may simply reflect co-occurrence in training data
