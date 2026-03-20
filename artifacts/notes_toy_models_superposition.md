# Detailed Notes: "Toy Models of Superposition"
**Citation:** Elhage et al., Transformer Circuits Thread, Sept 14, 2022
**arXiv:** 2209.10652
**Authors:** Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, Christopher Olah
**Affiliations:** Anthropic, Harvard
**URL:** https://transformer-circuits.pub/2022/toy_model/index.html

---

## 1. Main Hypothesis and Research Question

### Central Question
Why do neural network neurons sometimes correspond cleanly to interpretable features ("monosemantic neurons") but often respond to many unrelated inputs ("polysemantic neurons")?

### The Superposition Hypothesis
The paper's core claim is that neural networks "want to represent more features than they have neurons," so they exploit properties of high-dimensional spaces to pack more features than dimensions by tolerating controlled interference. This is called **superposition**.

Formally: a linear representation exhibits superposition if the matrix W^T W is not invertible — i.e., there are more feature directions than embedding dimensions.

### Central Metaphor
A small neural network can noisily "simulate" a much larger, sparse, disentangled network. The observed polysemantic neuron is a compressed projection of what would be a monosemantic neuron in the imagined larger model.

### Two Countervailing Forces
The paper frames the key tension as:
- **Privileged Basis**: Some architectures (those with activation functions like ReLU on hidden layers) create a privileged basis that encourages features to align with individual neurons (monosemanticity).
- **Superposition**: The pressure to represent more features than there are neurons pushes features away from neuron-alignment, causing polysemanticity.

---

## 2. Key Definitions

### Feature
Three candidate definitions offered:
1. Features as arbitrary functions of input (rejected as too broad)
2. Features as interpretable/human-understandable properties (rejected as too narrow — features need not be human-interpretable)
3. **Preferred definition:** "Properties of the input which a sufficiently large neural network will reliably dedicate a neuron to representing." Curve detectors appear reliably across many vision models and thus constitute a feature.

### Linear Representation Hypothesis
Two sub-properties:
- **Decomposability:** Network activations can be decomposed into features whose meanings are independent.
- **Linearity:** Each feature f_i is represented as a direction W_i in activation space. Multiple features activating with values x_f1, x_f2, ... are represented as x_f1 * W_f1 + x_f2 * W_f2 + ...

The paper argues linearity is natural because: (a) neural networks are predominantly linear computation; (b) linear features are the natural output of weight-template pattern matching; (c) linear features are "linearly accessible" to downstream layers; (d) they permit non-local generalization.

### Superposition (formal)
A linear representation **exhibits superposition** if W^T W is not invertible. Equivalently: there are more feature directions than embedding dimensions, so features are not mutually orthogonal.

### Polysemanticity
A neuron is polysemantic if it responds to multiple unrelated features. Under the superposition hypothesis, polysemanticity is an *inevitable consequence* of superposition: if more features are packed into fewer dimensions, every basis direction (neuron) will project onto multiple feature directions.

### Monosemantic Neuron
A neuron dedicated to a single feature. Occurs when a feature has a privileged basis and has not been pushed into superposition.

### Privileged Basis
A representation has a privileged basis when something about the architecture (e.g., a ReLU activation function) makes certain directions (basis dimensions = neurons) special. Non-privileged basis examples: word embeddings, transformer residual stream. Privileged basis examples: conv net neurons, transformer MLP layer activations.

### Feature Dimensionality
The "fraction of a dimension" allocated to a feature i:

  D_i = ||W_i||^2 / sum_j (W_hat_i · W_j)^2

where W_hat_i is the unit vector of W_i. Numerator measures how much the feature is represented; denominator measures how many other features share that dimension. An antipodal pair has D = 1/2; a dedicated dimension has D = 1; an unrepresented feature has D = 0.

---

## 3. Experimental Setup

### Toy Model Architecture (Primary: "ReLU Output Model")
- Input x ∈ R^n (feature vector, synthetic)
- Hidden layer h = Wx, where W is n×m (compression to m dims)
- Output x' = ReLU(W^T h + b) = ReLU(W^T Wx + b)
- W^T is used as the decoder (tied weights), justified by: avoids ambiguity about feature directions; empirically works; approximately principled via compressed sensing
- Bias b is added; negative bias is crucial for filtering interference noise

### Linear Model Baseline
- x' = W^T Wx + b (no ReLU)
- Always performs PCA; never exhibits superposition
- Represents top-m most important features as orthogonal basis

### Computation Model ("ReLU Hidden Layer Model")
For studying computation in superposition:
- h = ReLU(W_1 x) (hidden layer with ReLU — creates privileged basis)
- y' = ReLU(W_2 h + b)
- W_1 and W_2 are learned independently
- Task: compute y = abs(x)

### Synthetic Feature Data
Each input dimension x_i is a "feature" with:
- **Sparsity S**: x_i = 0 with probability S; otherwise uniformly sampled from [0, 1]
- **Importance I_i**: scalar multiplier on squared error loss for feature i
- Typical experiments: importance I_i = 0.7^i or I_i = 0.8^i (geometric decay)
- Sparsity is uniform across features (S_i = S) in most experiments

Three key assumptions about real features being modeled:
1. **Feature Sparsity**: most features are inactive most of the time (sparse)
2. **More features than neurons**: far more potential features than model has neurons
3. **Features vary in importance**: not all features equally useful to a task

### Loss Function
Weighted MSE: L = sum_x sum_i I_i * (x_i - x'_i)^2

### Key Experimental Parameters
- Main basic experiments: n = 20 features, m = 5 hidden dimensions, I_i = 0.7^i
- Uniform superposition geometry: n = 400, m = 30, I_i = 1
- Computation experiments: n = 100 features, m = 40 neurons, I_i = 0.8^i
- Phase diagram: n = 2, m = 1 (analytically tractable)
- Privileged basis: n = 10, m = 5, I_i = 0.75^i (1000 runs per sparsity level, take lowest loss)

---

## 4. Key Findings About When/Why Superposition Occurs

### Condition 1: Feature Sparsity is Necessary
- With dense features (1 - S = 1.0), the ReLU output model behaves like a linear model: represents top-m features orthogonally, ignores the rest.
- As sparsity increases, superposition emerges: the model begins representing more features non-orthogonally.
- Intuition: interference between two features only causes loss when both are simultaneously active. With sparse features, simultaneous co-activation is rare, so interference cost is low relative to the benefit of representing the feature at all.

### Condition 2: Nonlinearity Enables Superposition
- The critical insight is that adding a single ReLU to the output radically changes what the model can do.
- In a linear model, interference is symmetrically bad and always net-negative → superposition never optimal.
- With ReLU: **negative interference is free** in the 1-sparse case (ReLU clips to zero). A negative bias can further convert small positive interferences into effectively negative ones. This asymmetry enables superposition.

### Mathematical Understanding: Two Competing Forces
Loss decomposes into:
- **Feature benefit** ~ sum_i I_i (1 - ||W_i||^2)^2  [benefit from representing each feature]
- **Interference** ~ sum_{i≠j} I_j (W_j · W_i)^2  [cost when features are non-orthogonal]

For the linear model, these forces make superposition never optimal. For the ReLU model, the ReLU makes negative interference essentially free, changing the calculus so that sparse features can be profitably packed in superposition.

### Condition 3: Feature Importance Modulates the Phase Diagram
- Features with higher relative importance are less likely to go into superposition.
- Very important features get dedicated dimensions even at high sparsity.
- Less important features are more readily packed into superposition.
- At sufficient sparsity, even the most important features eventually enter superposition.

### Three Outcomes for Any Feature
1. Not learned at all (||W_i|| ≈ 0)
2. Learned in superposition (0 < ||W_i|| < 1, non-orthogonal to others)
3. Represented with a dedicated dimension (||W_i|| ≈ 1, orthogonal to others)

Transitions between these are sharp (phase transitions).

---

## 5. Phase Transitions and Geometric Structures

### The Phase Diagram
The paper constructs a 2D phase diagram with axes:
- **Feature density** (1 - S): how often features activate
- **Relative feature importance**: how important the "extra" feature is relative to others

Three regions in the phase diagram:
- **Not represented** (gray): feature importance too low or density too high
- **Dedicated dimension** (blue): feature is important enough to displace another
- **Superposition / antipodal pair** (red): feature is stored non-orthogonally

**Key result:** There is a genuine first-order phase change. The optimal weight configuration *discontinuously* changes in magnitude and degree of superposition as parameters cross a boundary. (Analytically confirmed: there is a crossover between loss functions for different weight configurations, causing a discontinuity in the derivative of the optimal loss.)

### Uniform Superposition and Polytopes
When all features have equal importance and sparsity, the model's feature embeddings organize into **specific geometric structures** corresponding to uniform polytopes.

The key metric: D* = m / ||W||_F^2 (dimensions per feature) plotted against sparsity.

**Sticky points** at rational fractions corresponding to specific polytopes:

| Dimensionality D_i | Geometry | Structure |
|---|---|---|
| 1 | Dedicated Dimension | 1 feat in 1 dim |
| 3/4 | Tetrahedron | 4 feats in 3 dims |
| 2/3 | Triangle | 3 feats in 2 dims |
| 1/2 | Digon (Antipodal Pair) | 2 feats in 1 dim |
| 2/5 | Pentagon | 5 feats in 2 dims |
| 3/8 | Square Antiprism | 8 feats in 3 dims |
| 0 | Not learned | — |

The stickiness at D = 1/2 (antipodal pairs) is particularly prominent — antipodal pairs are so effective that the model preferentially uses them over a wide range of sparsity. The authors compare this qualitative stickiness to the **fractional quantum Hall effect**.

### Connection to the Thomson Problem
The loss (in the uniform case with fixed ||W_i|| = 1) reduces to a **generalized Thomson problem**: packing points on the surface of an m-dimensional sphere with a slightly unusual energy function. The optimal configurations are the same uniform polyhedra that solve the classical Thomson problem in chemistry (which asks how to arrange n equal charges on a sphere to minimize energy). This explains why square antiprisms (Thomson solutions) appear in the toy model — they are not famous geometric shapes but are important solutions to charge-packing problems in molecular geometry.

### Tegum Products
Many Thomson solutions decompose as **tegum products**: constructions that embed two polytopes in orthogonal subspaces. For example:
- Triangular bipyramid = triangle × antipode (3 × 2/3 + 2 × 1/2 features)
- Pentagonal bipyramid = pentagon × antipode (5 × 2/5 + 2 × 1/2 features)
- Octahedron = three antipodes (3 × 1/2 features)

Tegum product structure implies: features in different tegum factors have **zero interference** with each other. This explains why models prefer tegum products — it minimizes compound interference.

### Polytopes and Low-Rank Matrices
There is an exact correspondence between:
- **Superposition strategies** (ways to pack n features in m dimensions)
- **Symmetric positive-definite low-rank matrices** W^T W
- **Polytopes** (sets of n points in m-dimensional space)

Every strategy for representing n features in m dimensions corresponds to a specific polytope, and vice versa. The optimal strategy for n equally important, equally sparse features is the most symmetric (uniform) polytope.

### Non-Uniform Superposition
When features differ in importance or sparsity:
- Varying a single feature's sparsity **continuously deforms** the polytope (e.g., a regular pentagon stretches as one feature becomes denser or sparser).
- At a critical threshold, there is a **discontinuous phase transition** — the pentagon collapses to a pair of digons (another first-order phase change confirmed by loss curve crossover).
- Non-uniform superposition is thus a "deformation of uniform superposition" before snapping to a different polytope.

### Correlated Features
- **Correlated features** (that co-occur together) strongly prefer to be represented **orthogonally**, often in different tegum factors. This creates a "local almost-orthogonal basis" within correlated feature sets even when the overall model is in superposition.
- When correlated features can't be orthogonal, they prefer to be **side-by-side** (positive interference preferred over negative).
- When capacity is insufficient, correlated features **collapse** into their principal component — a PCA-like solution for the pair.
- **Anti-correlated features** (at most one active at a time) prefer to be **antipodal** — negative interference is cheap because both are never simultaneously active.

### PCA vs. Superposition Tradeoff
- Dense + correlated features → PCA-like behavior dominates
- Sparse features → superposition dominates
- Both sparse and correlated → mixtures of both strategies

---

## 6. Connection to Feature Compression and Partial Mappings Between Concepts

### The Core Compression Idea
Superposition is the model's strategy for compressing more features than it has dimensions. The weight matrix W acts as a partial isometric encoder: each feature is compressed into a direction in the lower-dimensional space, and the ReLU nonlinearity at reconstruction time acts as a filter to suppress the interference noise.

This is directly analogous to **compressed sensing**: given a sparse signal in R^n, it can be encoded into a lower-dimensional R^m and then recovered, provided m = Ω(k log(n/k)) where k is the maximum number of simultaneously active features.

### Partial Maps Between Feature Spaces
The matrix W^T W (an n×n symmetric low-rank matrix) encodes the "overlap structure" of features in the lower-dimensional representation. Its off-diagonal terms encode interference between features. When W^T W is identity-like (diagonal), features are orthogonal and there is no superposition. When it is not identity-like, features are non-orthogonally embedded.

The paper establishes that W^T W corresponds exactly to a polytope (the set of feature embedding directions), making the geometry of superposition the geometry of the partial mapping from feature-space to neuron-space.

### Local Orthogonal Bases as Partial Isomorphisms
The finding that **correlated features form local almost-orthogonal bases** is directly relevant to partial isomorphisms: within a correlated feature set (e.g., the animal features cluster), the restriction of the representation to that cluster is nearly isometric/orthogonal — a near-isomorphism between the sub-cluster of features and a corresponding sub-space of the representation. Globally the representation is not isomorphic (it's in superposition), but locally (within co-occurring feature bundles) it is.

### "Virtual Neurons"
Term coined by Adam Jermyn in the paper's context: in a model with superposition, the "true" features correspond to virtual neurons in an imagined larger, monosemantic model. The actual model is a lossy projection of this virtual model. The mapping from the virtual model to the actual model is a partial, approximate structure-preserving map.

---

## 7. Unexpected / Counterintuitive Structures

The paper explicitly flags many results as surprising. These are particularly relevant to the "unexpected partial isomorphisms" research direction:

### 7a. The Fractional Quantum Hall Analogy
The "stickiness" of the D* = m/||W||_F^2 curve at rational fractions (1, 1/2, 2/3, 3/4, 2/5, 3/8, ...) is described as "very vaguely resembling the fractional quantum Hall effect." The plateaux correspond to stable geometric configurations analogous to incompressible quantum fluid states at fractional filling factors. This is counterintuitive: a simple 2-layer ReLU network exhibits behavior qualitatively analogous to a topological quantum phenomenon.

### 7b. Regular Polytopes from Gradient Descent
That gradient descent spontaneously finds solutions corresponding to known geometric objects from mathematics (tetrahedra, pentagons, square antiprisms) is described as "too elegant to be true." These are not explicitly encoded in the optimization — they emerge from the interaction of the loss function, sparsity, and nonlinearity.

### 7c. Discrete Energy Level Jumps During Training
Feature dimensionalities don't vary smoothly during training. Instead, they exhibit discrete **"energy level jumps"** where features abruptly transition from one fractional dimensionality to another (e.g., from 1/3 to 1/2). These jumps are accompanied by sudden drops in the loss curve. The authors suggest that seemingly smooth loss decreases in large models may be composed of many such micro-jumps.

### 7d. Learning as Geometric Transformations
The learning dynamics of the toy model for 6 correlated features in 3 dimensions can be decomposed into a sequence of distinct geometric transformations, each corresponding to a visible kink in the loss curve:
- A: Random initialization (near zero)
- B: Two correlated feature groups push apart along one axis
- C: Each group expands into a triangle
- D: Triangles rotate into an antiprism (final solution — equivalent to an octahedron)

This geometric decomposition of gradient descent dynamics is unexpected.

### 7e. Asymmetric Superposition Motif
In computation models (absolute value task), a discovered weight motif has no natural prior motivation:
- One neuron stores two features with **asymmetric weights** [2, -1/2], causing one feature to heavily interfere with the other but not vice versa.
- A second neuron then provides **targeted inhibition** to convert the problematic positive interference into a harmless negative one.
- This "asymmetric superposition with inhibition" motif has no obvious analogue in non-superposition circuits.

### 7f. "Confused Feature" Phase (from external replication)
Tom McGrath's (DeepMind) analytical work on the n=2, m=1 case found an additional unexpected phase: W_1 ≈ W_2 ≈ 1/sqrt(2) — the two features are represented in the **same direction** (not antipodal, not ignored). This "confused feature" regime occurs when sparsity is low and both features are important. It does not require feature correlation and was not anticipated by the original authors.

### 7g. Code-Like Structures in Privileged Basis
In the ReLU hidden layer model (with privileged basis), features at high sparsity begin to be represented as **binary codes** across multiple neurons:
- Three features in two neurons using a binary code
- Features as "pairs of neurons" (all neurons polysemantic)
- One neuron distinguishing important from unimportant features, others encoding specific identities

These combinatorial code structures emerge from gradient descent without explicit design.

---

## 8. Datasets / Benchmarks Used

**No real datasets.** The paper uses entirely **synthetic data** generated procedurally. This is a deliberate methodological choice: real-world feature structure is unknown, but the toy model requires knowing ground-truth features.

Synthetic data parameters:
- x_i = 0 with probability S (sparsity parameter)
- x_i ~ Uniform[0, 1] otherwise (in basic experiments)
- x_i ~ Uniform[-1, 1] in computation experiments (to make abs(x) non-trivial)
- Feature importance: I_i = 0.7^i, 0.8^i, 0.9^i (geometric decay) or I_i = 1 (uniform)

The toy model framework is specifically designed so that the ground-truth feature structure is known, enabling unambiguous demonstration of superposition. This is the key advantage of synthetic data: "we do know what the features are!"

The paper notes this as a major limitation of studying superposition in real models: "At present, it's challenging to study superposition in models because we have no ground truth for what the features are."

---

## 9. Code Availability

Two Colab notebooks provided (linked in paper):
1. **Toy model framework notebook** — reproduces core diagrams (feature embedding visualizations, basic superposition results, geometry plots). Note: was rewritten from internal codebase, not comprehensive.
2. **Theoretical phase change notebook** — analytical calculations for specific weight configurations, loss curves for different polytope solutions.

**GitHub repo:** mentioned but specific URL not given in paper text (Colab links to "our Github repo").

**Independent replications (not using paper's code):**
- Redwood Research (Kshitij Sachan): replicated Demonstrating Superposition, Phase Change, and Uniform Superposition Geometry sections. Found phase diagrams depend on activation function.
- DeepMind (Tom McGrath): replicated Demonstrating Superposition and Phase Change sections; added analytical solution for n=2, m=1 case.
- OpenAI (Jeffrey Wu and Dan Mossing): replicated Basic Results, Feature Dimensionality geometry, and Energy Level Jumps sections.

---

## 10. Key Figures and Interpretations

### Figure: Sparsity Sweep (Opening Figure)
Three panels showing 5 features embedded in 2 dimensions at 0%, 80%, 90% sparsity:
- **0% sparsity**: Top 2 features get orthogonal dedicated dimensions. Other 3 features are not embedded (zero vectors).
- **80% sparsity**: Top 4 features arranged as 2 antipodal pairs. Least important feature still not embedded.
- **90% sparsity**: All 5 features embedded as a pentagon. "Positive interference" label shows the non-orthogonality cost.
**Interpretation:** Sparsity drives the transition from PCA-like to superposition representation.

### Figure: Phase Diagram (n=2, m=1)
2D plot with axes: Feature Density (1-S) on y-axis (log scale), Relative Feature Importance on x-axis.
Three colored regions: gray (not represented), blue (dedicated dimension), red (superposition).
Both empirical and theoretical versions match closely.
**Key result:** Genuine first-order phase change between regions; the boundary is a real mathematical discontinuity.

### Figure: Uniform Superposition Geometry Plot
Line plot of D* = m/||W||_F^2 vs. 1/(1-S) with dotted lines at fractions 1, 1/2, 2/3, 3/4, 2/5, 3/8.
Scatter overlay of individual feature dimensionalities showing clustering at exactly those fractions.
Feature geometry graphs (insets) showing triangle, digon, pentagon, tetrahedron, square antiprism, and dense cloud configurations at different sparsity levels.
**Key result:** Superposition is not "one thing" — it has many sub-phases corresponding to specific geometric configurations. The stickiness at rational fractions resembles fractional quantum Hall plateaux.

### Figure: W^T W Matrix for Correlated Features
Block-diagonal structure showing two sets of 10 correlated features (n=20, m=10). Within each correlated set, the W^T W sub-block is near-identity (orthogonal representation). Between sets, there is interference.
**Key result:** Local orthogonal bases form within correlated feature clusters — a form of "local non-superposition" even within a globally superposed model.

### Figure: Feature Dimensionality Over Training (Energy Level Jumps)
Y-axis: feature dimensionality; X-axis: training steps (log scale). Multiple colored lines (one per feature) show: early in training all features have low dimensionality; then features abruptly jump to stable fractional values (0.5 for antipodal pairs) in discrete steps rather than continuous transitions. Loss curve (bottom panel) shows corresponding sudden drops.
**Key result:** Training dynamics are discrete, not continuous. Features make sudden jumps between geometric configurations.

### Figure: Geometric Transformations During Training (3D)
Four snapshots of 6 feature vectors in 3D space during training (top-down and side views):
- A: Random initialization (clustered near origin)
- B: Two groups (correlated pairs) push apart on one axis
- C: Each group expands into a triangle (3 distinct directions per group)
- D: Triangles rotate relative to each other, forming a triangular antiprism (equivalent to octahedron)
**Key result:** Gradient descent proceeds through recognizable geometric transformations. The loss curve shows corresponding regimes.

### Figure: Adversarial Vulnerability vs. Features Per Dimension
Two-panel figure: top shows adversarial vulnerability (relative to non-superposition model) vs. sparsity; bottom shows features per dimension (1/D*) vs. sparsity. The two curves closely parallel each other.
**Key result:** Adversarial vulnerability tracks the degree of superposition almost exactly. More features packed per dimension → more vulnerable.

### Figure: Computation in Superposition (Absolute Value, n=100, m=40)
Stacked weight bar plots at 7 sparsity levels (1-S from 1.0 to 0.001):
- Dense regime (1-S = 1.0, 0.3): All neurons monosemantic (each dedicated to one feature's abs computation).
- Intermediate (1-S = 0.1, 0.03): Monosemantic neurons handle important features; increasing polysemantic neurons handle less important ones.
- Sparse regime (1-S = 0.001): All neurons highly polysemantic.
**Key result:** Computation (not just storage) can be performed in superposition. The model implements abs(x) via superposed circuits at high sparsity.

### Figure: Asymmetric Superposition Motif (Circuit Diagram)
Two panels decomposing a partially-superposed absolute value circuit:
- Left: many simple abs(x_i) sub-circuits (monosemantic neurons)
- Right: two instances of "asymmetric superposition + inhibition" motif, each involving:
  - One neuron with asymmetric input weights [2, -1/2] for two features
  - One inhibitory neuron that neutralizes positive interference
**Key result:** A specific, nameable motif for how computation is done while in superposition. Unexpected mechanistic structure.

---

## 11. Connections to "Unexpected Partial Isomorphisms" Research

### How Superposition Creates Partial Structure-Preservation

The superposition hypothesis is fundamentally about **partial mappings** between two representations:
1. The "virtual" disentangled model (n features, n neurons, identity-like W^T W)
2. The "observed" compressed model (n features, m < n neurons, rank-m W^T W)

The map from virtual to observed is:
- A linear projection (compression) W : R^n → R^m
- The transposed linear projection (decompression) W^T : R^m → R^n
- A nonlinear filter (ReLU) that suppresses interference

This is not an isomorphism. However:
- **Globally**: W^T W is approximately identity-like (near isometry) in directions that are well-represented
- **Locally**: Within co-occurring feature clusters, the restriction is near-orthogonal (local isometry)
- **Structurally**: The pattern of which features are packed together (tegum product structure) preserves the correlation/anti-correlation structure of the input distribution

### Tegum Products as Structural Preservation
The preference for tegum product structures means that **orthogonal subspaces of the feature space** are mapped to **orthogonal subspaces of the representation**. The inter-factor structure (which features don't interfere with which) is preserved even though the within-factor structure may be compressed. This is a specific form of partial structure-preservation: the "independence graph" of features (which features can co-occur) maps to the "interference graph" of feature embeddings in a structured way.

### Anti-correlation → Antipodal Structure
The finding that anti-correlated features prefer antipodal embeddings is a specific case of an unexpected partial structure: the **sign structure** of the correlation (negative) maps to the **sign structure** of the interference (negative cosine similarity = antipodal). The model discovers this mapping without being explicitly told about it.

### "Local Non-Superposition" as a Partial Isomorphism Tool
The finding that correlated features form local orthogonal bases suggests that for any narrow enough sub-distribution of inputs (those that activate only correlated features), the model's representation is locally nearly isomorphic to the input feature space. The paper explicitly suggests this could enable "local non-superposition" analysis methods (like PCA) that would be unprincipled globally.

### The Superposition-as-Simulation Framing
The paper's core framing — that the observed model is a "noisy simulation" of a larger sparse model — is exactly the partial isomorphism framing: there exists a larger model M_large such that the observed model M_obs approximately implements M_large on sparse inputs. The quality of this approximation (the faithfulness of the partial isomorphism) degrades as features become less sparse or as more features are packed in per dimension.

---

## 12. Strategic Implications (for Interpretability Research)

### "Solving Superposition"
The paper argues that identifying and enumerating over all features is a necessary primitive for safety-relevant interpretability ("enumerative safety"). Three approaches:

1. **Create models without superposition**: L1 regularization on hidden activations removes superposition but at performance cost. Mixture of Experts (MoE) architectures may achieve similar capacity without the efficiency penalty of superposition by making neuron sparsity explicit.

2. **Find an overcomplete basis post-hoc**: Treat superposition as a sparse coding problem — given activation matrix [d × m] and assuming n underlying features, find matrices A [d × n] and B [n × m] such that A is sparse. Challenges: unknown n, major engineering scale, no training signal pushing toward sparsity.

3. **Hybrid approaches**: Modify training/architecture to reduce (not eliminate) superposition, then apply sparse coding to the partially-reduced result.

### Key Safety Argument
Without solving superposition:
- Feature enumeration is impossible (can't enumerate over what's in superposition)
- Circuit analysis is unreliable (weights connecting polysemantic neurons are hard to interpret)
- Even cosine similarity is misleading (unrelated features share positive dot products due to interference)
- The "semantic dictionary" approach to understanding activations fails

---

## 13. Open Questions Left by the Paper

1. Is there a statistical test for detecting superposition in a representation?
2. Can we estimate the feature importance curve and sparsity curve of real models?
3. Should we expect superposition to decrease with scale (if the feature importance curve is steep enough)?
4. How much of the geometric structure generalizes beyond the toy model?
5. What class of computation can be performed in superposition? Is sparse structure required?
6. Does superposition decrease efficiency (loss per FLOP) vs. dedicated neurons? (Preliminary evidence: superposition is not optimal per activation frequency, but asymptotically approaches it.)
7. How do phase changes in features connect to phase changes in compressed sensing?
8. What is the relationship between superposition and non-robust features?
9. How does feature correlation structure affect the optimal superposition packing?
10. Can models use nonlinear representations instead of superposition? (Paper argues: probably not, due to decompression cost and learning difficulty.)

---

## 14. Appendix Content (Skimmed)

### Nonlinear Compression Appendix
Analyzes whether models might use nonlinear feature representations instead of (or in addition to) linear superposition. The simplest example: encoding two [0,1) dimensions x and y into a single [0,1) dimension t = (floor(Zx) + y) / Z (quantization-based encoding). With epsilon-approximations of discontinuities, this outperforms linear compression (choosing one dimension or averaging) for sufficiently small epsilon. However, the paper argues nonlinear compression is unlikely to be used pervasively because: (a) decompression is needed before linear computation, (b) it may be hard for gradient descent to learn, (c) the neuron cost may not be worth it vs. superposition.

### Compressed Sensing Lower Bounds Appendix
Formal connection between compressed sensing and the toy model:

**Theorem 1:** If the toy model T(x) = ReLU(W_2 W_1 x - b) recovers all x with ||T(x) - x||_2 ≤ epsilon and W_1 has the (delta, k) restricted isometry property, then the inner dimension m = Omega(k log(n/k)).

**Proof sketch:** Frame the toy model as a compressed sensing algorithm, handle the non-exact-sparsity via a denoising lemma, then apply the deterministic CS lower bound of Do Ba et al.

Since k = O((1-S)n) (number of simultaneously active features is bounded by sparsity), the bound gives m = Omega(-n(1-S)log(1-S)), so the number of features is linear in m but modulated by sparsity. Higher sparsity → more features can be packed per dimension.

### Comments and Replications Appendix
External researchers confirmed core results:
- **Redwood Research (Kshitij Sachan)**: Replicated basic superposition, phase change, and uniform geometry. Found phase diagrams depend on activation function.
- **DeepMind (Tom McGrath)**: Replicated basic results + found new "confused feature" phase (W_1 ≈ W_2 ≈ 1/sqrt(2)) at low sparsity, high importance for both features. Found loss surface has two minima in some parameter regimes — connects to energy level jumps.
- **OpenAI (Jeffrey Wu, Dan Mossing)**: Replicated basic results, geometry, and energy level jumps.
