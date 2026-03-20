# Research Plan: Unexpected Partial Isomorphisms

## Motivation & Novelty Assessment

### Why This Research Matters
Humans naturally create analogies to reason about complex systems—calling Waymo sensors a "hat" or describing organizational structure with body metaphors ("head", "hands", "heart"). These analogies are *partial isomorphisms*: structure-preserving maps between domains that hold for some properties but not all. LLMs, trained via compression of vast text, may discover their own partial isomorphisms that are counterintuitive to humans but computationally useful. Understanding these mappings would (a) reveal novel reasoning strategies models employ, (b) potentially suggest new analogies useful for human understanding, and (c) deepen our understanding of how neural compression creates emergent conceptual structure.

### Gap in Existing Work
The literature on superposition (Elhage et al., 2022), polysemanticity (Scherlis et al., 2022), and representation convergence (Huh et al., 2024) establishes that models compress concepts together. The OthelloGPT work (Nanda et al., 2023) provides a striking example of an unexpected encoding (MINE/YOURS vs. BLACK/WHITE). However, **no systematic study has catalogued which unexpected concept pairings LLMs create, measured how "surprising" these pairings are relative to human intuition, or tested whether these pairings confer reasoning advantages**. The gap is between knowing that superposition exists and understanding what specific counterintuitive analogies it creates.

### Our Novel Contribution
We conduct a three-pronged investigation:
1. **Discovery**: Use embedding similarity analysis to find concept pairs that are close in LLM representation space but distant in human similarity judgments
2. **Validation via LLM probing**: Use GPT-4.1 to explore whether models generate and leverage unexpected analogical mappings when reasoning
3. **Mechanistic evidence**: Use TransformerLens on Pythia models to find shared activation patterns between semantically distant concepts

### Experiment Justification
- **Experiment 1** (Embedding Surprise): Needed to systematically discover candidate unexpected partial isomorphisms at scale, using quantitative comparison of model vs. human similarity
- **Experiment 2** (LLM Analogical Reasoning): Needed to test the functional hypothesis—do these unexpected mappings actually help reasoning? This uses real LLM API calls.
- **Experiment 3** (Activation Analysis): Needed to provide mechanistic evidence that the discovered mappings exist at the representation level, not just the behavioral level

## Research Question
Do LLMs internally use counterintuitive partial isomorphisms—mappings between concept domains that humans wouldn't naturally make—because compressing concepts together helps efficient reasoning?

## Background and Motivation
Analogical reasoning is fundamental to human cognition. We call sensors a "hat," describe code with "branches" and "trunks," and map organizational hierarchies to body parts. These partial isomorphisms preserve some structural relationships while ignoring others. LLMs, trained to predict text through massive compression, may discover their own set of useful partial isomorphisms that differ from human conventions. The superposition literature shows models routinely entangle features; the question is whether these entanglements form *meaningful* analogical structure.

## Hypothesis Decomposition
- **H1**: LLM embedding spaces contain concept pairs with high model-similarity but low human-similarity (unexpected proximity)
- **H2**: When prompted to reason using these unexpected analogies, LLMs perform comparably or better than with conventional analogies
- **H3**: At the activation level, semantically distant concepts that models treat as similar share activation patterns (polysemantic features)

## Proposed Methodology

### Approach
Three complementary experiments, progressing from broad discovery to mechanistic evidence:

### Experimental Steps

**Experiment 1: Embedding Surprise Discovery**
1. Curate ~200 concepts across 10+ domains (body, architecture, computing, cooking, music, warfare, biology, geography, finance, sports, etc.)
2. Compute pairwise cosine similarities using a state-of-the-art embedding model (text-embedding-3-large)
3. Compute human similarity scores using WordNet path similarity and ConceptNet relatedness
4. Define "surprise score" = model_similarity - human_similarity (normalized)
5. Identify top-50 most surprising pairs and categorize the structural relationships they share

**Experiment 2: LLM Analogical Reasoning Probing**
1. Take top surprising pairs from Exp 1
2. Construct reasoning tasks where the analogy between concepts is relevant
3. Test GPT-4.1 with: (a) no analogy hint, (b) conventional analogy hint, (c) unexpected analogy hint
4. Measure reasoning quality via accuracy on structured questions
5. Also: ask the model to freely generate analogies for concepts and measure how many are "conventional" vs. "novel"

**Experiment 3: Activation Analysis**
1. Use TransformerLens with Pythia-410M or Pythia-1.4B
2. Feed sentences about concept pairs identified as "surprising" in Exp 1
3. Extract MLP activations and compute cosine similarity between activation patterns
4. Compare activation similarity for surprising vs. non-surprising pairs
5. Identify specific neurons that activate for both concepts in a surprising pair

### Baselines
- Random concept pairs (null distribution for surprise scores)
- Conventional analogy pairs (e.g., "king:queen::man:woman") as positive controls
- Human-rated similar pairs as calibration

### Evaluation Metrics
- **Surprise score**: (model_sim - human_sim), measuring unexpectedness
- **Reasoning accuracy**: Performance on structured QA tasks with/without analogy hints
- **Activation overlap**: Cosine similarity of MLP activation patterns for concept pairs
- **Cohen's d**: Effect size for reasoning improvement from unexpected analogies

### Statistical Analysis Plan
- Wilcoxon signed-rank test for paired comparisons (non-normal distributions expected)
- Bootstrap confidence intervals for surprise scores
- Permutation tests for activation similarity significance
- Bonferroni correction for multiple comparisons
- α = 0.05

## Expected Outcomes
- **Supporting H1**: A distribution of surprise scores with a significant right tail (many unexpectedly similar pairs)
- **Supporting H2**: Comparable or improved reasoning when using unexpected analogies vs. no analogy
- **Supporting H3**: Higher activation overlap for model-surprising pairs than predicted by human similarity alone

## Timeline and Milestones
- Phase 1 (Planning): 20 min ✓
- Phase 2 (Environment + Data): 15 min
- Phase 3 (Implementation): 60 min
- Phase 4 (Experiments): 90 min
- Phase 5 (Analysis): 30 min
- Phase 6 (Documentation): 20 min

## Potential Challenges
- Embedding similarity may not capture the kind of analogical structure we're looking for (mitigation: use multiple embedding models)
- Human similarity baselines may be noisy (mitigation: use multiple sources—WordNet + ConceptNet)
- Activation analysis on Pythia may not reveal clean patterns (mitigation: use multiple layers, aggregate)
- API rate limits (mitigation: cache all responses, batch requests)

## Success Criteria
1. Identify ≥20 concept pairs with significantly elevated surprise scores
2. Qualitative analysis shows these pairs share meaningful structural properties
3. At least one experiment shows statistically significant evidence for the hypothesis
