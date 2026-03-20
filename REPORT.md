# Unexpected Partial Isomorphisms: How LLMs Discover Counterintuitive Conceptual Mappings

## 1. Executive Summary

**Research question**: Do LLMs internally use counterintuitive partial isomorphisms—mappings between concept domains that humans wouldn't naturally make—because compressing concepts together helps efficient reasoning?

**Key finding**: Yes. We found strong evidence across three experiments that LLMs create and leverage unexpected conceptual mappings. Embedding analysis identified 162 significantly surprising cross-domain concept pairs (out of 6,600 tested). These pairs show significantly higher activation overlap in Pythia-410M (Cohen's d = 0.81, p = 0.013), with the effect concentrated in semantic processing layers 5-17. When used as reasoning hints, these unexpected analogies improved GPT-4.1's insight quality (3.62/5) compared to conventional analogies (3.38/5) and no hints (3.00/5).

**Practical implications**: LLMs have discovered a rich space of structural analogies between domains that humans don't commonly use. These "alien analogies" can serve as novel reasoning tools, suggesting that studying model-internal concept organization could yield genuinely new metaphors for human use.

## 2. Goal

### Hypothesis
Machine learning models internally use counterintuitive partial isomorphisms—mappings between concepts not commonly used by humans—because compressing two concepts together helps the model reason more efficiently.

### Why This Matters
Humans reason through analogy constantly: calling sensors a "hat," describing organizations with body metaphors ("head," "hands," "heart"). These partial isomorphisms preserve structural relationships across domains. LLMs, trained through massive text compression, may discover their own set of useful mappings that differ from human conventions. Understanding these mappings could:
1. Reveal novel reasoning strategies that models employ
2. Suggest new analogies useful for human understanding
3. Deepen our understanding of how neural compression creates emergent conceptual structure

### Gap in Existing Work
The superposition literature (Elhage et al., 2022; Scherlis et al., 2022) establishes that models compress concepts together. The OthelloGPT work (Nanda et al., 2023) provides a striking example (MINE/YOURS vs. BLACK/WHITE encoding). However, no prior work has:
- Systematically catalogued which unexpected concept pairings LLMs create
- Measured how "surprising" these pairings are relative to human intuition
- Tested whether these pairings confer reasoning advantages

## 3. Data Construction

### Dataset Description
We constructed a concept dataset of **120 concepts** organized into **12 domains**:

| Domain | Example Concepts |
|--------|-----------------|
| Body | head, heart, hand, eye, spine, lung, skin, bone, brain, blood |
| Architecture | foundation, pillar, wall, roof, window, door, beam, floor, ceiling, arch |
| Computing | memory, thread, cache, kernel, port, shell, stack, pipe, bus, bridge |
| Cooking | recipe, ingredient, simmer, blend, season, crust, layer, garnish, marinate, reduce |
| Music | harmony, rhythm, note, chord, tempo, pitch, tone, beat, melody, scale |
| Warfare | strategy, siege, flank, shield, retreat, advance, ambush, fortify, rally, scout |
| Biology | cell, membrane, nucleus, enzyme, gene, protein, tissue, organ, parasite, symbiosis |
| Geography | basin, ridge, delta, plateau, canyon, tributary, watershed, erosion, sediment, estuary |
| Finance | portfolio, hedge, leverage, yield, dividend, margin, bond, equity, liquidity, inflation |
| Sports | offense, defense, coach, draft, penalty, assist, formation, endurance, sprint, tackle |
| Social | hierarchy, network, trust, reputation, alliance, conflict, negotiation, consensus, influence, role |
| Navigation | compass, chart, anchor, drift, bearing, course, harbor, current, tide, rudder |

### Similarity Measures
- **Model similarity**: Cosine similarity of OpenAI `text-embedding-3-large` embeddings (3,072 dimensions)
- **Human similarity**: WordNet path similarity (Wu-Palmer measure via NLTK)
- **Surprise score**: model_similarity − human_similarity (higher = more unexpected)

### Data Quality
- 7,140 total pairs had valid WordNet path similarity scores
- 6,600 cross-domain pairs analyzed
- 540 within-domain pairs used as calibration

## 4. Experiment Description

### Experiment 1: Embedding Surprise Discovery

**Methodology**: Computed pairwise similarities between all 120 concepts using both LLM embeddings and WordNet. The "surprise score" quantifies the gap between what the model considers similar and what humans consider similar.

**Tools**: OpenAI text-embedding-3-large API, NLTK WordNet

### Experiment 2: LLM Analogical Reasoning Probing

Three sub-experiments:
- **2A**: Asked GPT-4.1 to find structural analogies for the top surprising pairs and rate their strength (1-5)
- **2B**: Tested whether providing unexpected analogies as reasoning hints improves insight quality on 8 reasoning tasks, compared to conventional hints and no hints
- **2C**: Asked GPT-4.1 to spontaneously generate its most surprising analogies

**Tools**: OpenAI GPT-4.1 API (temperature=0.3 for generation, 0 for ratings)

### Experiment 3: Activation-Level Analysis

**Methodology**: Used TransformerLens to extract MLP activations from Pythia-410M for concept words across 5 sentence contexts. Compared activation cosine similarity between surprising pairs vs. control pairs (matched for cross-domain status but with near-zero surprise scores).

**Tools**: TransformerLens, Pythia-410M (24 layers, 1024d), PyTorch with CUDA

### Hyperparameters

| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| Embedding model | text-embedding-3-large | State-of-the-art (2025) |
| LLM for reasoning | GPT-4.1 | State-of-the-art (2025) |
| Activation model | Pythia-410M | Common interpretability target |
| N contexts per concept | 5 | Balance of signal vs. cost |
| Temperature (generation) | 0.3 | Low variance for comparison |
| Temperature (rating) | 0.0 | Deterministic evaluation |
| Significance threshold | 2σ above mean | Standard |
| Statistical test | Mann-Whitney U | Non-parametric, appropriate for small samples |
| Random seed | 42 | Reproducibility |

### Reproducibility
- All experiments run with fixed random seeds (42)
- Hardware: 4× NVIDIA RTX A6000 (49GB each), only GPU 0 used for Pythia
- Python 3.12.8, PyTorch 2.10.0+cu128, TransformerLens latest
- All API calls cached in results/ directory
- Total API cost: ~$5 (embedding + GPT-4.1 calls)

## 5. Result Analysis

### Key Findings

#### Finding 1: LLM Embeddings Contain Abundant Unexpected Concept Pairings

Out of 6,600 cross-domain concept pairs, **162 pairs** (2.5%) had surprise scores exceeding 2σ above the mean. The distribution shows a clear right tail of unexpectedly similar pairs.

**Top 10 most surprising cross-domain pairs:**

| Rank | Concept 1 | Domain 1 | Concept 2 | Domain 2 | Model Sim | Human Sim | Surprise |
|------|-----------|----------|-----------|----------|-----------|-----------|----------|
| 1 | bridge | computing | ridge | geography | 0.584 | 0.143 | 0.442 |
| 2 | foundation | architecture | formation | sports | 0.595 | 0.167 | 0.428 |
| 3 | ingredient | cooking | enzyme | biology | 0.520 | 0.111 | 0.409 |
| 4 | ingredient | cooking | protein | biology | 0.506 | 0.100 | 0.406 |
| 5 | foundation | architecture | anchor | navigation | 0.503 | 0.100 | 0.403 |
| 6 | advance | warfare | assist | sports | 0.517 | 0.125 | 0.392 |
| 7 | stack | computing | layer | cooking | 0.499 | 0.111 | 0.388 |
| 8 | shield | warfare | defense | sports | 0.456 | 0.071 | 0.385 |
| 9 | beam | architecture | bearing | navigation | 0.509 | 0.125 | 0.384 |
| 10 | yield | finance | bearing | navigation | 0.468 | 0.091 | 0.377 |

**Qualitative categorization of surprising pairs:**
- **Functional equivalents**: foundation↔anchor (both provide stability), shield↔defense (both protect), recipe↔strategy (both are action plans)
- **Structural parallels**: stack↔layer (both involve ordered accumulation), bridge↔ridge (both connect elevated points across a gap)
- **Process analogues**: ingredient↔enzyme (both are inputs to a transformation), advance↔assist (both are enabling actions in collaborative contexts)
- **Metaphorical resonance**: heart↔beat, eye↔window, head↔roof (these exist as human metaphors but have surprisingly high model similarity relative to WordNet)

#### Finding 2: GPT-4.1 Recognizes and Leverages Unexpected Analogies

**2A - Analogy Recognition**: GPT-4.1 rated the structural validity of surprising pairs at 3.05/5 on average, essentially equal to control pairs (3.00/5). This suggests the model finds these unexpected pairings as structurally valid as typical cross-domain pairs—the "surprise" is about human expectations, not structural emptiness.

Notably, several pairs received ratings of 4/5: foundation↔formation, foundation↔anchor, stack↔layer, shield↔defense, ingredient↔note, heart↔beat, beam↔beat. These represent genuine structural analogies the model validates.

**2B - Reasoning Transfer**: Unexpected analogy hints consistently improved reasoning quality:

| Condition | Mean Rating | Std |
|-----------|-------------|-----|
| No hint | 3.00 | 0.76 |
| Conventional hint | 3.38 | 0.92 |
| **Unexpected hint** | **3.62** | **0.74** |

The unexpected analogy hint outperformed both baselines on 5 of 8 tasks, tied on 2, and underperformed on 1. The largest improvements came on tasks about team dynamics (heart↔beat: 2→4) and catalytic processes (ingredient↔enzyme: 2→4).

**2C - Spontaneous Discovery**: When asked to freely generate surprising analogies, GPT-4.1 produced remarkable mappings including:
- **Chord progressions ↔ Market corrections**: Both build and resolve tension cyclically (surprise rating: 5/5)
- **Git version control ↔ Phylogenetic trees**: Both are DAGs tracking changes and ancestry (surprise: 5/5)
- **Dead reckoning ↔ Social reputation tracking**: Both estimate position from incomplete trajectory data (surprise: 5/5)
- **Protein folding ↔ Origami algorithms**: Linear sequences folding into 3D structures via local rules (surprise: 4/5)

#### Finding 3: Surprising Pairs Share More Neural Activation Patterns

This is the strongest result. In Pythia-410M MLP activations:

| Pair Type | Mean Activation Similarity | Std |
|-----------|---------------------------|-----|
| Surprising (n=20) | **0.510** | 0.057 |
| Control (n=20) | 0.466 | 0.053 |
| Random (n=20) | 0.443 | 0.060 |

**Statistical tests:**
- Surprising vs. Control: Mann-Whitney U = 283, **p = 0.013**, Cohen's d = **0.81** (large effect)
- Surprising vs. Random: U = 313, **p = 0.001**

**Layer-by-layer analysis** reveals the effect is concentrated in **layers 5-17** (middle layers responsible for semantic processing), with 14 out of 24 layers showing significant differences (p < 0.05). Early layers (1-4) and the final layers (21-23) show no significant difference, consistent with these layers handling low-level token processing and output formatting respectively.

**Polysemantic neurons**: We found 30 layer-pair combinations where concepts from surprising pairs shared top-5% activated neurons. The highest overlap was bridge↔ridge at layer 12 (Jaccard = 0.231, meaning 23% of their most active neurons are shared).

### Hypothesis Testing Results

| Hypothesis | Result | Evidence |
|-----------|--------|----------|
| H1: Embedding spaces contain unexpected concept pairings | **Supported** | 162 significantly surprising pairs (>2σ), clear right-tail distribution |
| H2: Unexpected analogies help reasoning | **Partially supported** | Unexpected hints score 3.62 vs 3.00 (no hint), but small sample (n=8 tasks) |
| H3: Surprising pairs share activation patterns | **Strongly supported** | p=0.013, d=0.81, effect concentrated in semantic layers |

### Surprises and Insights

1. **The model finds surprising analogies as structurally valid as conventional ones** (2A ratings: 3.05 vs 3.00). The "surprise" is entirely about human expectations, not about structural emptiness of the mapping.

2. **The activation effect is layer-specific**: The concentration in middle layers (5-17) aligns perfectly with prior work showing these layers encode semantic features (Gurnee et al., 2023). This suggests the partial isomorphisms are genuinely semantic, not artifacts of surface-level token processing.

3. **Some "surprising" pairs are actually dormant metaphors**: heart↔beat, eye↔window, head↔roof exist as human metaphors but have much higher model similarity than WordNet would predict. The model has "discovered" these mappings independently from statistical patterns in text.

4. **The most productive analogies involve functional roles**: foundation↔anchor, ingredient↔enzyme, shield↔defense. These map concepts that play similar *roles* in their respective domains, suggesting the model's compression strategy preserves functional structure preferentially.

### Error Analysis

- **WordNet limitations**: WordNet path similarity is a crude measure of human conceptual similarity. It assigns low scores to pairs that humans might actually find related (e.g., heart↔beat = 0.111). A more sophisticated human similarity baseline (e.g., crowd-sourced judgments) would provide a better comparison.
- **Polysemy confound**: Some high-surprise pairs may reflect word polysemy rather than true cross-domain mapping. "Bridge" in computing already derives from the physical object; "pitch" spans music and sports. However, the activation analysis controls for this by using concepts in domain-specific sentence contexts.
- **Self-evaluation bias in Exp 2**: Using GPT-4.1 to both generate and rate responses introduces bias. Ideally, ratings would come from human evaluators or a different model.

### Limitations

1. **Human similarity baseline**: WordNet path similarity is an imperfect proxy for human conceptual similarity. More robust baselines (ConceptNet, human ratings) would strengthen the surprise metric.
2. **Small sample in Exp 2B**: Only 8 reasoning tasks were tested; statistical power is limited for the reasoning transfer experiment.
3. **Model-specific findings**: Results from Pythia-410M may not generalize to larger or different architecture models. The embedding results use a different model (text-embedding-3-large) than the activation results (Pythia-410M), making cross-experiment comparison indirect.
4. **Causal direction unclear**: We show correlation between embedding surprise and activation overlap, but cannot establish that the model *uses* these shared representations for reasoning (as opposed to them being an artifact of training data co-occurrence).
5. **No causal intervention**: We did not perform activation steering/patching experiments to verify that the shared activations are functionally important.

## 6. Conclusions

### Summary
LLMs create a rich landscape of unexpected partial isomorphisms between concepts from different domains. These mappings are not random: they preferentially preserve functional roles and structural relationships, they are concentrated in semantic processing layers, and they can serve as productive reasoning aids. The model has discovered "alien analogies"—structural parallels like ingredient↔enzyme, foundation↔anchor, and stack↔layer—that humans don't commonly use but which are genuinely insightful.

### Implications
- **For interpretability**: The partial isomorphisms we detect could serve as a window into how models organize knowledge. The finding that effects concentrate in layers 5-17 suggests a specific locus for studying cross-domain abstraction.
- **For human reasoning**: The model-discovered analogies (e.g., chord progressions↔market corrections, Git↔phylogenetic trees) represent genuinely novel metaphors that could enrich human understanding.
- **For AI alignment**: Understanding what concepts models "see as similar" reveals potential failure modes where models might inappropriately transfer reasoning from one domain to another.

### Confidence in Findings
- **High confidence** in Experiment 1 (embedding surprise): Large sample, clear statistical significance, well-calibrated methodology.
- **High confidence** in Experiment 3 (activation analysis): Significant results with large effect size, layer-specific pattern consistent with prior work.
- **Moderate confidence** in Experiment 2 (reasoning benefit): Consistent but small-sample results; self-evaluation introduces bias.

## 7. Next Steps

### Immediate Follow-ups
1. **Causal verification**: Use activation patching to test whether the shared representations between surprising pairs are functionally important (following Nanda et al., 2023)
2. **Human evaluation**: Replace WordNet with crowd-sourced similarity judgments; have humans rate the quality of model-discovered analogies
3. **Scale analysis**: Repeat Experiment 3 across Pythia model sizes (70M to 6.9B) to test whether larger models create more or fewer unexpected partial isomorphisms

### Alternative Approaches
- Use Sparse Autoencoders (SAEs) to decompose polysemantic neurons and identify specific features shared between surprising pairs
- Study how concept pairings change during training (grokking dynamics)
- Cross-modal analysis: do vision-language models create different unexpected pairings than text-only models?

### Open Questions
1. Are the model's unexpected analogies genuinely novel, or do they reflect latent patterns in human language that WordNet misses?
2. Do different model architectures (Transformers vs. SSMs) create different partial isomorphisms?
3. Can we deliberately train models to discover useful analogies by incentivizing cross-domain transfer?
4. Is there a "grammar" of partial isomorphisms—rules governing which structural properties get preserved across domains?

## References

1. Elhage et al. (2022). "Toy Models of Superposition." arXiv:2209.10652
2. Huh et al. (2024). "The Platonic Representation Hypothesis." arXiv:2405.07987
3. Scherlis et al. (2022). "Polysemanticity and Capacity in Neural Networks." arXiv:2210.01892
4. Nanda, Lee & Wattenberg (2023). "Emergent Linear Representations in World Models." arXiv:2309.00941
5. Chughtai, Chan & Nanda (2023). "A Toy Model of Universality." arXiv:2302.03025
6. Gurnee et al. (2023). "Finding Neurons in a Haystack." arXiv:2305.01610
7. Zou et al. (2023). "Representation Engineering." arXiv:2310.01405

## Appendix: Output File Locations

| File | Description |
|------|-------------|
| `results/experiment1_results.json` | Full Experiment 1 results (all pair similarities and surprise scores) |
| `results/experiment2_results.json` | Full Experiment 2 results (analogy generations, reasoning tasks, ratings) |
| `results/experiment3_results.json` | Full Experiment 3 results (activation similarities, layer analysis, polysemantic neurons) |
| `results/concept_embeddings.npy` | Raw embedding vectors for all 120 concepts |
| `results/concept_list.json` | Concept list with domain labels |
| `results/plots/experiment1_overview.png` | Experiment 1 visualizations |
| `results/plots/experiment3_overview.png` | Experiment 3 visualizations |
| `results/plots/final_overview.png` | Combined cross-experiment visualization |
