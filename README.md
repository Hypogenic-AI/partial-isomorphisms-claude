# Unexpected Partial Isomorphisms

**Can we find counterintuitive conceptual mappings that LLMs use internally but humans don't commonly make?** This project investigates whether language models discover "alien analogies"—structural parallels between concepts from different domains that are surprising to humans but useful for model reasoning.

## Key Findings

- **162 significantly surprising concept pairs** identified out of 6,600 cross-domain comparisons (e.g., ingredient↔enzyme, foundation↔anchor, stack↔layer)
- **Surprising pairs share more neural activation patterns** in Pythia-410M: Cohen's d = 0.81, p = 0.013, with the effect concentrated in semantic processing layers 5-17
- **Unexpected analogy hints improve GPT-4.1 reasoning quality** (3.62/5 vs 3.00/5 without hints, 3.38/5 with conventional hints)
- The model-discovered analogies preserve **functional roles** preferentially: concepts that play similar roles in their domains get compressed together

## Project Structure

```
├── REPORT.md                          # Full research report with results
├── planning.md                        # Research plan and methodology
├── src/
│   ├── experiment1_embedding_surprise.py   # Embedding-based surprise discovery
│   ├── experiment2_analogical_reasoning.py # LLM reasoning probing (GPT-4.1)
│   ├── experiment3_activation_analysis.py  # TransformerLens activation analysis
│   ├── visualize_exp1.py                   # Experiment 1 visualizations
│   └── final_analysis.py                   # Cross-experiment analysis
├── results/
│   ├── experiment1_results.json       # Embedding similarity & surprise scores
│   ├── experiment2_results.json       # Analogy generation & reasoning results
│   ├── experiment3_results.json       # Activation analysis results
│   └── plots/                         # Visualizations
├── literature_review.md               # Background literature synthesis
├── resources.md                       # Resource catalog
├── papers/                            # Downloaded research papers
├── datasets/                          # Pre-generated datasets
└── code/                              # Reference implementations
```

## Reproduction

```bash
# Setup
uv venv && source .venv/bin/activate
uv add numpy scipy matplotlib seaborn pandas openai nltk torch transformers transformer-lens
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Run experiments (requires OPENAI_API_KEY)
python src/experiment1_embedding_surprise.py
python src/experiment2_analogical_reasoning.py
python src/experiment3_activation_analysis.py

# Generate visualizations
python src/visualize_exp1.py
python src/final_analysis.py
```

**Requirements**: Python 3.12+, CUDA GPU (for Experiment 3), OpenAI API key

## See Also

- [Full Report](REPORT.md) for detailed methodology, results, and analysis
- [Planning Document](planning.md) for experimental design rationale
