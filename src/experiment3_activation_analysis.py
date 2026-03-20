"""
Experiment 3: Activation-Level Analysis with TransformerLens
Examine whether concept pairs identified as 'surprising' in Experiment 1
share more activation patterns than expected, providing mechanistic evidence
for unexpected partial isomorphisms.
"""

import os
import json
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load Experiment 1 results
with open("results/experiment1_results.json") as f:
    exp1 = json.load(f)

top_surprising = exp1["top_surprising_cross_domain"][:20]
all_cross = exp1["all_cross_domain"]
# Get low-surprise pairs as controls
low_surprise = [p for p in all_cross if abs(p["surprise_score"]) < 0.05]
random.shuffle(low_surprise)
control_pairs = low_surprise[:20]

print("Loading TransformerLens model...")
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("pythia-410m", device="cuda:0")
print(f"Model loaded: {model.cfg.model_name}, {model.cfg.n_layers} layers, {model.cfg.d_model}d")

def get_concept_activations(concept, n_contexts=5):
    """Get MLP activations for a concept across multiple sentence contexts."""
    contexts = [
        f"The {concept} is an important part of the system.",
        f"We need to consider the {concept} carefully.",
        f"The role of the {concept} cannot be understated.",
        f"Understanding the {concept} helps us see the bigger picture.",
        f"The {concept} connects to many other elements.",
    ]

    all_activations = []
    for ctx in contexts[:n_contexts]:
        tokens = model.to_tokens(ctx)
        # Find the position of the concept token
        str_tokens = model.to_str_tokens(ctx)
        concept_positions = []
        for i, t in enumerate(str_tokens):
            if concept.lower() in t.lower().strip():
                concept_positions.append(i)

        if not concept_positions:
            # Use all tokens average as fallback
            concept_positions = list(range(1, len(str_tokens) - 1))

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Extract MLP output activations at concept positions, across all layers
        layer_acts = []
        for layer in range(model.cfg.n_layers):
            mlp_out = cache[f"blocks.{layer}.mlp.hook_post"][0]  # [seq_len, d_mlp]
            concept_act = mlp_out[concept_positions].mean(dim=0)  # [d_mlp]
            layer_acts.append(concept_act.cpu().numpy())

        all_activations.append(np.stack(layer_acts))  # [n_layers, d_mlp]

    # Average across contexts
    avg_activation = np.mean(all_activations, axis=0)  # [n_layers, d_mlp]
    return avg_activation

def activation_similarity(act1, act2):
    """Compute cosine similarity between activation patterns, per layer and overall."""
    per_layer = []
    for l in range(act1.shape[0]):
        a1, a2 = act1[l], act2[l]
        cos = np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2) + 1e-10)
        per_layer.append(cos)
    overall = np.mean(per_layer)
    return overall, per_layer

print("\nComputing activations for surprising pairs...")
surprising_similarities = []
surprising_layer_sims = []

for i, pair in enumerate(top_surprising):
    print(f"  [{i+1}/20] {pair['word1']} <-> {pair['word2']}...", end=" ", flush=True)
    act1 = get_concept_activations(pair["word1"])
    act2 = get_concept_activations(pair["word2"])
    overall, per_layer = activation_similarity(act1, act2)
    surprising_similarities.append(overall)
    surprising_layer_sims.append(per_layer)
    print(f"sim={overall:.4f}")

print("\nComputing activations for control pairs...")
control_similarities = []
control_layer_sims = []

for i, pair in enumerate(control_pairs):
    print(f"  [{i+1}/20] {pair['word1']} <-> {pair['word2']}...", end=" ", flush=True)
    act1 = get_concept_activations(pair["word1"])
    act2 = get_concept_activations(pair["word2"])
    overall, per_layer = activation_similarity(act1, act2)
    control_similarities.append(overall)
    control_layer_sims.append(per_layer)
    print(f"sim={overall:.4f}")

# Also compute random baseline
print("\nComputing random pair baseline...")
all_concepts = list(set([p["word1"] for p in all_cross] + [p["word2"] for p in all_cross]))
random_pairs = [(random.choice(all_concepts), random.choice(all_concepts)) for _ in range(20)]
random_similarities = []

for i, (w1, w2) in enumerate(random_pairs):
    if w1 == w2:
        continue
    act1 = get_concept_activations(w1)
    act2 = get_concept_activations(w2)
    overall, _ = activation_similarity(act1, act2)
    random_similarities.append(overall)

# Statistical analysis
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

surp_mean = np.mean(surprising_similarities)
ctrl_mean = np.mean(control_similarities)
rand_mean = np.mean(random_similarities)

print(f"\nMean activation similarity:")
print(f"  Surprising pairs: {surp_mean:.4f} ± {np.std(surprising_similarities):.4f}")
print(f"  Control pairs:    {ctrl_mean:.4f} ± {np.std(control_similarities):.4f}")
print(f"  Random pairs:     {rand_mean:.4f} ± {np.std(random_similarities):.4f}")

# Wilcoxon test: surprising vs control
stat, p_value = stats.mannwhitneyu(surprising_similarities, control_similarities, alternative='greater')
print(f"\nMann-Whitney U (surprising > control): U={stat:.1f}, p={p_value:.4f}")

stat2, p_value2 = stats.mannwhitneyu(surprising_similarities, random_similarities, alternative='greater')
print(f"Mann-Whitney U (surprising > random):  U={stat2:.1f}, p={p_value2:.4f}")

# Effect sizes (Cohen's d)
cohens_d = (surp_mean - ctrl_mean) / np.sqrt((np.std(surprising_similarities)**2 + np.std(control_similarities)**2) / 2)
print(f"\nCohen's d (surprising vs control): {cohens_d:.3f}")

# Layer-by-layer analysis
print("\nPer-layer activation similarity (surprising pairs):")
surp_layers = np.array(surprising_layer_sims)
ctrl_layers = np.array(control_layer_sims)
for l in range(model.cfg.n_layers):
    sl = surp_layers[:, l]
    cl = ctrl_layers[:, l]
    _, p = stats.mannwhitneyu(sl, cl, alternative='greater')
    sig = "*" if p < 0.05 else " "
    print(f"  Layer {l:2d}: surprising={np.mean(sl):.4f}  control={np.mean(cl):.4f}  diff={np.mean(sl)-np.mean(cl):.4f}  p={p:.3f} {sig}")

# Identify polysemantic neurons
print("\n\nSearching for polysemantic neurons (shared top activations)...")
# For each surprising pair, find neurons that are highly activated by both concepts
polysemantic_neurons = []
for i, pair in enumerate(top_surprising[:10]):
    act1 = get_concept_activations(pair["word1"])
    act2 = get_concept_activations(pair["word2"])

    for layer in [model.cfg.n_layers // 4, model.cfg.n_layers // 2, 3 * model.cfg.n_layers // 4]:
        a1 = act1[layer]
        a2 = act2[layer]
        # Find neurons in top 5% for both concepts
        thresh1 = np.percentile(np.abs(a1), 95)
        thresh2 = np.percentile(np.abs(a2), 95)
        top1 = set(np.where(np.abs(a1) > thresh1)[0])
        top2 = set(np.where(np.abs(a2) > thresh2)[0])
        shared = top1 & top2
        if shared:
            polysemantic_neurons.append({
                "pair": (pair["word1"], pair["word2"]),
                "layer": layer,
                "n_shared_top_neurons": len(shared),
                "n_top1": len(top1),
                "n_top2": len(top2),
                "jaccard": len(shared) / len(top1 | top2),
            })

print(f"  Found {len(polysemantic_neurons)} layer-pair combos with shared top neurons")
for pn in polysemantic_neurons[:10]:
    print(f"    {pn['pair'][0]:12s} <-> {pn['pair'][1]:12s} layer={pn['layer']:2d}: "
          f"shared={pn['n_shared_top_neurons']}, jaccard={pn['jaccard']:.3f}")

# Save results
output = {
    "model": "pythia-410m",
    "n_layers": model.cfg.n_layers,
    "d_model": model.cfg.d_model,
    "surprising_pairs": [
        {"word1": p["word1"], "word2": p["word2"],
         "surprise_score": p["surprise_score"],
         "activation_similarity": float(s)}
        for p, s in zip(top_surprising, surprising_similarities)
    ],
    "control_pairs": [
        {"word1": p["word1"], "word2": p["word2"],
         "surprise_score": p["surprise_score"],
         "activation_similarity": float(s)}
        for p, s in zip(control_pairs, control_similarities)
    ],
    "statistics": {
        "surprising_mean": float(surp_mean),
        "surprising_std": float(np.std(surprising_similarities)),
        "control_mean": float(ctrl_mean),
        "control_std": float(np.std(control_similarities)),
        "random_mean": float(rand_mean),
        "random_std": float(np.std(random_similarities)),
        "mann_whitney_surp_vs_ctrl_p": float(p_value),
        "mann_whitney_surp_vs_rand_p": float(p_value2),
        "cohens_d_surp_vs_ctrl": float(cohens_d),
    },
    "per_layer_surprising": surp_layers.tolist(),
    "per_layer_control": ctrl_layers.tolist(),
    "polysemantic_neurons": polysemantic_neurons,
}

with open("results/experiment3_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nResults saved to results/experiment3_results.json")

###############################################################################
# Visualizations
###############################################################################
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Box plot comparison
ax = axes[0, 0]
data_for_box = [surprising_similarities, control_similarities, random_similarities]
bp = ax.boxplot(data_for_box, labels=['Surprising\nPairs', 'Control\nPairs', 'Random\nPairs'],
                patch_artist=True)
colors = ['#e74c3c', '#3498db', '#95a5a6']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Activation Cosine Similarity')
ax.set_title(f'Activation Similarity Comparison\n(p={p_value:.4f}, d={cohens_d:.3f})')

# 2. Layer-by-layer comparison
ax = axes[0, 1]
layers = range(model.cfg.n_layers)
surp_means = surp_layers.mean(axis=0)
ctrl_means = ctrl_layers.mean(axis=0)
surp_stds = surp_layers.std(axis=0)
ctrl_stds = ctrl_layers.std(axis=0)
ax.plot(layers, surp_means, 'r-', label='Surprising pairs', linewidth=2)
ax.fill_between(layers, surp_means - surp_stds, surp_means + surp_stds, alpha=0.2, color='red')
ax.plot(layers, ctrl_means, 'b-', label='Control pairs', linewidth=2)
ax.fill_between(layers, ctrl_means - ctrl_stds, ctrl_means + ctrl_stds, alpha=0.2, color='blue')
ax.set_xlabel('Layer')
ax.set_ylabel('Cosine Similarity')
ax.set_title('Per-Layer Activation Similarity')
ax.legend()

# 3. Scatter: surprise score vs activation similarity
ax = axes[1, 0]
surp_x = [p["surprise_score"] for p in top_surprising]
surp_y = surprising_similarities
ctrl_x = [p["surprise_score"] for p in control_pairs]
ctrl_y = control_similarities
ax.scatter(surp_x, surp_y, c='red', alpha=0.7, s=50, label='Surprising', zorder=5)
ax.scatter(ctrl_x, ctrl_y, c='blue', alpha=0.7, s=50, label='Control', zorder=5)
# Fit trend line
all_x = surp_x + ctrl_x
all_y = list(surp_y) + list(ctrl_y)
slope, intercept, r, p_corr, se = stats.linregress(all_x, all_y)
x_line = np.linspace(min(all_x), max(all_x), 100)
ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5, label=f'r={r:.3f}, p={p_corr:.3f}')
ax.set_xlabel('Surprise Score (Embedding)')
ax.set_ylabel('Activation Similarity (Pythia)')
ax.set_title('Embedding Surprise vs. Activation Overlap')
ax.legend(fontsize=8)

# 4. Polysemantic neuron Jaccard similarity
ax = axes[1, 1]
if polysemantic_neurons:
    pairs_labels = [f"{pn['pair'][0][:6]}-{pn['pair'][1][:6]}\nL{pn['layer']}" for pn in polysemantic_neurons[:15]]
    jaccards = [pn['jaccard'] for pn in polysemantic_neurons[:15]]
    ax.barh(range(len(pairs_labels)), jaccards, color='#9b59b6', alpha=0.8)
    ax.set_yticks(range(len(pairs_labels)))
    ax.set_yticklabels(pairs_labels, fontsize=7)
    ax.set_xlabel('Jaccard Index (Shared Top-5% Neurons)')
    ax.set_title('Polysemantic Neuron Overlap')
    ax.invert_yaxis()
else:
    ax.text(0.5, 0.5, 'No polysemantic neurons found', ha='center', va='center')

plt.tight_layout()
plt.savefig('results/plots/experiment3_overview.png', dpi=150, bbox_inches='tight')
print("Saved results/plots/experiment3_overview.png")
