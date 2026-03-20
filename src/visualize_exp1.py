"""Visualize Experiment 1 results."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

with open("results/experiment1_results.json") as f:
    data = json.load(f)

cross = data["all_cross_domain"]
top = data["top_surprising_cross_domain"][:30]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Scatter: model sim vs human sim
ax = axes[0, 0]
model_sims = [r["model_similarity"] for r in cross]
human_sims = [r["human_similarity"] for r in cross]
surprise = [r["surprise_score"] for r in cross]
sc = ax.scatter(human_sims, model_sims, c=surprise, cmap='RdYlBu_r', alpha=0.3, s=8)
plt.colorbar(sc, ax=ax, label='Surprise Score')
# Mark top 10
for i, r in enumerate(top[:10]):
    ax.annotate(f"{r['word1']}-{r['word2']}",
                (r["human_similarity"], r["model_similarity"]),
                fontsize=7, alpha=0.8)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
ax.set_xlabel('Human Similarity (WordNet)')
ax.set_ylabel('Model Similarity (Embedding)')
ax.set_title('Model vs. Human Similarity\n(Cross-Domain Pairs)')
ax.legend(fontsize=8)

# 2. Surprise score distribution
ax = axes[0, 1]
ax.hist(surprise, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
threshold = data["statistics"]["significant_threshold"]
ax.axvline(threshold, color='red', linestyle='--', label=f'2σ threshold ({threshold:.3f})')
ax.axvline(np.mean(surprise), color='orange', linestyle='-', label=f'Mean ({np.mean(surprise):.3f})')
ax.set_xlabel('Surprise Score (Model Sim - Human Sim)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Surprise Scores')
ax.legend(fontsize=8)

# 3. Top 20 surprising pairs - horizontal bar chart
ax = axes[1, 0]
labels = [f"{r['word1']} ↔ {r['word2']}" for r in top[:20]]
scores = [r["surprise_score"] for r in top[:20]]
colors = ['#e74c3c' if s > 0.4 else '#f39c12' if s > 0.35 else '#3498db' for s in scores]
bars = ax.barh(range(len(labels)), scores, color=colors, edgecolor='white')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Surprise Score')
ax.set_title('Top 20 Most Surprising Cross-Domain Pairs')
ax.invert_yaxis()

# 4. Domain heatmap: average surprise between domain pairs
ax = axes[1, 1]
domains = sorted(set(r["domain1"] for r in cross) | set(r["domain2"] for r in cross))
n_dom = len(domains)
dom_matrix = np.zeros((n_dom, n_dom))
dom_count = np.zeros((n_dom, n_dom))
for r in cross:
    i = domains.index(r["domain1"])
    j = domains.index(r["domain2"])
    dom_matrix[i, j] += r["surprise_score"]
    dom_matrix[j, i] += r["surprise_score"]
    dom_count[i, j] += 1
    dom_count[j, i] += 1
dom_count[dom_count == 0] = 1
dom_matrix /= dom_count
sns.heatmap(dom_matrix, xticklabels=domains, yticklabels=domains,
            cmap='RdYlBu_r', ax=ax, annot=True, fmt='.2f', annot_kws={'size': 6})
ax.set_title('Average Surprise Score\nBetween Domain Pairs')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
plt.setp(ax.get_yticklabels(), fontsize=7)

plt.tight_layout()
plt.savefig('results/plots/experiment1_overview.png', dpi=150, bbox_inches='tight')
print("Saved results/plots/experiment1_overview.png")
