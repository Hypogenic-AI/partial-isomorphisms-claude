"""
Experiment 1: Embedding Surprise Discovery
Find concept pairs that are close in LLM embedding space but distant in human similarity judgment.
"""

import os
import json
import random
import numpy as np
from itertools import combinations
from openai import OpenAI
from nltk.corpus import wordnet as wn
import time

random.seed(42)
np.random.seed(42)

# Concept domains with representative concepts
CONCEPT_DOMAINS = {
    "body": ["head", "heart", "hand", "eye", "spine", "lung", "skin", "bone", "brain", "blood"],
    "architecture": ["foundation", "pillar", "wall", "roof", "window", "door", "beam", "floor", "ceiling", "arch"],
    "computing": ["memory", "thread", "cache", "kernel", "port", "shell", "stack", "pipe", "bus", "bridge"],
    "cooking": ["recipe", "ingredient", "simmer", "blend", "season", "crust", "layer", "garnish", "marinate", "reduce"],
    "music": ["harmony", "rhythm", "note", "chord", "tempo", "pitch", "tone", "beat", "melody", "scale"],
    "warfare": ["strategy", "siege", "flank", "shield", "retreat", "advance", "ambush", "fortify", "rally", "scout"],
    "biology": ["cell", "membrane", "nucleus", "enzyme", "gene", "protein", "tissue", "organ", "parasite", "symbiosis"],
    "geography": ["basin", "ridge", "delta", "plateau", "canyon", "tributary", "watershed", "erosion", "sediment", "estuary"],
    "finance": ["portfolio", "hedge", "leverage", "yield", "dividend", "margin", "bond", "equity", "liquidity", "inflation"],
    "sports": ["offense", "defense", "coach", "draft", "penalty", "assist", "formation", "endurance", "sprint", "tackle"],
    "social": ["hierarchy", "network", "trust", "reputation", "alliance", "conflict", "negotiation", "consensus", "influence", "role"],
    "navigation": ["compass", "chart", "anchor", "drift", "bearing", "course", "harbor", "current", "tide", "rudder"],
}

def get_all_concepts():
    """Return flat list of (concept, domain) tuples."""
    concepts = []
    for domain, words in CONCEPT_DOMAINS.items():
        for w in words:
            concepts.append((w, domain))
    return concepts

def get_embeddings_batch(client, texts, model="text-embedding-3-large"):
    """Get embeddings for a batch of texts from OpenAI API."""
    # API limit is 2048 inputs per request
    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(input=batch, model=model)
        for item in response.data:
            embeddings.append(item.embedding)
        if i + batch_size < len(texts):
            time.sleep(0.5)  # Rate limit courtesy
    return np.array(embeddings)

def cosine_similarity_matrix(embeddings):
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return normalized @ normalized.T

def wordnet_similarity(word1, word2):
    """Compute WordNet path similarity between two words. Returns 0-1."""
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    if not synsets1 or not synsets2:
        return None
    max_sim = 0
    for s1 in synsets1[:3]:  # Limit to top 3 senses
        for s2 in synsets2[:3]:
            sim = s1.path_similarity(s2)
            if sim is not None and sim > max_sim:
                max_sim = sim
    return max_sim if max_sim > 0 else None

def compute_surprise_scores(model_sim_matrix, concepts):
    """Compute surprise scores for all concept pairs."""
    n = len(concepts)
    results = []

    for i in range(n):
        for j in range(i+1, n):
            word1, domain1 = concepts[i]
            word2, domain2 = concepts[j]

            model_sim = model_sim_matrix[i, j]
            human_sim = wordnet_similarity(word1, word2)

            if human_sim is None:
                continue

            same_domain = domain1 == domain2

            results.append({
                "word1": word1,
                "word2": word2,
                "domain1": domain1,
                "domain2": domain2,
                "same_domain": same_domain,
                "model_similarity": float(model_sim),
                "human_similarity": float(human_sim),
                "surprise_score": float(model_sim - human_sim),
            })

    return results

def main():
    print("=" * 60)
    print("Experiment 1: Embedding Surprise Discovery")
    print("=" * 60)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    concepts = get_all_concepts()
    concept_words = [c[0] for c in concepts]
    print(f"\nTotal concepts: {len(concepts)} across {len(CONCEPT_DOMAINS)} domains")

    # Step 1: Get embeddings
    print("\nStep 1: Computing embeddings...")
    embeddings = get_embeddings_batch(client, concept_words)
    print(f"  Embedding shape: {embeddings.shape}")

    # Step 2: Compute model similarity matrix
    print("\nStep 2: Computing model similarity matrix...")
    model_sim_matrix = cosine_similarity_matrix(embeddings)

    # Step 3: Compute surprise scores
    print("\nStep 3: Computing surprise scores with WordNet baselines...")
    results = compute_surprise_scores(model_sim_matrix, concepts)
    print(f"  Total pairs with valid human similarity: {len(results)}")

    # Step 4: Filter to cross-domain pairs and sort by surprise
    cross_domain = [r for r in results if not r["same_domain"]]
    cross_domain.sort(key=lambda x: x["surprise_score"], reverse=True)

    print(f"\n  Cross-domain pairs: {len(cross_domain)}")

    # Top surprising cross-domain pairs
    print("\n" + "=" * 60)
    print("TOP 30 MOST SURPRISING CROSS-DOMAIN PAIRS")
    print("(High model similarity, low human similarity)")
    print("=" * 60)
    for i, r in enumerate(cross_domain[:30]):
        print(f"  {i+1:2d}. {r['word1']:15s} ({r['domain1']:12s}) <-> {r['word2']:15s} ({r['domain2']:12s}) "
              f"model={r['model_similarity']:.3f}  human={r['human_similarity']:.3f}  surprise={r['surprise_score']:.3f}")

    # Also show the LEAST surprising (high human sim, high model sim - expected pairs)
    same_domain = [r for r in results if r["same_domain"]]
    same_domain.sort(key=lambda x: x["model_similarity"], reverse=True)

    print("\n" + "=" * 60)
    print("CONTROL: TOP 10 MOST SIMILAR SAME-DOMAIN PAIRS (expected)")
    print("=" * 60)
    for i, r in enumerate(same_domain[:10]):
        print(f"  {i+1:2d}. {r['word1']:15s} <-> {r['word2']:15s} ({r['domain1']:12s}) "
              f"model={r['model_similarity']:.3f}  human={r['human_similarity']:.3f}")

    # Compute statistics
    surprise_scores = [r["surprise_score"] for r in cross_domain]
    print(f"\n\nSurprise score statistics (cross-domain):")
    print(f"  Mean: {np.mean(surprise_scores):.4f}")
    print(f"  Std:  {np.std(surprise_scores):.4f}")
    print(f"  Max:  {np.max(surprise_scores):.4f}")
    print(f"  Min:  {np.min(surprise_scores):.4f}")

    # Identify pairs >2 std above mean as "significantly surprising"
    threshold = np.mean(surprise_scores) + 2 * np.std(surprise_scores)
    significant = [r for r in cross_domain if r["surprise_score"] > threshold]
    print(f"\n  Pairs >2σ above mean (threshold={threshold:.4f}): {len(significant)}")

    # Save results
    output = {
        "config": {
            "n_concepts": len(concepts),
            "n_domains": len(CONCEPT_DOMAINS),
            "embedding_model": "text-embedding-3-large",
            "human_similarity": "wordnet_path_similarity",
        },
        "statistics": {
            "total_pairs": len(results),
            "cross_domain_pairs": len(cross_domain),
            "surprise_mean": float(np.mean(surprise_scores)),
            "surprise_std": float(np.std(surprise_scores)),
            "significant_threshold": float(threshold),
            "n_significant": len(significant),
        },
        "top_surprising_cross_domain": cross_domain[:50],
        "top_similar_same_domain": same_domain[:20],
        "all_cross_domain": cross_domain,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/experiment1_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Save embeddings for later use
    np.save("results/concept_embeddings.npy", embeddings)
    with open("results/concept_list.json", "w") as f:
        json.dump(concepts, f)

    print("\nResults saved to results/experiment1_results.json")
    return output

if __name__ == "__main__":
    main()
