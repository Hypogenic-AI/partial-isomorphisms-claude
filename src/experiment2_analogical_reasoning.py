"""
Experiment 2: LLM Analogical Reasoning Probing
Test whether LLMs generate and leverage unexpected analogical mappings when reasoning.

Two sub-experiments:
A) Free analogy generation: Ask LLM to find analogies between surprising pairs → do models recognize the mapping?
B) Reasoning benefit: Test whether providing unexpected analogies improves reasoning on transfer tasks.
"""

import os
import json
import random
import time
import numpy as np
from openai import OpenAI

random.seed(42)
np.random.seed(42)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def call_gpt(messages, model="gpt-4.1", temperature=0.3, max_tokens=1000):
    """Call GPT API with retry logic."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"

# Load top surprising pairs from Experiment 1
with open("results/experiment1_results.json") as f:
    exp1 = json.load(f)

top_surprising = exp1["top_surprising_cross_domain"][:25]
# Also get some low-surprise pairs as controls
all_cross = exp1["all_cross_domain"]
median_idx = len(all_cross) // 2
control_pairs = all_cross[median_idx:median_idx+15]  # Middle-surprise pairs
random.shuffle(control_pairs)
control_pairs = control_pairs[:10]

###############################################################################
# Sub-experiment A: Free Analogy Generation
###############################################################################
print("=" * 60)
print("Experiment 2A: Free Analogy Generation")
print("=" * 60)

analogy_results = []

for pair in top_surprising[:20]:
    w1, d1 = pair["word1"], pair["domain1"]
    w2, d2 = pair["word2"], pair["domain2"]

    prompt = f"""Consider the concepts "{w1}" (from the domain of {d1}) and "{w2}" (from the domain of {d2}).

These might seem unrelated, but find structural similarities between them. What properties, roles, or functions do they share? Think step by step about:
1. What role does each play in its domain?
2. What structural properties do they share?
3. Rate the strength of the analogy from 1 (no meaningful connection) to 5 (strong structural parallel).

Be specific and concrete. If there truly is no meaningful connection, say so."""

    response = call_gpt([{"role": "user", "content": prompt}])

    # Ask for a numeric rating separately for clean extraction
    rating_prompt = f"""Based on your analysis of the analogy between "{w1}" ({d1}) and "{w2}" ({d2}), give a single number from 1-5 rating the strength of this structural analogy. Reply with ONLY the number."""

    rating = call_gpt([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
        {"role": "user", "content": rating_prompt},
    ], temperature=0)

    try:
        rating_num = int(rating.strip()[0])
    except:
        rating_num = None

    result = {
        "word1": w1, "domain1": d1,
        "word2": w2, "domain2": d2,
        "surprise_score": pair["surprise_score"],
        "model_similarity": pair["model_similarity"],
        "analogy_explanation": response,
        "analogy_rating": rating_num,
    }
    analogy_results.append(result)
    print(f"  {w1:15s} <-> {w2:15s}: rating={rating_num}, surprise={pair['surprise_score']:.3f}")

# Same for control pairs
control_analogy_results = []
for pair in control_pairs:
    w1, d1 = pair["word1"], pair["domain1"]
    w2, d2 = pair["word2"], pair["domain2"]

    prompt = f"""Consider the concepts "{w1}" (from the domain of {d1}) and "{w2}" (from the domain of {d2}).

These might seem unrelated, but find structural similarities between them. What properties, roles, or functions do they share? Think step by step about:
1. What role does each play in its domain?
2. What structural properties do they share?
3. Rate the strength of the analogy from 1 (no meaningful connection) to 5 (strong structural parallel).

Be specific and concrete. If there truly is no meaningful connection, say so."""

    response = call_gpt([{"role": "user", "content": prompt}])

    rating_prompt = f"""Based on your analysis of the analogy between "{w1}" ({d1}) and "{w2}" ({d2}), give a single number from 1-5 rating the strength of this structural analogy. Reply with ONLY the number."""

    rating = call_gpt([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
        {"role": "user", "content": rating_prompt},
    ], temperature=0)

    try:
        rating_num = int(rating.strip()[0])
    except:
        rating_num = None

    control_analogy_results.append({
        "word1": w1, "domain1": d1,
        "word2": w2, "domain2": d2,
        "surprise_score": pair["surprise_score"],
        "model_similarity": pair["model_similarity"],
        "analogy_explanation": response,
        "analogy_rating": rating_num,
    })
    print(f"  [control] {w1:15s} <-> {w2:15s}: rating={rating_num}, surprise={pair['surprise_score']:.3f}")

###############################################################################
# Sub-experiment B: Reasoning Transfer Test
###############################################################################
print("\n" + "=" * 60)
print("Experiment 2B: Reasoning Transfer Test")
print("=" * 60)

# Design reasoning tasks where analogy could help
# For each surprising pair, create a question about one concept that could benefit
# from thinking about the other concept

TRANSFER_TASKS = [
    {
        "pair": ("foundation", "anchor"),
        "question": "A startup is struggling because its core value proposition keeps shifting. What concept from building architecture might help diagnose this problem?",
        "expected_insight": "foundation - the company needs a stable foundation/anchor before building up",
        "conventional_hint": "Think about what a building needs before you add floors.",
        "unexpected_hint": "Think about what an anchor does for a ship - it provides a fixed reference point against drifting currents.",
    },
    {
        "pair": ("ingredient", "enzyme"),
        "question": "A software team has many talented engineers but projects keep failing. What might be missing?",
        "expected_insight": "catalytic element - like an enzyme to ingredients, or a process/PM to enable combination",
        "conventional_hint": "Think about what makes a team work, like a machine with parts.",
        "unexpected_hint": "Think about cooking: having great ingredients isn't enough - you need an enzyme-like catalyst that transforms raw components into something functional.",
    },
    {
        "pair": ("stack", "layer"),
        "question": "How would you explain the concept of progressive disclosure in UI design to someone who knows cooking but not tech?",
        "expected_insight": "layers in a dish - each layer is revealed as you eat/interact",
        "conventional_hint": "Think about peeling an onion.",
        "unexpected_hint": "Think about a layered dish like lasagna or a trifle - you experience each layer in sequence, and each one builds on what came before, just like a computing stack.",
    },
    {
        "pair": ("shield", "defense"),
        "question": "A company's PR team is constantly reacting to crises. How should they restructure?",
        "expected_insight": "proactive defense vs reactive shielding",
        "conventional_hint": "Think about how sports teams organize their defense.",
        "unexpected_hint": "Think about the difference between a shield (blocks what comes at you) and a strategic defense (anticipates and positions before the attack). Which does your PR team need?",
    },
    {
        "pair": ("recipe", "strategy"),
        "question": "Why do some detailed project plans fail while rough outlines succeed?",
        "expected_insight": "over-specification vs adaptive framework",
        "conventional_hint": "Think about military strategy - detailed plans rarely survive contact with the enemy.",
        "unexpected_hint": "Think about the difference between a recipe (exact steps, exact amounts) and a cooking strategy (understanding flavor principles). The recipe fails when ingredients change; the strategy adapts.",
    },
    {
        "pair": ("heart", "beat"),
        "question": "A remote team has lost its sense of cohesion. What's missing?",
        "expected_insight": "rhythm/pulse - regular cadence of interaction",
        "conventional_hint": "Think about what holds a family together.",
        "unexpected_hint": "Think about music: a band stays together through a shared beat. Your team needs a heartbeat - a regular rhythm of connection that everyone can feel and synchronize to.",
    },
    {
        "pair": ("eye", "window"),
        "question": "How should a data dashboard be designed for executives?",
        "expected_insight": "selective visibility - showing what matters, filtering the rest",
        "conventional_hint": "Think about a car dashboard - show only critical gauges.",
        "unexpected_hint": "Think about the eye as a window: it doesn't show everything, it focuses. And a window frames a specific view of the world. Your dashboard should be both an eye (actively focusing) and a window (framing a specific perspective).",
    },
    {
        "pair": ("head", "roof"),
        "question": "A department has grown too large to manage. How should it be restructured?",
        "expected_insight": "the covering/containing element at the top needs to match what's beneath",
        "conventional_hint": "Think about how a tree grows branches.",
        "unexpected_hint": "Think about architecture: when a building expands, the roof must expand too, or you need multiple roofs. Similarly, the 'head' of a department can only cover so much - you need multiple heads/roofs for multiple structures.",
    },
]

transfer_results = []

for task in TRANSFER_TASKS:
    results_for_task = {}

    # Condition 1: No hint
    response_no_hint = call_gpt([{"role": "user", "content": task["question"]}])

    # Condition 2: Conventional hint
    response_conv = call_gpt([{"role": "user", "content": task["question"] + "\n\nHint: " + task["conventional_hint"]}])

    # Condition 3: Unexpected analogy hint
    response_unexpected = call_gpt([{"role": "user", "content": task["question"] + "\n\nHint: " + task["unexpected_hint"]}])

    # Rate each response for insight quality
    for condition, response in [("no_hint", response_no_hint), ("conventional", response_conv), ("unexpected", response_unexpected)]:
        rating_prompt = f"""Rate the following response to this question for insight quality (1-5 scale):
- 1: Generic/superficial advice
- 2: Reasonable but common advice
- 3: Good advice with some novel framing
- 4: Insightful advice with creative structural thinking
- 5: Deeply insightful advice that reframes the problem productively

Question: {task["question"]}
Response: {response}

Reply with ONLY a single number 1-5."""

        rating = call_gpt([{"role": "user", "content": rating_prompt}], temperature=0)
        try:
            rating_num = int(rating.strip()[0])
        except:
            rating_num = None
        results_for_task[condition] = {"response": response, "rating": rating_num}

    transfer_results.append({
        "pair": task["pair"],
        "question": task["question"],
        "no_hint": results_for_task["no_hint"],
        "conventional": results_for_task["conventional"],
        "unexpected": results_for_task["unexpected"],
    })
    print(f"  {str(task['pair']):40s} no_hint={results_for_task['no_hint']['rating']}  conv={results_for_task['conventional']['rating']}  unexpected={results_for_task['unexpected']['rating']}")

###############################################################################
# Sub-experiment C: Spontaneous Analogy Discovery
###############################################################################
print("\n" + "=" * 60)
print("Experiment 2C: Spontaneous Analogy Discovery")
print("=" * 60)

# Ask the model to freely generate unexpected analogies
discovery_prompt = """I'm studying how different domains share hidden structural similarities. For each of the following domain pairs, generate the most surprising but genuinely insightful analogy you can find - a mapping between concepts that most people wouldn't think of but which reveals real structural parallels.

For each pair, provide:
1. The specific concepts being mapped
2. The structural property they share
3. Why this mapping is non-obvious
4. A "surprise rating" (1-5) for how unexpected this mapping is

Domain pairs:
1. Cooking ↔ Computer Science
2. Music ↔ Finance
3. Biology ↔ Architecture
4. Navigation ↔ Social dynamics
5. Sports ↔ Geography
6. Warfare ↔ Cooking
7. Body ↔ Computing
8. Music ↔ Biology
9. Finance ↔ Navigation
10. Architecture ↔ Sports

Be creative but rigorous - the analogy must hold structurally, not just metaphorically."""

discovery_response = call_gpt([{"role": "user", "content": discovery_prompt}], max_tokens=3000)

# Also ask for the model's OWN most surprising analogies
own_prompt = """Now forget those domain pairs. What are the 10 most surprising structural analogies you can think of between ANY two concepts from different domains? These should be mappings that:
- Are genuinely unexpected (not common metaphors like "time is money")
- Have real structural validity (not just surface similarity)
- Could help someone think about one domain using insights from another

For each, explain the structural mapping briefly and rate its surprise value (1-5)."""

own_response = call_gpt([{"role": "user", "content": own_prompt}], max_tokens=3000)

###############################################################################
# Save all results
###############################################################################
output = {
    "experiment_2a_analogy_recognition": {
        "surprising_pairs": analogy_results,
        "control_pairs": control_analogy_results,
        "statistics": {
            "surprising_mean_rating": np.mean([r["analogy_rating"] for r in analogy_results if r["analogy_rating"]]),
            "control_mean_rating": np.mean([r["analogy_rating"] for r in control_analogy_results if r["analogy_rating"]]),
        }
    },
    "experiment_2b_reasoning_transfer": {
        "tasks": transfer_results,
        "statistics": {
            "no_hint_mean": np.mean([t["no_hint"]["rating"] for t in transfer_results if t["no_hint"]["rating"]]),
            "conventional_mean": np.mean([t["conventional"]["rating"] for t in transfer_results if t["conventional"]["rating"]]),
            "unexpected_mean": np.mean([t["unexpected"]["rating"] for t in transfer_results if t["unexpected"]["rating"]]),
        }
    },
    "experiment_2c_spontaneous_discovery": {
        "prompted_analogies": discovery_response,
        "free_analogies": own_response,
    }
}

with open("results/experiment2_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("\n\nResults saved to results/experiment2_results.json")

# Print summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
stats_2a = output["experiment_2a_analogy_recognition"]["statistics"]
stats_2b = output["experiment_2b_reasoning_transfer"]["statistics"]
print(f"\n2A - Analogy Recognition:")
print(f"  Surprising pairs mean rating: {stats_2a['surprising_mean_rating']:.2f}")
print(f"  Control pairs mean rating:    {stats_2a['control_mean_rating']:.2f}")
print(f"\n2B - Reasoning Transfer:")
print(f"  No hint mean:        {stats_2b['no_hint_mean']:.2f}")
print(f"  Conventional mean:   {stats_2b['conventional_mean']:.2f}")
print(f"  Unexpected mean:     {stats_2b['unexpected_mean']:.2f}")
