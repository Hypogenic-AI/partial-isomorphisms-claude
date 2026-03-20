"""
Pythia Model Probing Guide & Dataset Access
============================================
This module documents how to access Pythia language models and The Pile
for sparse probing experiments, following:

  "Finding Neurons in a Haystack: Case Studies with Sparse Probing"
   (Gurnee et al., 2023)   https://arxiv.org/abs/2305.01610

The paper probes 100+ features in Pythia models (70M – 12B parameters)
trained on The Pile. Features probed include:
  - Part-of-speech tags (noun, verb, adjective, ...)
  - Named entity types (PERSON, ORG, GPE, ...)
  - Position features (start-of-sentence, etc.)
  - Semantic categories (numbers, months, countries, ...)
  - Casing (ALL_CAPS, Title_Case, camelCase, ...)

Relation to partial isomorphism hypothesis
-------------------------------------------
The probing setup lets us ask:
  "Do different layers, models, or training checkpoints develop
   *equivalent* or *isomorphic* feature representations, or do they
   use counterintuitive cross-feature mappings?"

E.g. do neurons tuned for PERSON entities in Pythia-70M map onto the
same direction as PERSON neurons in Pythia-160M, or does the mapping
'twist' in a non-trivial way (partial isomorphism)?

Installation
------------
    pip install transformers datasets torch accelerate

Optional (for faster tokenisation):
    pip install tokenizers

Usage
-----
Run this file directly for a self-check:
    python pythia_probing_guide.py

Or import individual helpers:
    from pythia_probing_guide import load_pythia, extract_activations
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Available Pythia model sizes
# ---------------------------------------------------------------------------

PYTHIA_MODELS = {
    "70m":    "EleutherAI/pythia-70m",
    "70m-d":  "EleutherAI/pythia-70m-deduped",
    "160m":   "EleutherAI/pythia-160m",
    "160m-d": "EleutherAI/pythia-160m-deduped",
    "410m":   "EleutherAI/pythia-410m",
    "410m-d": "EleutherAI/pythia-410m-deduped",
    "1b":     "EleutherAI/pythia-1b",
    "1.4b":   "EleutherAI/pythia-1.4b",
    "2.8b":   "EleutherAI/pythia-2.8b",
    "6.9b":   "EleutherAI/pythia-6.9b",
    "12b":    "EleutherAI/pythia-12b",
}

# Training checkpoints available for each model (step numbers)
# Pythia models have checkpoints at steps: 0, 1, 2, 4, 8, 16, 32, 64, 128,
# 256, 512, 1000, 2000, ..., 143000 (143 checkpoints total).
PYTHIA_CHECKPOINT_STEPS = (
    [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    + list(range(1000, 144000, 1000))
)

# ---------------------------------------------------------------------------
# Feature categories probed by Gurnee et al.
# ---------------------------------------------------------------------------

PROBING_FEATURES = {
    "part_of_speech": [
        "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CONJ",
        "PROPN", "NUM", "INTJ", "PART", "PUNCT", "SYM", "X",
    ],
    "named_entity": [
        "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY",
        "PERCENT", "QUANTITY", "ORDINAL", "CARDINAL", "EVENT",
        "FAC", "LANGUAGE", "LAW", "NORP", "PRODUCT", "WORK_OF_ART",
    ],
    "morphological": [
        "is_plural", "is_singular", "is_past_tense", "is_present_tense",
        "is_future", "is_gerund", "is_participle", "is_comparative",
        "is_superlative",
    ],
    "orthographic": [
        "all_caps", "title_case", "all_lower", "camel_case",
        "has_digit", "has_hyphen", "has_apostrophe",
        "starts_sentence", "ends_sentence",
    ],
    "semantic": [
        "is_month", "is_weekday", "is_number_word", "is_country",
        "is_language_name", "is_color", "is_body_part",
        "is_food", "is_emotion_word",
    ],
}

ALL_FEATURE_NAMES: list[str] = [
    f"{cat}/{feat}"
    for cat, feats in PROBING_FEATURES.items()
    for feat in feats
]


# ---------------------------------------------------------------------------
# Helper: load a Pythia model and tokeniser
# ---------------------------------------------------------------------------

def load_pythia(
    model_size: str = "70m",
    checkpoint_step: Optional[int] = None,
    device: str = "cpu",
):
    """Load a Pythia model and tokeniser.

    Parameters
    ----------
    model_size:       Key in PYTHIA_MODELS, e.g. "70m", "160m-d".
    checkpoint_step:  Training step to load (None = final checkpoint).
    device:           "cpu", "cuda", or "mps".

    Returns
    -------
    (model, tokenizer) from HuggingFace transformers.

    Example
    -------
    >>> model, tokenizer = load_pythia("70m")
    >>> inputs = tokenizer("Hello world", return_tensors="pt")
    >>> outputs = model(**inputs, output_hidden_states=True)
    >>> hidden = outputs.hidden_states   # tuple of (1, seq_len, hidden_dim)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as e:
        raise ImportError(
            "transformers and torch are required. Install with:\n"
            "  pip install transformers torch"
        ) from e

    repo = PYTHIA_MODELS.get(model_size)
    if repo is None:
        raise ValueError(f"Unknown model size {model_size!r}. "
                         f"Choose from: {list(PYTHIA_MODELS.keys())}")

    revision = f"step{checkpoint_step}" if checkpoint_step is not None else "main"
    print(f"Loading {repo} revision={revision} ...")
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=revision,
        output_hidden_states=True,
    ).to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Helper: extract layer activations for a batch of texts
# ---------------------------------------------------------------------------

def extract_activations(
    texts: list[str],
    model,
    tokenizer,
    layer_indices: Optional[list[int]] = None,
    batch_size: int = 8,
    max_length: int = 128,
    device: str = "cpu",
):
    """Extract hidden-state activations from a Pythia model.

    Parameters
    ----------
    texts:         List of input strings.
    model:         Loaded Pythia model (from load_pythia).
    tokenizer:     Corresponding tokeniser.
    layer_indices: Which layers to extract (None = all layers).
    batch_size:    Number of texts per forward pass.
    max_length:    Truncate inputs to this many tokens.
    device:        Torch device string.

    Returns
    -------
    activations : dict mapping layer_index -> np.ndarray of shape
                  (n_texts, max_length, hidden_dim).
                  Shorter sequences are zero-padded.
    token_ids   : np.ndarray (n_texts, max_length) of token ids.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("torch is required.") from e

    n = len(texts)
    all_layer_acts: dict[int, list] = {}

    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        # hidden_states: tuple of (batch, seq, hidden_dim) for each layer
        hs = out.hidden_states  # len = n_layers + 1 (includes embedding layer)
        layers_to_save = layer_indices if layer_indices else list(range(len(hs)))

        for li in layers_to_save:
            acts = hs[li].cpu().float().numpy()  # (batch, seq, d)
            all_layer_acts.setdefault(li, []).append(acts)

        if (start // batch_size) % 10 == 0:
            print(f"  Processed {min(start + batch_size, n)}/{n} texts ...")

    import numpy as np
    result = {
        li: np.concatenate(batches, axis=0)
        for li, batches in all_layer_acts.items()
    }
    return result


# ---------------------------------------------------------------------------
# Helper: The Pile access
# ---------------------------------------------------------------------------

def load_pile_sample(
    n_docs: int = 1000,
    seed: int = 42,
    streaming: bool = True,
):
    """Load a sample of documents from The Pile via HuggingFace datasets.

    The Pile (Gao et al., 2021) is available on HuggingFace at:
      EleutherAI/pile

    Parameters
    ----------
    n_docs:    Number of documents to load.
    seed:      Random seed for shuffling.
    streaming: If True, streams the dataset instead of downloading the full
               ~800 GB corpus.

    Returns
    -------
    List of text strings.

    Notes
    -----
    The dataset hub entry may require authentication or acceptance of terms.
    If access is restricted, use the Pile-10k subset instead:
      EleutherAI/pile-uncopyrighted  or  NeelNanda/pile-10k
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise ImportError(
            "datasets library required. Install with:\n"
            "  pip install datasets"
        ) from e

    print(f"Loading {n_docs} documents from The Pile (streaming={streaming}) ...")
    try:
        ds = load_dataset(
            "EleutherAI/pile",
            split="train",
            streaming=streaming,
        )
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        docs = [row["text"] for row in ds.take(n_docs)]
    except Exception:
        print("  Falling back to NeelNanda/pile-10k ...")
        ds = load_dataset("NeelNanda/pile-10k", split="train")
        import random
        rng = random.Random(seed)
        idxs = rng.sample(range(len(ds)), min(n_docs, len(ds)))
        docs = [ds[i]["text"] for i in idxs]

    print(f"  Loaded {len(docs)} documents.")
    return docs


# ---------------------------------------------------------------------------
# Helper: probing feature labels from spaCy
# ---------------------------------------------------------------------------

def label_tokens_with_spacy(
    texts: list[str],
    features: Optional[list[str]] = None,
    model_name: str = "en_core_web_sm",
):
    """Use spaCy to produce token-level feature labels for probing.

    Parameters
    ----------
    texts:      Input strings (one per document).
    features:   Subset of feature keys to compute (None = all).
                Format: "category/feature_name", e.g. "part_of_speech/NOUN".
    model_name: spaCy model to load.

    Returns
    -------
    labels : dict mapping feature_key -> list of per-token label arrays.
             Each array has shape (n_tokens,) with dtype bool.

    Notes
    -----
    Requires: pip install spacy && python -m spacy download en_core_web_sm
    """
    try:
        import spacy  # type: ignore
        nlp = spacy.load(model_name)
    except ImportError as e:
        raise ImportError(
            "spaCy is required for token labelling. Install with:\n"
            "  pip install spacy\n"
            "  python -m spacy download en_core_web_sm"
        ) from e

    if features is None:
        features = ALL_FEATURE_NAMES

    labels: dict[str, list] = {f: [] for f in features}

    for text in texts:
        doc = nlp(text)
        toks = list(doc)
        n = len(toks)

        for feat in features:
            cat, fname = feat.split("/", 1)
            arr = _compute_feature(toks, cat, fname)
            labels[feat].append(arr)

    return labels


def _compute_feature(tokens, category: str, feature_name: str):
    """Compute a boolean feature array for a list of spaCy tokens."""
    import numpy as np

    n = len(tokens)
    arr = np.zeros(n, dtype=bool)

    if category == "part_of_speech":
        arr = np.array([t.pos_ == feature_name for t in tokens])
    elif category == "named_entity":
        arr = np.array([t.ent_type_ == feature_name for t in tokens])
    elif category == "morphological":
        for i, t in enumerate(tokens):
            morph = str(t.morph)
            if feature_name == "is_plural":
                arr[i] = "Number=Plur" in morph
            elif feature_name == "is_singular":
                arr[i] = "Number=Sing" in morph
            elif feature_name == "is_past_tense":
                arr[i] = "Tense=Past" in morph
            elif feature_name == "is_present_tense":
                arr[i] = "Tense=Pres" in morph
            elif feature_name == "is_gerund":
                arr[i] = "VerbForm=Ger" in morph
            elif feature_name == "is_participle":
                arr[i] = "VerbForm=Part" in morph
    elif category == "orthographic":
        for i, t in enumerate(tokens):
            txt = t.text
            if feature_name == "all_caps":
                arr[i] = txt.isupper() and len(txt) > 1
            elif feature_name == "title_case":
                arr[i] = txt.istitle()
            elif feature_name == "all_lower":
                arr[i] = txt.islower()
            elif feature_name == "has_digit":
                arr[i] = any(c.isdigit() for c in txt)
            elif feature_name == "has_hyphen":
                arr[i] = "-" in txt
            elif feature_name == "starts_sentence":
                arr[i] = t.is_sent_start or False
    elif category == "semantic":
        MONTHS = {"january","february","march","april","may","june","july",
                  "august","september","october","november","december"}
        WEEKDAYS = {"monday","tuesday","wednesday","thursday","friday",
                    "saturday","sunday"}
        COLORS = {"red","blue","green","yellow","orange","purple","pink",
                  "brown","black","white","gray","grey"}
        for i, t in enumerate(tokens):
            low = t.text.lower()
            if feature_name == "is_month":
                arr[i] = low in MONTHS
            elif feature_name == "is_weekday":
                arr[i] = low in WEEKDAYS
            elif feature_name == "is_color":
                arr[i] = low in COLORS
            elif feature_name == "is_number_word":
                NUMBER_WORDS = {"zero","one","two","three","four","five",
                                "six","seven","eight","nine","ten","eleven",
                                "twelve","hundred","thousand","million","billion"}
                arr[i] = low in NUMBER_WORDS
    return arr


# ---------------------------------------------------------------------------
# Self-check / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Pythia Probing Guide -- environment check\n")
    print(f"Available Pythia model sizes: {list(PYTHIA_MODELS.keys())}")
    print(f"Total probing features defined: {len(ALL_FEATURE_NAMES)}")
    print(f"\nFeature categories:")
    for cat, feats in PROBING_FEATURES.items():
        print(f"  {cat}: {len(feats)} features")

    # Check optional dependencies
    for pkg in ["transformers", "torch", "datasets", "spacy"]:
        try:
            __import__(pkg)
            print(f"  {pkg}: AVAILABLE")
        except ImportError:
            print(f"  {pkg}: NOT INSTALLED  (pip install {pkg})")

    print("\nTo install all dependencies for probing experiments:")
    print("  python3 -m pip install transformers torch datasets spacy accelerate")
    print("  python3 -m spacy download en_core_web_sm")

    print("\nExample workflow (after installation):")
    print("  model, tok = load_pythia('70m')")
    print("  docs = load_pile_sample(n_docs=1000)")
    print("  acts = extract_activations(docs[:10], model, tok, layer_indices=[0, 6, 12])")
    print("  labels = label_tokens_with_spacy(docs[:10])")
    print("  # Fit linear probe: acts[layer_idx][token_positions] -> labels[feature_key]")
