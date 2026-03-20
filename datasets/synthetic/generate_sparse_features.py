"""
Synthetic Sparse Feature Dataset Generator
===========================================
Replicates the data-generating process from:
  "Toy Models of Superposition" (Elhage et al., 2022)
  https://transformer-circuits.pub/2022/toy_model/index.html

The core idea: features are sparse and independently distributed.
A data point x in R^n has each coordinate independently set to a
nonzero value drawn from Uniform(0, 1] with probability (1 - sparsity),
and to 0.0 with probability sparsity.

This generator also supports:
- Correlated feature groups (for studying partial isomorphisms between
  groups of features compressed into the same subspace)
- Importance-weighted features (some features matter more than others)
- Multiple sparsity regimes in a single dataset

Usage
-----
    python generate_sparse_features.py               # uses defaults
    python generate_sparse_features.py --n_samples 200000 --n_features 512 --sparsity 0.99

The output is saved under datasets/synthetic/ as .npz archives.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def make_feature_matrix(
    n_samples: int,
    n_features: int,
    sparsity: float,
    feature_importance: np.ndarray | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Return an (n_samples, n_features) float32 array of sparse features.

    Parameters
    ----------
    n_samples:          Number of data points to generate.
    n_features:         Dimensionality of feature space.
    sparsity:           Probability that any given feature is 0. Must be in
                        [0, 1).  Typical values: 0.9, 0.99, 0.999.
    feature_importance: Optional (n_features,) array of positive importance
                        weights. When provided, feature values are multiplied
                        by these weights, mirroring the importance-weighted
                        loss in the TMS paper.
    seed:               Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features), dtype float32.
    """
    rng = np.random.default_rng(seed)

    # Draw magnitudes uniformly from (0, 1]
    magnitudes = rng.uniform(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)

    # Zero out features according to sparsity
    mask = rng.random(size=(n_samples, n_features)) < (1.0 - sparsity)
    X = magnitudes * mask.astype(np.float32)

    if feature_importance is not None:
        importance = np.asarray(feature_importance, dtype=np.float32)
        if importance.shape != (n_features,):
            raise ValueError(
                f"feature_importance must have shape ({n_features},), "
                f"got {importance.shape}"
            )
        X = X * importance[np.newaxis, :]

    return X


def make_correlated_groups(
    n_samples: int,
    n_features: int,
    n_groups: int,
    within_group_corr: float = 0.8,
    sparsity: float = 0.99,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate features with intra-group correlations.

    Returns (X, group_labels) where group_labels[i] is the group id for
    feature i. Features within a group tend to co-activate, which is
    a prerequisite for the network to learn to compress them together.

    This configuration is particularly useful for studying whether a network
    learns a *shared* representation for each group (expected) or an
    *unexpected* partial isomorphism where features from different groups
    are mapped to the same direction.
    """
    rng = np.random.default_rng(seed)

    features_per_group = n_features // n_groups
    group_labels = np.repeat(np.arange(n_groups), features_per_group)
    # Handle remainder
    remainder = n_features - features_per_group * n_groups
    if remainder:
        group_labels = np.concatenate([
            group_labels,
            np.full(remainder, n_groups - 1, dtype=int),
        ])

    X = np.zeros((n_samples, n_features), dtype=np.float32)

    for g in range(n_groups):
        g_mask = group_labels == g
        g_size = g_mask.sum()

        # Group-level activation: whether the whole group fires this sample
        group_active = rng.random(n_samples) < (1.0 - sparsity)

        for fi in np.where(g_mask)[0]:
            # Feature is active if group is active AND individual bernoulli fires
            ind_noise = rng.random(n_samples) < within_group_corr
            active = group_active & ind_noise
            X[active, fi] = rng.uniform(0.0, 1.0, size=active.sum()).astype(np.float32)

    return X, group_labels


def geometric_importance(n_features: int, decay: float = 0.7) -> np.ndarray:
    """Return geometrically decaying importance weights.

    Matches the parameterisation used in the TMS paper where feature i has
    importance decay^i, so the first feature matters most.
    """
    return np.array([decay ** i for i in range(n_features)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset configuration presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    # Minimal toy model matching TMS Figure 1
    "tms_tiny": dict(
        n_samples=100_000,
        n_features=5,
        hidden_dims=[2],          # 5 features -> 2D hidden space (superposition)
        sparsity=0.99,
        importance_decay=0.7,
        description=(
            "5 features, 2D hidden space, sparsity=0.99. "
            "Directly matches the running example in the TMS paper."
        ),
    ),
    # Medium scale for probing experiments
    "tms_medium": dict(
        n_samples=500_000,
        n_features=80,
        hidden_dims=[20],
        sparsity=0.99,
        importance_decay=0.9,
        description=(
            "80 features compressed into 20 dims. "
            "Useful for studying which features survive compression."
        ),
    ),
    # Large scale, uniform importance
    "tms_large_uniform": dict(
        n_samples=1_000_000,
        n_features=256,
        hidden_dims=[64],
        sparsity=0.995,
        importance_decay=None,    # uniform importance
        description=(
            "256 uniform-importance features -> 64 dims. "
            "Tests whether compression is qualitatively different without "
            "a pre-ordained feature hierarchy."
        ),
    ),
    # Correlated groups -- partial isomorphism target
    "correlated_groups": dict(
        n_samples=500_000,
        n_features=100,
        n_groups=10,
        within_group_corr=0.8,
        sparsity=0.99,
        description=(
            "10 groups of 10 correlated features. "
            "Designed to probe whether inter-group partial isomorphisms "
            "emerge when groups share structural similarity."
        ),
    ),
    # Varying sparsity sweep (for phase-transition experiments)
    "sparsity_sweep": dict(
        n_samples=200_000,
        n_features=40,
        hidden_dims=[10],
        sparsities=[0.5, 0.7, 0.9, 0.95, 0.99, 0.999],
        importance_decay=0.9,
        description=(
            "Same 40-feature / 10-dim setup at 6 sparsity levels. "
            "Useful for charting the superposition phase transition."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_dataset(
    out_dir: Path,
    name: str,
    X: np.ndarray,
    metadata: dict,
    group_labels: np.ndarray | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"X": X}
    if group_labels is not None:
        arrays["group_labels"] = group_labels
    path = out_dir / f"{name}.npz"
    np.savez_compressed(path, **arrays)

    meta_path = out_dir / f"{name}_meta.json"
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved {path}  shape={X.shape}  size={path.stat().st_size/1e6:.1f} MB")
    return path


def load_dataset(path: str | Path) -> tuple[np.ndarray, dict, np.ndarray | None]:
    """Load a dataset saved by save_dataset.

    Returns (X, metadata, group_labels_or_None).
    """
    path = Path(path)
    data = np.load(path)
    X = data["X"]
    group_labels = data["group_labels"] if "group_labels" in data else None

    meta_path = path.with_name(path.stem + "_meta.json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return X, metadata, group_labels


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def generate_all_presets(out_dir: Path, seed: int = 42) -> None:
    print(f"\nGenerating all preset datasets -> {out_dir}\n")
    start = time.time()

    for preset_name, cfg in PRESETS.items():
        print(f"[{preset_name}]  {cfg['description']}")

        if preset_name == "correlated_groups":
            X, grp = make_correlated_groups(
                n_samples=cfg["n_samples"],
                n_features=cfg["n_features"],
                n_groups=cfg["n_groups"],
                within_group_corr=cfg["within_group_corr"],
                sparsity=cfg["sparsity"],
                seed=seed,
            )
            metadata = {k: v for k, v in cfg.items() if isinstance(v, (str, int, float))}
            save_dataset(out_dir, preset_name, X, metadata, grp)

        elif preset_name == "sparsity_sweep":
            importance = (
                geometric_importance(cfg["n_features"], cfg["importance_decay"])
                if cfg.get("importance_decay") else None
            )
            for s in cfg["sparsities"]:
                X = make_feature_matrix(
                    n_samples=cfg["n_samples"],
                    n_features=cfg["n_features"],
                    sparsity=s,
                    feature_importance=importance,
                    seed=seed,
                )
                name = f"{preset_name}_s{str(s).replace('.','p')}"
                meta = {
                    "preset": preset_name,
                    "sparsity": s,
                    "n_samples": cfg["n_samples"],
                    "n_features": cfg["n_features"],
                }
                save_dataset(out_dir, name, X, meta)

        else:
            importance = (
                geometric_importance(cfg["n_features"], cfg["importance_decay"])
                if cfg.get("importance_decay") else None
            )
            X = make_feature_matrix(
                n_samples=cfg["n_samples"],
                n_features=cfg["n_features"],
                sparsity=cfg["sparsity"],
                feature_importance=importance,
                seed=seed,
            )
            metadata = {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, list))}
            save_dataset(out_dir, preset_name, X, metadata)

        print()

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic sparse feature datasets (TMS-style)"
    )
    p.add_argument("--preset", choices=list(PRESETS.keys()) + ["all"],
                   default="all",
                   help="Which preset to generate (default: all)")
    p.add_argument("--out_dir", type=Path,
                   default=Path(__file__).parent,
                   help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    # Quick custom generation flags
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--n_features", type=int, default=None)
    p.add_argument("--sparsity", type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.n_samples and args.n_features and args.sparsity:
        # Custom one-off generation
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        X = make_feature_matrix(
            n_samples=args.n_samples,
            n_features=args.n_features,
            sparsity=args.sparsity,
            seed=args.seed,
        )
        name = f"custom_n{args.n_features}_s{str(args.sparsity).replace('.','p')}"
        meta = {
            "n_samples": args.n_samples,
            "n_features": args.n_features,
            "sparsity": args.sparsity,
            "seed": args.seed,
        }
        save_dataset(out_dir, name, X, meta)
    elif args.preset == "all":
        generate_all_presets(args.out_dir, seed=args.seed)
    else:
        cfg = PRESETS[args.preset]
        print(f"Generating preset '{args.preset}': {cfg['description']}")
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.preset == "correlated_groups":
            X, grp = make_correlated_groups(
                n_samples=cfg["n_samples"],
                n_features=cfg["n_features"],
                n_groups=cfg["n_groups"],
                within_group_corr=cfg["within_group_corr"],
                sparsity=cfg["sparsity"],
                seed=args.seed,
            )
            meta = {k: v for k, v in cfg.items() if isinstance(v, (str, int, float))}
            save_dataset(out_dir, args.preset, X, meta, grp)
        elif args.preset == "sparsity_sweep":
            importance = (
                geometric_importance(cfg["n_features"], cfg["importance_decay"])
                if cfg.get("importance_decay") else None
            )
            for s in cfg["sparsities"]:
                X = make_feature_matrix(
                    cfg["n_samples"], cfg["n_features"], s,
                    feature_importance=importance, seed=args.seed,
                )
                name = f"{args.preset}_s{str(s).replace('.','p')}"
                save_dataset(out_dir, name, X, {"sparsity": s})
        else:
            importance = (
                geometric_importance(cfg["n_features"], cfg["importance_decay"])
                if cfg.get("importance_decay") else None
            )
            X = make_feature_matrix(
                cfg["n_samples"], cfg["n_features"], cfg["sparsity"],
                feature_importance=importance, seed=args.seed,
            )
            meta = {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, list))}
            save_dataset(out_dir, args.preset, X, meta)
