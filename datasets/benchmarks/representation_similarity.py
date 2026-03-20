"""
Representation Similarity Benchmarks
======================================
Standard methods for measuring cross-model representation alignment,
relevant for detecting and quantifying partial isomorphisms.

Methods implemented
-------------------
1. CKA (Centered Kernel Alignment)      -- Kornblith et al. (2019)
2. PWCCA (Projection-Weighted CCA)      -- Morcos et al. (2018)
3. RSA (Representational Similarity Analysis) -- Kriegeskorte et al. (2008)
4. Procrustes alignment                 -- used in Ding et al. (2021)
5. Mutual kNN overlap                   -- simple nonparametric baseline
6. Partial isomorphism score            -- custom metric for this project

References
----------
- "Similarity of Neural Network Representations Revisited" (Kornblith et al.)
  https://arxiv.org/abs/1905.00414
- "Revisiting Model Similarity" (Morcos et al.)
  https://arxiv.org/abs/1810.11750
- "Grounding Representation Similarity with Statistical Testing" (Ding et al.)
  https://arxiv.org/abs/2108.01wrap

Usage
-----
    from representation_similarity import (
        cka_linear, cka_rbf,
        procrustes_similarity,
        rsa_correlation,
        partial_isomorphism_score,
    )
    sim = cka_linear(X1, X2)   # X1, X2: (n_samples, d) numpy arrays
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# 1. Centered Kernel Alignment (CKA)
# ---------------------------------------------------------------------------

def _center_gram(K: np.ndarray) -> np.ndarray:
    """Return doubly-centred Gram matrix."""
    n = K.shape[0]
    ones = np.ones((n, 1), dtype=K.dtype)
    H = np.eye(n, dtype=K.dtype) - ones @ ones.T / n
    return H @ K @ H


def cka_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between representations X and Y.

    Parameters
    ----------
    X, Y : (n_samples, d) float arrays. n_samples must match.

    Returns
    -------
    cka : float in [0, 1].  1.0 = identical structure, 0.0 = orthogonal.
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    XXT = X @ X.T
    YYT = Y @ Y.T
    HSIC_XY = np.linalg.norm(_center_gram(XXT) * _center_gram(YYT), "fro")
    HSIC_XX = np.linalg.norm(_center_gram(XXT) ** 2, "fro") ** 0.5
    HSIC_YY = np.linalg.norm(_center_gram(YYT) ** 2, "fro") ** 0.5
    if HSIC_XX == 0 or HSIC_YY == 0:
        return 0.0
    return float(HSIC_XY / (HSIC_XX * HSIC_YY))


def cka_rbf(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
    """RBF-kernel CKA (unbiased estimator).

    Parameters
    ----------
    sigma : RBF bandwidth.  None = median heuristic.
    """
    def rbf_kernel(Z: np.ndarray, sig: float) -> np.ndarray:
        sq = np.sum(Z ** 2, axis=1, keepdims=True)
        D = sq + sq.T - 2 * (Z @ Z.T)
        return np.exp(-D / (2 * sig ** 2))

    if sigma is None:
        # Median heuristic for X
        sq = np.sum(X ** 2, axis=1, keepdims=True)
        D = sq + sq.T - 2 * (X @ X.T)
        sigma = float(np.sqrt(np.median(D[D > 0])))

    KX = rbf_kernel(X, sigma)
    KY = rbf_kernel(Y, sigma)
    return float(
        np.sum(_center_gram(KX) * _center_gram(KY))
        / (np.linalg.norm(_center_gram(KX), "fro") * np.linalg.norm(_center_gram(KY), "fro") + 1e-12)
    )


# ---------------------------------------------------------------------------
# 2. Procrustes alignment
# ---------------------------------------------------------------------------

def procrustes_similarity(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int | None = None,
) -> dict:
    """Orthogonal Procrustes alignment of X onto Y.

    Finds rotation R minimising ||XR - Y||_F and returns the residual
    disparity and similarity score.

    Parameters
    ----------
    X, Y          : (n, d) arrays.
    n_components  : If given, reduce both to this many PCA components first.

    Returns
    -------
    dict with keys:
      similarity   : float in [0, 1]  (1 = perfect alignment)
      disparity    : float (Procrustes distance, lower = more similar)
      rotation     : (d, d) orthogonal matrix R
    """
    if n_components is not None:
        X = _pca(X, n_components)
        Y = _pca(Y, n_components)

    # Normalise
    X = X / (np.linalg.norm(X, "fro") + 1e-12)
    Y = Y / (np.linalg.norm(Y, "fro") + 1e-12)

    # SVD-based solution to Procrustes
    M = X.T @ Y
    U, s, Vt = np.linalg.svd(M)
    R = U @ Vt  # optimal rotation

    XR = X @ R
    disparity = float(np.linalg.norm(XR - Y, "fro") ** 2)
    similarity = 1.0 - disparity / 2.0  # in [0,1] after normalisation
    return {"similarity": similarity, "disparity": disparity, "rotation": R}


# ---------------------------------------------------------------------------
# 3. Representational Similarity Analysis (RSA)
# ---------------------------------------------------------------------------

def rsa_correlation(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "cosine",
) -> float:
    """Spearman correlation between pairwise distance matrices of X and Y.

    Parameters
    ----------
    metric : "cosine" or "euclidean".

    Returns
    -------
    Spearman rho in [-1, 1].
    """
    RX = _rdm(X, metric)
    RY = _rdm(Y, metric)
    # Extract upper triangle (excluding diagonal)
    tri = np.triu_indices(RX.shape[0], k=1)
    rho, _ = spearmanr(RX[tri], RY[tri])
    return float(rho)


def _rdm(X: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Representational dissimilarity matrix."""
    n = X.shape[0]
    if metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Xn = X / norms
        cos_sim = Xn @ Xn.T
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        return 1.0 - cos_sim
    elif metric == "euclidean":
        sq = np.sum(X ** 2, axis=1, keepdims=True)
        D2 = sq + sq.T - 2 * (X @ X.T)
        return np.sqrt(np.clip(D2, 0, None))
    else:
        raise ValueError(f"Unknown metric {metric!r}")


# ---------------------------------------------------------------------------
# 4. Mutual k-NN overlap
# ---------------------------------------------------------------------------

def mutual_knn_overlap(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 10,
) -> float:
    """Fraction of shared k-nearest neighbours across the two representations.

    Returns a value in [0, 1] where 1 means the two representations have
    identical local neighbourhood structure.
    """
    # For efficiency, use cosine distance
    def knn_sets(Z: np.ndarray, k_: int) -> list[set]:
        norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        Zn = Z / norms
        sim = Zn @ Zn.T
        np.fill_diagonal(sim, -np.inf)
        idx = np.argsort(-sim, axis=1)[:, :k_]
        return [set(row) for row in idx]

    nx = knn_sets(X, k)
    ny = knn_sets(Y, k)
    overlap = np.mean([
        len(nx[i] & ny[i]) / k for i in range(len(nx))
    ])
    return float(overlap)


# ---------------------------------------------------------------------------
# 5. Partial isomorphism score (custom)
# ---------------------------------------------------------------------------

def partial_isomorphism_score(
    X: np.ndarray,
    Y: np.ndarray,
    feature_labels_X: np.ndarray,
    feature_labels_Y: np.ndarray,
    n_components: int = 20,
) -> dict:
    """Measure the degree of *partial* (i.e. unexpected cross-feature)
    alignment between two representations.

    Intuition
    ---------
    A *full* isomorphism would map each feature direction in X to the
    *same* feature direction in Y.  A *partial* isomorphism means the
    optimal linear map mixes feature directions in an unexpected way.

    We quantify this by:
    1. Computing the Procrustes rotation R from X -> Y.
    2. For each feature in feature_labels_X, checking whether R maps
       that feature's direction primarily onto the *same* feature in
       feature_labels_Y, or onto a *different* one.

    Parameters
    ----------
    X, Y                : (n_samples, d) representations from two models.
    feature_labels_X/Y  : (n_samples,) integer feature class labels.
    n_components        : PCA dimensionality before alignment.

    Returns
    -------
    dict with:
      aligned_fraction  : fraction of features mapping to the same label
      confusion_matrix  : (n_classes, n_classes) alignment confusion
      unexpected_pairs  : list of (label_X, label_Y, strength) tuples
                          where label_X != label_Y
    """
    from collections import defaultdict

    # PCA-reduce
    Xr = _pca(X, n_components)
    Yr = _pca(Y, n_components)

    # Per-class mean vectors
    classes = np.unique(feature_labels_X)
    n_cls = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    means_X = np.array([
        Xr[feature_labels_X == c].mean(axis=0) for c in classes
    ])  # (n_cls, n_components)
    means_Y = np.array([
        Yr[feature_labels_Y == c].mean(axis=0) for c in classes
    ])

    # Procrustes on class means
    proc = procrustes_similarity(means_X, means_Y)
    R = proc["rotation"]

    aligned_means = means_X @ R  # (n_cls, n_components)

    # Build confusion: for each class in X, which class in Y is closest?
    norms_Y = np.linalg.norm(means_Y, axis=1, keepdims=True) + 1e-12
    norms_A = np.linalg.norm(aligned_means, axis=1, keepdims=True) + 1e-12
    cos_sim = (aligned_means / norms_A) @ (means_Y / norms_Y).T  # (n_cls, n_cls)

    assigned_Y = np.argmax(cos_sim, axis=1)  # best-matching Y class for each X class
    aligned_fraction = float((assigned_Y == np.arange(n_cls)).mean())

    confusion = np.zeros((n_cls, n_cls), dtype=np.float32)
    for i in range(n_cls):
        confusion[i] = np.clip(cos_sim[i], 0, None)

    # Unexpected pairs: X-class maps to DIFFERENT Y-class
    unexpected_pairs = []
    for i, cx in enumerate(classes):
        j = assigned_Y[i]
        if j != i:
            strength = float(cos_sim[i, j])
            unexpected_pairs.append((int(cx), int(classes[j]), strength))
    unexpected_pairs.sort(key=lambda t: -t[2])

    return {
        "aligned_fraction": aligned_fraction,
        "confusion_matrix": confusion,
        "unexpected_pairs": unexpected_pairs,
        "procrustes_similarity": proc["similarity"],
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """Reduce X to n_components via PCA (mean-centred SVD)."""
    X = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    return X @ Vt[:k].T


def compare_all_metrics(
    X: np.ndarray,
    Y: np.ndarray,
    k_knn: int = 10,
    n_proc_components: int = 50,
) -> dict:
    """Run all similarity metrics and return a summary dict.

    Parameters
    ----------
    X, Y : (n_samples, d) representation matrices.

    Returns
    -------
    dict with keys: cka_linear, cka_rbf, procrustes, rsa_cosine,
                    rsa_euclidean, mutual_knn.
    """
    results = {}
    print("Computing CKA (linear) ...")
    results["cka_linear"] = cka_linear(X, Y)
    print(f"  cka_linear = {results['cka_linear']:.4f}")

    print("Computing RSA (cosine) ...")
    results["rsa_cosine"] = rsa_correlation(X, Y, metric="cosine")
    print(f"  rsa_cosine = {results['rsa_cosine']:.4f}")

    print(f"Computing Procrustes (top {n_proc_components} PCA dims) ...")
    proc = procrustes_similarity(X, Y, n_components=n_proc_components)
    results["procrustes_similarity"] = proc["similarity"]
    results["procrustes_disparity"] = proc["disparity"]
    print(f"  procrustes_similarity = {results['procrustes_similarity']:.4f}")

    print(f"Computing mutual {k_knn}-NN overlap ...")
    results["mutual_knn"] = mutual_knn_overlap(X, Y, k=k_knn)
    print(f"  mutual_knn_overlap = {results['mutual_knn']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Representation Similarity Benchmarks -- self-test\n")
    rng = np.random.default_rng(0)
    n, d = 200, 64

    # Case 1: identical representations
    X = rng.standard_normal((n, d)).astype(np.float32)
    print("--- Identical representations ---")
    r = compare_all_metrics(X, X.copy())
    print()

    # Case 2: rotated (should be high similarity for rotation-invariant metrics)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Y_rot = X @ Q
    print("--- Rotated representations ---")
    r = compare_all_metrics(X, Y_rot)
    print()

    # Case 3: independent noise
    Z = rng.standard_normal((n, d)).astype(np.float32)
    print("--- Independent representations ---")
    r = compare_all_metrics(X, Z)
    print()

    print("Self-test complete.")
