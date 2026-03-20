"""
Group Operation Dataset Generator
===================================
Generates datasets of the form (a, b, a*b) for a finite group G, following
the setup from:
  "A Toy Model of Universality: Reverse Engineering How Networks Learn
   Group Operations"  (Chughtai et al., 2023)
  https://arxiv.org/abs/2302.03025

Also closely related to:
  "Progress measures for grokking via mechanistic interpretability"
   (Nanda et al., 2023) which uses modular arithmetic (Z/pZ, +).

The full dataset for a group of order n contains n^2 triples -- all pairs.
This is intentionally exhaustive; train/test splits are created at call time
so that the model must genuinely learn the group table rather than
memorising a subset of it.

Supported groups
----------------
  cyclic_n        -- Z/nZ with addition  (e.g. cyclic_113 for Nanda et al.)
  symmetric_n     -- S_n, permutation group of degree n  (n <= 5 practical)
  dihedral_n      -- D_n, dihedral group of order 2n
  klein4          -- Z/2Z x Z/2Z (Klein four-group)
  quaternion      -- Q8, quaternion group of order 8
  all_cyclic      -- all cyclic groups up to a given order (sweep)

Usage
-----
    python generate_group_ops.py --group cyclic_113
    python generate_group_ops.py --group symmetric_4
    python generate_group_ops.py --group all_cyclic --max_order 200
    python generate_group_ops.py --all
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Abstract group interface
# ---------------------------------------------------------------------------

class FiniteGroup:
    """Finite group represented by its Cayley table."""

    def __init__(
        self,
        name: str,
        order: int,
        elements: list,
        op: Callable,  # op(a, b) -> c, all in elements
    ):
        self.name = name
        self.order = order
        self.elements = elements
        self._elem_to_idx = {e: i for i, e in enumerate(elements)}

        # Pre-compute full Cayley table (order x order int array)
        table = np.zeros((order, order), dtype=np.int32)
        for i, a in enumerate(elements):
            for j, b in enumerate(elements):
                c = op(a, b)
                table[i, j] = self._elem_to_idx[c]
        self.table = table

    def compose(self, i: int, j: int) -> int:
        """Return index of elements[i] * elements[j]."""
        return int(self.table[i, j])

    def all_triples(self) -> np.ndarray:
        """Return (order^2, 3) array of (a_idx, b_idx, c_idx) triples."""
        n = self.order
        idxs = np.arange(n)
        aa, bb = np.meshgrid(idxs, idxs, indexing="ij")
        aa = aa.ravel()
        bb = bb.ravel()
        cc = self.table[aa, bb]
        return np.stack([aa, bb, cc], axis=1).astype(np.int32)

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "order": self.order,
            "n_triples": self.order ** 2,
        }


# ---------------------------------------------------------------------------
# Group constructors
# ---------------------------------------------------------------------------

def make_cyclic(n: int) -> FiniteGroup:
    """Z/nZ with addition."""
    elements = list(range(n))
    return FiniteGroup(
        name=f"cyclic_{n}",
        order=n,
        elements=elements,
        op=lambda a, b: (a + b) % n,
    )


def make_dihedral(n: int) -> FiniteGroup:
    """D_n: dihedral group of order 2n.

    Elements are represented as (rotation, reflection) pairs where
    rotation in {0,...,n-1} and reflection in {0, 1}.
    Group law: (r1, s1) * (r2, s2) = (r1 + (-1)^s1 * r2 mod n, s1 XOR s2).
    """
    elements = [(r, s) for s in range(2) for r in range(n)]
    def op(a, b):
        r1, s1 = a
        r2, s2 = b
        r = (r1 + ((-1) ** s1) * r2) % n
        s = (s1 + s2) % 2
        return (r, s)
    return FiniteGroup(name=f"dihedral_{n}", order=2 * n, elements=elements, op=op)


def make_klein4() -> FiniteGroup:
    """Z/2Z x Z/2Z (Klein four-group), order 4."""
    elements = [(0, 0), (0, 1), (1, 0), (1, 1)]
    return FiniteGroup(
        name="klein4",
        order=4,
        elements=elements,
        op=lambda a, b: ((a[0] + b[0]) % 2, (a[1] + b[1]) % 2),
    )


def make_quaternion() -> FiniteGroup:
    """Q8: quaternion group of order 8.

    Elements: {1, -1, i, -i, j, -j, k, -k} encoded as integers 0..7.
    Multiplication follows the standard quaternion rules.
    """
    # Encoding: 0=1, 1=-1, 2=i, 3=-i, 4=j, 5=-j, 6=k, 7=-k
    # Multiplication table stored directly.
    elems = [0, 1, 2, 3, 4, 5, 6, 7]
    names = ["1", "-1", "i", "-i", "j", "-j", "k", "-k"]

    # Build Cayley table from the standard rules:
    # i^2 = j^2 = k^2 = ijk = -1
    # Products: ij=k, ji=-k, jk=i, kj=-i, ki=j, ik=-j
    def q8_mul(a: int, b: int) -> int:
        # Sign convention: elem = sign * basis
        sign_a = -1 if a % 2 == 1 else 1
        sign_b = -1 if b % 2 == 1 else 1
        basis_a = a // 2   # 0=1-basis, 1=i-basis, 2=j-basis, 3=k-basis
        basis_b = b // 2

        # Multiplication table for basis elements (ignoring sign):
        # 0*x = x, x*0 = x (identity)
        # 1*1 = 0 with sign -1 (i^2 = -1)
        # 1*2 = 3, 2*1 = -3 (ij=k, ji=-k)
        # 1*3 = -2, 3*1 = 2 (ik=-j, ki=j)
        # 2*3 = 1, 3*2 = -1 (jk=i, kj=-i)
        # 2*2 = 0 with sign -1, 3*3 = 0 with sign -1

        basis_mul = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0],
        ]
        sign_mul = [
            [1,  1,  1,  1],
            [1, -1,  1, -1],
            [1, -1, -1,  1],
            [1,  1, -1, -1],
        ]
        result_basis = basis_mul[basis_a][basis_b]
        result_sign = sign_a * sign_b * sign_mul[basis_a][basis_b]
        result_elem = result_basis * 2 + (0 if result_sign > 0 else 1)
        return result_elem

    return FiniteGroup(name="quaternion", order=8, elements=elems, op=q8_mul)


def make_symmetric(n: int) -> FiniteGroup:
    """S_n: symmetric group on n elements, order n!.

    Elements are tuples representing permutations in one-line notation.
    Only practical for n <= 5 (order 120).
    """
    import math
    if n > 5:
        raise ValueError(
            f"S_{n} has order {math.factorial(n)}, too large for exhaustive table. "
            "Use n <= 5."
        )
    elements = list(itertools.permutations(range(n)))

    def compose_perms(a, b):
        # (a composed after b): apply b first, then a
        return tuple(a[b[i]] for i in range(n))

    return FiniteGroup(
        name=f"symmetric_{n}",
        order=len(elements),
        elements=elements,
        op=compose_perms,
    )


def make_direct_product(G1: FiniteGroup, G2: FiniteGroup) -> FiniteGroup:
    """G1 x G2 direct product."""
    elements = list(itertools.product(range(G1.order), range(G2.order)))
    op_table_1 = G1.table
    op_table_2 = G2.table

    def op(a, b):
        return (op_table_1[a[0], b[0]], op_table_2[a[1], b[1]])

    return FiniteGroup(
        name=f"{G1.name}_x_{G2.name}",
        order=G1.order * G2.order,
        elements=elements,
        op=op,
    )


# ---------------------------------------------------------------------------
# Registry of named groups
# ---------------------------------------------------------------------------

def get_group(name: str) -> FiniteGroup:
    """Construct a named group. Supported names:

    cyclic_<n>        e.g. cyclic_113
    dihedral_<n>      e.g. dihedral_6   (order 12)
    symmetric_<n>     e.g. symmetric_4  (order 24)
    klein4
    quaternion
    """
    if name.startswith("cyclic_"):
        n = int(name.split("_")[1])
        return make_cyclic(n)
    if name.startswith("dihedral_"):
        n = int(name.split("_")[1])
        return make_dihedral(n)
    if name.startswith("symmetric_"):
        n = int(name.split("_")[1])
        return make_symmetric(n)
    if name == "klein4":
        return make_klein4()
    if name == "quaternion":
        return make_quaternion()
    raise ValueError(f"Unknown group name: {name!r}")


# ---------------------------------------------------------------------------
# Dataset save/load
# ---------------------------------------------------------------------------

def save_group_dataset(out_dir: Path, group: FiniteGroup) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    triples = group.all_triples()
    table_path = out_dir / f"{group.name}_table.npy"
    triples_path = out_dir / f"{group.name}_triples.npy"
    meta_path = out_dir / f"{group.name}_meta.json"

    np.save(table_path, group.table)
    np.save(triples_path, triples)

    meta = group.metadata()
    meta["table_file"] = table_path.name
    meta["triples_file"] = triples_path.name
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"  {group.name:20s}  order={group.order:5d}  "
        f"triples={triples.shape[0]:7d}  "
        f"table={table_path.stat().st_size/1e3:.1f} kB"
    )
    return triples_path


def load_group_dataset(out_dir: Path, group_name: str):
    """Return (triples, table, metadata) for a saved group dataset."""
    triples = np.load(out_dir / f"{group_name}_triples.npy")
    table = np.load(out_dir / f"{group_name}_table.npy")
    meta_path = out_dir / f"{group_name}_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return triples, table, meta


def make_train_test_split(
    triples: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly split (a, b, c) triples into train/test sets."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(triples))
    n_test = int(len(triples) * test_fraction)
    return triples[idx[n_test:]], triples[idx[:n_test]]


# ---------------------------------------------------------------------------
# Default groups to generate
# ---------------------------------------------------------------------------

DEFAULT_GROUPS = [
    # Cyclic groups: the primary subject of Nanda et al. (2023)
    "cyclic_2",
    "cyclic_5",
    "cyclic_13",
    "cyclic_59",
    "cyclic_97",
    "cyclic_113",   # exact group used in the grokking paper
    # Non-abelian groups
    "dihedral_6",   # order 12
    "dihedral_10",  # order 20
    "symmetric_3",  # order 6  (S3)
    "symmetric_4",  # order 24 (S4)
    # Special small groups
    "klein4",
    "quaternion",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate group operation datasets")
    p.add_argument("--group", type=str, default=None,
                   help="Single group name (e.g. cyclic_113)")
    p.add_argument("--all", action="store_true",
                   help="Generate all default groups")
    p.add_argument("--all_cyclic", action="store_true",
                   help="Generate all cyclic groups Z/nZ for n in range")
    p.add_argument("--max_order", type=int, default=150,
                   help="Maximum group order for --all_cyclic sweep")
    p.add_argument("--out_dir", type=Path,
                   default=Path(__file__).parent,
                   help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    groups_to_generate = []
    if args.group:
        groups_to_generate = [args.group]
    elif args.all_cyclic:
        groups_to_generate = [f"cyclic_{n}" for n in range(2, args.max_order + 1)]
    elif args.all or not args.group:
        groups_to_generate = DEFAULT_GROUPS

    print(f"\nGenerating {len(groups_to_generate)} group dataset(s) -> {out_dir}\n")
    for gname in groups_to_generate:
        try:
            g = get_group(gname)
            save_group_dataset(out_dir, g)
        except Exception as e:
            print(f"  ERROR generating {gname}: {e}")

    print("\nDone.")
    print("\nQuick usage example:")
    print("  from generate_group_ops import load_group_dataset, make_train_test_split")
    print("  triples, table, meta = load_group_dataset(out_dir, 'cyclic_113')")
    print("  train, test = make_train_test_split(triples, test_fraction=0.2)")
