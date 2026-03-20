"""
Othello Game Sequence Dataset
==============================
Generates and downloads Othello game sequences as used in:
  "Emergent World Representations: Exploring a Sequence Model Trained on a
   Synthetic Task"  (Li et al., 2022)   https://arxiv.org/abs/2210.13382
  "Othello is Solved" (Nanda et al., 2023)  -- probing / linear reps

Each game is a sequence of board moves encoded as integers 0-63
(row-major, 8x8 board). The standard dataset contains ~130 000 games.

Data sources
------------
1. Primary: roamresearch/OthelloGPT on HuggingFace Datasets
   (requires `datasets` library from HuggingFace)
2. Fallback: synthetic games generated from the open-source Othello engine
   (pure Python, no external deps beyond numpy)

This script:
  - Tries to download the official dataset via HuggingFace `datasets`
  - Falls back to generating synthetic games if that library is unavailable
  - Provides utility functions for converting move sequences to board states

Usage
-----
    python generate_othello.py                 # download or generate ~50k games
    python generate_othello.py --n_games 5000  # generate synthetic games only
    python generate_othello.py --hf_only       # only try HuggingFace download
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Board constants
# ---------------------------------------------------------------------------

EMPTY, BLACK, WHITE = 0, 1, -1
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def rc_to_idx(r: int, c: int) -> int:
    return r * 8 + c

def idx_to_rc(idx: int) -> tuple[int, int]:
    return idx // 8, idx % 8


# ---------------------------------------------------------------------------
# Minimal Othello engine (no external deps)
# ---------------------------------------------------------------------------

class OthelloBoard:
    """8x8 Othello board."""

    def __init__(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        # Standard starting position
        self.board[3, 3] = WHITE
        self.board[3, 4] = BLACK
        self.board[4, 3] = BLACK
        self.board[4, 4] = WHITE
        self.current_player = BLACK  # Black moves first

    def copy(self) -> "OthelloBoard":
        b = OthelloBoard.__new__(OthelloBoard)
        b.board = self.board.copy()
        b.current_player = self.current_player
        return b

    def _flips(self, r: int, c: int, player: int) -> list[tuple[int, int]]:
        """Return list of cells that would be flipped by placing player at (r,c)."""
        if self.board[r, c] != EMPTY:
            return []
        flips = []
        opp = -player
        for dr, dc in DIRS:
            line = []
            nr, nc = r + dr, c + dc
            while 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr, nc] == opp:
                line.append((nr, nc))
                nr += dr
                nc += dc
            if line and 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr, nc] == player:
                flips.extend(line)
        return flips

    def legal_moves(self, player: Optional[int] = None) -> list[int]:
        if player is None:
            player = self.current_player
        moves = []
        for r in range(8):
            for c in range(8):
                if self._flips(r, c, player):
                    moves.append(rc_to_idx(r, c))
        return moves

    def play(self, idx: int) -> bool:
        """Play move at cell idx for current player. Returns True on success."""
        r, c = idx_to_rc(idx)
        flips = self._flips(r, c, self.current_player)
        if not flips:
            return False
        self.board[r, c] = self.current_player
        for fr, fc in flips:
            self.board[fr, fc] = self.current_player
        self.current_player = -self.current_player
        return True

    def is_game_over(self) -> bool:
        if self.legal_moves():
            return False
        # Current player has no moves; check if opponent also has none
        if self.legal_moves(-self.current_player):
            return False
        return True

    def board_state(self) -> np.ndarray:
        """Return (64,) int8 array: -1=white, 0=empty, 1=black."""
        return self.board.ravel().copy()


def generate_random_game(rng: random.Random) -> Optional[list[int]]:
    """Play a game with random move selection. Returns move sequence or None."""
    board = OthelloBoard()
    moves = []
    passes = 0
    while not board.is_game_over() and len(moves) < 64:
        legal = board.legal_moves()
        if not legal:
            # Must pass
            board.current_player = -board.current_player
            passes += 1
            if passes >= 2:
                break
            continue
        passes = 0
        move = rng.choice(legal)
        board.play(move)
        moves.append(move)
    return moves if len(moves) >= 5 else None


def generate_games_synthetic(n_games: int, seed: int = 42) -> list[list[int]]:
    """Generate n_games random Othello games."""
    rng = random.Random(seed)
    games = []
    attempts = 0
    while len(games) < n_games:
        attempts += 1
        g = generate_random_game(rng)
        if g is not None:
            games.append(g)
        if attempts > n_games * 5:
            print(f"  Warning: only generated {len(games)} games after {attempts} attempts")
            break
    return games


def get_board_states(moves: list[int]) -> np.ndarray:
    """Return (len(moves)+1, 64) array of board states after each move."""
    board = OthelloBoard()
    states = [board.board_state()]
    for m in moves:
        board.play(m)
        states.append(board.board_state())
    return np.stack(states, axis=0).astype(np.int8)


# ---------------------------------------------------------------------------
# HuggingFace download attempt
# ---------------------------------------------------------------------------

def try_download_huggingface(out_dir: Path) -> bool:
    """Attempt to download the official OthelloGPT dataset from HuggingFace.

    Returns True if successful, False otherwise.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("  `datasets` library not available. Install with:")
        print("    python3 -m pip install datasets")
        return False

    print("  Attempting HuggingFace download: roamresearch/OthelloGPT ...")
    try:
        ds = load_dataset("roamresearch/OthelloGPT", split="train")
        games = [row["moves"] for row in ds]
        save_games(out_dir, games, source="huggingface")
        print(f"  Downloaded {len(games)} games from HuggingFace.")
        return True
    except Exception as e:
        print(f"  HuggingFace download failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_games(out_dir: Path, games: list[list[int]], source: str = "synthetic") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Pad sequences to max length with -1 sentinel
    max_len = max(len(g) for g in games)
    arr = np.full((len(games), max_len), fill_value=-1, dtype=np.int8)
    lengths = np.array([len(g) for g in games], dtype=np.int16)
    for i, g in enumerate(games):
        arr[i, : len(g)] = g

    np.save(out_dir / "games.npy", arr)
    np.save(out_dir / "game_lengths.npy", lengths)

    meta = {
        "source": source,
        "n_games": len(games),
        "max_seq_len": max_len,
        "mean_seq_len": float(lengths.mean()),
        "encoding": "integers 0-63, row-major 8x8, -1=padding",
        "board_values": {"empty": 0, "black": 1, "white": -1},
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved {len(games)} games  "
          f"(mean_len={meta['mean_seq_len']:.1f}, max_len={max_len})")


def load_games(out_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (games_arr, lengths, meta).  games_arr[i] is padded with -1."""
    arr = np.load(out_dir / "games.npy")
    lengths = np.load(out_dir / "game_lengths.npy")
    meta = json.loads((out_dir / "meta.json").read_text())
    return arr, lengths, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate/download Othello game sequences")
    p.add_argument("--n_games", type=int, default=50_000,
                   help="Number of synthetic games to generate as fallback")
    p.add_argument("--hf_only", action="store_true",
                   help="Only try HuggingFace download, do not fall back to synthetic")
    p.add_argument("--synthetic_only", action="store_true",
                   help="Skip HuggingFace download, generate synthetic games")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=Path,
                   default=Path(__file__).parent)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.out_dir

    if not args.synthetic_only:
        success = try_download_huggingface(out_dir)
    else:
        success = False

    if not success and not args.hf_only:
        print(f"\n  Generating {args.n_games} synthetic random games ...")
        games = generate_games_synthetic(args.n_games, seed=args.seed)
        save_games(out_dir, games, source="synthetic_random")

    print("\nDone.")
    print("\nQuick usage example:")
    print("  from generate_othello import load_games, get_board_states")
    print("  games, lengths, meta = load_games(out_dir)")
    print("  # Get board state sequence for game 0:")
    print("  states = get_board_states(games[0][:lengths[0]].tolist())")
