# dataset.py – only class definitions
# --------------------------------------------------
# Minimal dataset module that exposes *only* torch
# Dataset classes (no executable code, no helper
# functions, no tests). Import it anywhere without
# side‑effects.

from __future__ import annotations

import pickle
import os
from pathlib import Path
from typing import List, Tuple

import torch

__all__ = [
    "HNTitles",
    "HNTitlesWithScore",
]

# ---------------------------------------------------------------------------
# Utility functions to handle file paths and loading
# ---------------------------------------------------------------------------

def get_tokens_dir() -> Path:
    """Get path to tokens directory consistent with hn_tokeniser.py"""
    return Path(os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__)
                )
            )
        ), 'tokens'
    ))

def _load_pickle(path: str | Path):
    """Load a pickle file from the specified path"""
    path = Path(path)
    
    # If path is not absolute and doesn't exist as is, try looking in tokens directory
    if not path.is_absolute() and not path.exists():
        tokens_path = get_tokens_dir() / path
        if tokens_path.exists():
            path = tokens_path
    
    return pickle.load(open(path, "rb"))


# ---------------------------------------------------------------------------
# 1️⃣  Hacker News titles – variable‑length token sequences
# ---------------------------------------------------------------------------
class HNTitles(torch.utils.data.Dataset):
    """Iterable of padded / unpadded token‑ID sequences for each HN title."""

    def __init__(self, tokens_path: str | Path = "title_token_ids.pkl"):
        self.title_ids: List[List[int]] = _load_pickle(tokens_path)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.title_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        return torch.tensor(self.title_ids[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# 2️⃣  (tokens, score) pairs – for regression tasks
# ---------------------------------------------------------------------------
class HNTitlesWithScore(torch.utils.data.Dataset):
    """Dataset yielding (token_ids, score) tuples."""

    def __init__(
        self,
        tokens_path: str | Path = "title_token_ids.pkl",
        scores_path: str | Path = "scores.pkl",
    ):
        self.title_ids: List[List[int]] = _load_pickle(tokens_path)
        self.scores: torch.Tensor = torch.tensor(_load_pickle(scores_path), dtype=torch.float32)
        assert len(self.title_ids) == len(self.scores), "tokens & scores length mismatch"

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.scores)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        ids = torch.tensor(self.title_ids[idx], dtype=torch.long)
        score = self.scores[idx].unsqueeze(0)  # keep shape (1,)
        return ids, score