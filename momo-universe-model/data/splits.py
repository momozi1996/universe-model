"""Deterministic train/val/test split by trajectory seed (SPEC 2.3)."""
import numpy as np


def split_indices(n_total, n_train, n_val, n_test, seed=42):
    """Split trajectory indices deterministically. Raises if counts mismatch."""
    assert n_train + n_val + n_test == n_total, "split counts must sum to n_total"
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    return {
        "train": np.sort(perm[:n_train]),
        "val": np.sort(perm[n_train:n_train + n_val]),
        "test": np.sort(perm[n_train + n_val:]),
    }
