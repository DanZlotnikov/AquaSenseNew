"""Dataset loader for FishSaltModel.

Each NPZ file built by build_salt_dataset.py contains:
    X          : (N, 39, 12)  float32 — per-file z-normalised acoustic waveform
    X_phys     : (N, 3)       float32 — recording-level physics features used
                                        by the model to self-determine water
                                        conditions (see fish_salt_model.py)
    y_presence : (N,)         int64   — 1 fish present / 0 absent  (10 % positive)
    y_weight   : (N,)         float64 — fish weight in grams; 0 when absent
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class FishSaltDataset(Dataset):
    """Torch Dataset wrapping a single NPZ split file for FishSaltModel."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X        = torch.from_numpy(data["X"].astype(np.float32))           # (N,39,12)
        self.X_phys   = torch.from_numpy(data["X_phys"].astype(np.float32))      # (N,3)
        self.presence = torch.from_numpy(data["y_presence"].astype(np.float32))  # (N,)
        self.weight   = torch.from_numpy(data["y_weight"].astype(np.float32))    # (N,)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict:
        return {
            "X":        self.X[idx],
            "X_phys":   self.X_phys[idx],
            "presence": self.presence[idx],
            "weight":   self.weight[idx],
        }


def make_salt_dataloader(
    npz_path: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Return a DataLoader for the given NPZ split file (FishSaltModel format)."""
    dataset = FishSaltDataset(npz_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
