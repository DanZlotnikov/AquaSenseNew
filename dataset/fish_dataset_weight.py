"""Fish dataset loader — weight prediction variant.

Each .npz split file (v2) contains:
    X          : (N, 39, 47)  float32 — normalised input tensor per sample
                                         channels  0-11 : 12-channel 40 Hz waveform (time-varying)
                                         channels 12-46 : 35 physics features tiled across time (constant)
    y_presence : (N,)         int64   — 1 if fish present, 0 otherwise (20 % positive)
    y_weight   : (N,)         float64 — fish weight in g; 0 when fish is absent
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class FishWeightDataset(Dataset):
    """Torch Dataset wrapping a single .npz split file for weight prediction."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        # Store as float32 tensors; presence is float32 for BCEWithLogitsLoss
        self.X        = torch.from_numpy(data["X"].astype(np.float32))            # (N, 39, 47)
        self.presence = torch.from_numpy(data["y_presence"].astype(np.float32))   # (N,)
        self.weight   = torch.from_numpy(data["y_weight"].astype(np.float32))     # (N,)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict:
        return {
            "X":        self.X[idx],
            "presence": self.presence[idx],
            "weight":   self.weight[idx],
        }


def make_weight_dataloader(
    npz_path: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Return a DataLoader for the given .npz split file (weight target)."""
    dataset = FishWeightDataset(npz_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
