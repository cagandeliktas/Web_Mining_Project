"""In-memory torch dataset over a FeatureSet split, with fast tensor batching.

In plain words: Holds one split's arrays as PyTorch tensors and hands them to the training loop in
batches.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class InteractionData(Dataset):
    """One split of a FeatureSet as tensors. ``batches`` slices tensors directly
    (no per-row Python), which is much faster than a DataLoader for dense data."""

    def __init__(self, part: dict, device: torch.device | str = "cpu"):
        """Convert one split's arrays to tensors and move them to the chosen device."""
        self.device = torch.device(device)
        self.X_user = torch.from_numpy(np.ascontiguousarray(part["X_user"])).to(self.device)
        self.X_item = torch.from_numpy(np.ascontiguousarray(part["X_item"])).to(self.device)
        self.user_idx = torch.from_numpy(part["user_idx"]).to(self.device)
        self.item_idx = torch.from_numpy(part["item_idx"]).to(self.device)
        self.y = torch.from_numpy(part["y"]).to(self.device)

    def __len__(self) -> int:
        """Number of reviews in this split."""
        return len(self.y)

    def __getitem__(self, i):
        """Return the user features, item features, ids and true rating for row(s) i."""
        return self.X_user[i], self.X_item[i], self.user_idx[i], self.item_idx[i], self.y[i]

    def batches(self, batch_size: int, shuffle: bool = False, generator: torch.Generator | None = None):
        """Yield the data in chunks of `batch_size` (shuffled for training, in order for evaluation).
        """
        n = len(self)
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        order = order.to(self.device)
        for start in range(0, n, batch_size):
            yield self[order[start : start + batch_size]]
