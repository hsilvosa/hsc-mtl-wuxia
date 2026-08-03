import torch
from torch.utils.data import DataLoader, Subset

from .constants import DEVICE


def build_dataloader(tokenized_dataset, batch_size: int, shuffle: bool = True):
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle)


def select_fraction(dataset, fraction: float, seed: int = 42):
    if fraction >= 1.0:
        return dataset
    n = max(1, int(len(dataset) * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    return Subset(dataset, indices)
