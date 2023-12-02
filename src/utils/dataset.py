from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ApplyTransform(Dataset):
    """Apply transform to dataset"""

    def __init__(self, dataset: Dataset, transform: nn.Module):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        image, target = self.dataset[idx]
        image = self.transform(image)
        return image, target

    def __len__(self) -> int:
        return len(self.dataset)