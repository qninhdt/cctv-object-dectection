from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ApplyTransform(Dataset):
    """Apply transform to dataset"""

    def __init__(self, dataset: Dataset, transform: nn.Module):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        output = self.transform(
            image=sample["image"],
            bboxes=sample["boxes"],
            clean_image=sample["clean_image"],
            labels=sample["labels"],
        )
        sample["image"] = output["image"]
        sample["boxes"] = torch.tensor(output["bboxes"], dtype=torch.float32)
        sample["clean_image"] = output["clean_image"]
        # normalize boxes and convert to cxcywh
        w, h = sample["image"].shape[1:]
        scale = torch.tensor([w, h, w, h], dtype=torch.float32)
        sample["boxes"] = sample["boxes"] / scale
        sample["boxes"][:, 0] = sample["boxes"][:, 0] + sample["boxes"][:, 2] / 2
        sample["boxes"][:, 1] = sample["boxes"][:, 1] + sample["boxes"][:, 3] / 2

        return sample

    def __len__(self) -> int:
        return len(self.dataset)
