from typing import Any, Dict

import torchvision.transforms.v2 as T
import torch
import torch.nn as nn


class SquarePad(nn.Module):
    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"]

        h = image.shape[1]
        w = image.shape[2]

        if h > w:
            pad = T.Pad(((h - w) // 2, 0))
        elif w > h:
            pad = T.Pad((0, (w - h) // 2))

        return T.Compose([pad])(sample)
    
class Normalize(nn.Module):
    def __init__(self, mean: list, std: list) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        boxes = sample["boxes"]

        w, h = boxes.canvas_size
        scale = torch.tensor([w, h, w, h], dtype=torch.float32)

        sample["nboxes"] = torch.clone(boxes) / scale
        sample["boxes"] = torch.clone(boxes)

        return T.Compose([T.Normalize(self.mean, self.std)])(sample)
