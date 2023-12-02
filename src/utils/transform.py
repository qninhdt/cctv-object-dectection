from typing import Any, Dict

import torchvision.transforms.v2 as T

import torch.nn as nn

class SquarePad(nn.Module):
    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample['image']

        h = image.shape[1]
        w = image.shape[2]
        
        if h > w:
            pad = T.Pad(((h - w) // 2, 0))
        elif w > h:
            pad = T.Pad((0, (w - h) // 2))

        return T.Compose([pad])(sample)