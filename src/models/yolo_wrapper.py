import torch
from torch import nn
from ultralytics import YOLO

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class YOLOWrapper(nn.Module):

    def __init__(
        self
    ) -> None:
        super().__init__()

        self.yolo = YOLO('yolov8n.pt')
        print(self.yolo.model)

        for param in self.yolo.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.yolo(x)
        print(output[0].shape)
        return output[0]
