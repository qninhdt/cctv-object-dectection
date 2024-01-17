from typing import List, Tuple, Callable
from pathlib import Path

import json

import torch
import numpy as np
from random import Random
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
from torch.utils.data import Dataset
import cv2


class CCTVDataset(Dataset):
    def __init__(self, unclean_data_dir: str, clean_data_dir: str):
        super().__init__()

        self.unclean_data_dir = Path(unclean_data_dir)
        self.clean_data_dir = Path(clean_data_dir)

        self.annotations: List[dict] = json.load(
            open(self.unclean_data_dir / "annotations.json")
        )

        self.images: List[dict] = self.annotations["images"]
        Random(1).shuffle(self.images)

        self.categorys = self.annotations["categories"]

    def __getitem__(self, idx: int) -> dict:
        # copy annotation of image
        target: dict = self.images[idx].copy()

        # load unimage
        unclean_image = cv2.imread(
            str(self.unclean_data_dir / "images" / target["filename"])
        )

        # load clean image
        clean_image = cv2.imread(
            str(self.clean_data_dir / "images" / target["filename"])
        )

        boxes = [
            [bbox["x"], bbox["y"], bbox["width"], bbox["height"]]
            for bbox in target["boxes"]
        ]
        labels = [bbox["category_id"] for bbox in target["boxes"]]

        boxes = torch.tensor(boxes, dtype=torch.float32)
        # cxcywh -> xywh
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2

        boxes = torch.abs(boxes)

        labels = torch.tensor(labels, dtype=torch.int64)

        sample = {
            "clean_image": clean_image,
            "image": unclean_image,
            "boxes": boxes,
            "labels": labels,
            "type": target["type"],
        }

        return sample

    def __len__(self) -> int:
        return len(self.annotations["images"])

    def num_classes(self) -> int:
        return len(self.annotations["categories"])

    def get_category(self, idx: int) -> str:
        return self.categorys[idx - 1]
