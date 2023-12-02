from typing import List, Tuple, Callable
from pathlib import Path

import json

import torch
import numpy as np
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
from torch.utils.data import Dataset
from PIL.Image import Image, open as open_image


class CCTVDataset(Dataset):
    def __init__(self, data_dir: str) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)

        self.annotations: List[dict] = json.load(
            open(self.data_dir / "annotations.json")
        )

        self.images: List[dict] = self.annotations["images"]

        self.categorys = self.annotations["categories"]

    def __getitem__(self, idx: int) -> dict:
        # copy annotation of image
        target: dict = self.images[idx].copy()

        # load image
        orig_image: Image = open_image(
            self.data_dir / "images" / target["filename"]
        ).convert("RGB")

        w, h = orig_image.size

        orig_image = torch.tensor(np.array(orig_image, dtype=np.uint8)).permute(2, 0, 1)
        image = orig_image.clone()

        boxes = [
            [bbox["x"], bbox["y"], bbox["width"], bbox["height"]]
            for bbox in target["boxes"]
        ]
        classes = [bbox["category_id"] for bbox in target["boxes"]]

        boxes = tv_tensors.BoundingBoxes(boxes, format="XYWH", canvas_size=(h, w))
        classes = torch.as_tensor(classes, dtype=torch.int64)

        size = torch.tensor([int(h), int(w)])
        image_id = torch.tensor([target["id"]])

        sample = {
            "image": image,
            "image_id": image_id,
            "orig_image": orig_image,
            "size": size,
            "boxes": boxes,
            "labels": classes,
        }

        return sample

    def __len__(self) -> int:
        return len(self.annotations["images"])

    def num_classes(self) -> int:
        return len(self.annotations["categories"])

    def get_category(self, idx: int) -> str:
        return self.categorys[idx - 1]
