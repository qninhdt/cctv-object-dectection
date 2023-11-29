from typing import List, Tuple, Callable
from pathlib import Path

import json

import torch
from torchvision.transforms.functional import to_tensor
from PIL.Image import Image, open as open_image
from torch.utils.data import Dataset


class CCTVDataset(Dataset):
    def __init__(self, data_dir: str, transforms: Callable) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.transforms = transforms

        self.annotations: List[dict] = json.load(
            open(self.data_dir / "annotations.json"))

        self.images: List[dict] = self.annotations['images']

        print(f"Loaded {len(self.images)} images.")

    def __getitem__(self, idx: int) -> Tuple[Image, dict]:

        # copy annotation of image
        target: dict = self.images[idx].copy()

        # load image
        image: Image = open_image(self.data_dir / 'images' /
                                  target['filename']).convert('RGB')

        # convert to tensor
        image: torch.Tensor = self.transforms(image)

        return image, target

        # w, h = image.size

        # boxes = [[bbox['x'], bbox['y'], bbox['x'] + bbox['width'],
        #           bbox['y'] + bbox['height']] for bbox in target['boxes']]
        # classes = [bbox['category_id'] for bbox in target['boxes']]

        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # classes = torch.as_tensor(classes, dtype=torch.int64)

        # size = torch.tensor([int(h), int(w)])
        # original_size = torch.as_tensor([int(h), int(w)])
        # image_id = torch.tensor([target['id']])

        # # clamp boxes to image
        # boxes[:, 0].clamp_(min=0, max=w)
        # boxes[:, 1].clamp_(min=0, max=h)
        # boxes[:, 2].clamp_(min=0, max=w)
        # boxes[:, 3].clamp_(min=0, max=h)

    def __len__(self) -> int:
        return len(self.annotations['images'])

    def num_classes(self) -> int:
        return len(self.annotations['categories'])
