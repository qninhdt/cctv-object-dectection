import json
from typing import Any, Dict, Optional, Tuple, List

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2 as T
from utils.dataset import ApplyTransform
from models.detr.util.misc import nested_tensor_from_tensor_list
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .cctv_dataset import CCTVDataset

IMAGE_SIZE = 720


class CCTVDataModule(LightningDataModule):
    def __init__(
        self,
        clean_data_dir: str,
        unclean_data_dir: str,
        train_val_test_split: Tuple[int, int, int],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        for_detr: bool = False,
    ) -> None:
        """Initialize a `CCTVDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param for_detr: Whether the data module is used for DETR. Defaults to `False`.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.for_detr = for_detr

        # data transformations
        self.train_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.LongestMaxSize(max_size=IMAGE_SIZE, interpolation=1),
                A.PadIfNeeded(
                    min_height=IMAGE_SIZE,
                    min_width=IMAGE_SIZE,
                    border_mode=0,
                    value=(0, 0, 0),
                ),
                # A.RandomResizedCrop(
                #     IMAGE_SIZE,
                #     IMAGE_SIZE,
                #     scale=(0.2, 1),
                #     ratio=(1, 1),
                # ),
                A.Normalize(),
                ToTensorV2(),
            ],
            additional_targets={
                "clean_image": "image",
            },
            bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
        )

        self.transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=IMAGE_SIZE, interpolation=1),
                A.PadIfNeeded(
                    min_height=IMAGE_SIZE,
                    min_width=IMAGE_SIZE,
                    border_mode=0,
                    value=(0, 0, 0),
                ),
                A.Normalize(),
                ToTensorV2(),
            ],
            additional_targets={
                "clean_image": "image",
            },
            bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
        )

        self.dataset: Optional[CCTVDataset] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes()

    def prepare_data(self) -> None:
        return

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.dataset = CCTVDataset(
                clean_data_dir=self.hparams.clean_data_dir,
                unclean_data_dir=self.hparams.unclean_data_dir,
            )

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=self.dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(1234),
            )

            self.data_train = ApplyTransform(self.data_train, self.train_transforms)
            self.data_val = ApplyTransform(self.data_val, self.transforms)
            self.data_test = ApplyTransform(self.data_test, self.transforms)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self._create_dataloader(self.data_train, self.batch_size_per_device)

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return self._create_dataloader(self.data_val, self.batch_size_per_device)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """

        return self._create_dataloader(self.data_test, self.batch_size_per_device)

    def _create_dataloader(self, dataset: Dataset, batch_size: int) -> DataLoader[Any]:
        """Create and return a dataloader.

        :param dataset: The dataset to create a dataloader for.
        :return: The dataloader.
        """

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, List[dict]]:
        """Collate a batch of data.

        :param batch: The batch to collate.
        :return: The collated batch.
        """
        images = [x["image"] for x in batch]
        clean_images = [x["clean_image"] for x in batch]
        targets = [
            {k: v for k, v in x.items() if k != "image" and k != "clean_image"}
            for x in batch
        ]

        clean_images = torch.stack(clean_images, dim=0)

        if self.for_detr:
            images = nested_tensor_from_tensor_list(images)

        return images, clean_images, targets
