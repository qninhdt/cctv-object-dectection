import json
from typing import Any, Dict, Optional, Tuple, List

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import transforms as T

from .cctv_dataset import CCTVDataset


class CCTVDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        train_val_test_split: Tuple[int, int, int],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        """Initialize a `CCTVDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize((64, 64)),
            T.Normalize((0.1307,), (0.3081,))
        ])

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
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.dataset = CCTVDataset(
                data_dir=self.hparams.data_dir,
                transforms=self.transforms,
            )

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=self.dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(1234),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self._create_dataloader(self.data_train, self.batch_size_per_device, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return self._create_dataloader(self.data_val, self.batch_size_per_device, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """

        return self._create_dataloader(self.data_test, 1, shuffle=False)

    def _create_dataloader(self, dataset: Dataset, batch_size: int,
                           shuffle: bool) -> DataLoader[Any]:
        """Create and return a dataloader.

        :param dataset: The dataset to create a dataloader for.
        :return: The dataloader.
        """
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Collate a batch of data.

        :param batch: The batch to collate.
        :return: The collated batch.
        """
        images, targets = list(zip(*batch))

        # batch_size x [3, 640, 640] -> [batch_size, 3, 640, 640]
        images = torch.stack(images)

        return images, targets
