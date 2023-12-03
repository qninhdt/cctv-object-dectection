from typing import Any, Dict, Tuple, List

import torch
from lightning import LightningModule

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import MaxMetric, MeanMetric

from .detr import DETR, SetCriterion, PostProcess
from .util.misc import reduce_dict

from utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DETRModule(LightningModule):
    def __init__(
        self,
        net: Tuple[DETR, SetCriterion, PostProcess],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model, self.criterion, self.postprocessor = net
        self.backbone = self.model.backbone

        # metric objects for calculating mAP across batches
        self.train_mAP = MeanAveragePrecision("cxcywh", "bbox", [0.5, 0.75])
        self.val_mAP = MeanAveragePrecision("cxcywh", "bbox", [0.5, 0.75])
        self.test_mAP = MeanAveragePrecision("cxcywh", "bbox", [0.5, 0.75])

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        self.train_loss.reset()
        self.train_mAP.reset()

    def on_train_epoch_end(self) -> None:
        metrics = self.train_mAP.compute()

        self.log("train/mAP", metrics['map'], prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'])

    def on_validation_epoch_start(self) -> None:
        self.val_loss.reset()
        self.val_mAP.reset()

        metrics = self.val_mAP.compute()
        self.log("val/mAP", metrics['map'], prog_bar=True)

    def model_step(
        self, batch: Tuple[torch.Tensor, List[dict]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """

        samples, targets = batch
        preds = self.forward(samples)
        losses = self.criterion(preds, targets)

        weight_dict = self.criterion.weight_dict

        loss = sum(losses[k] * weight_dict[k] for k in losses.keys() if k in weight_dict)

        reduced_losses = reduce_dict(losses)
        reduced_scaled_losses = {k: v * weight_dict[k]
                                    for k, v in reduced_losses.items() if k in weight_dict}
        
        reduced_loss = sum(reduced_scaled_losses.values())

        preds = self.postprocess(preds, targets)

        return preds, targets, loss, reduced_loss, reduced_losses

    def training_step(
        self, batch: Tuple[torch.Tensor, List[dict]], batch_idx: int
    ) -> torch.Tensor:
        
        preds, targets, loss, reduced_loss, reduced_losses = self.model_step(batch)

        # # update and log metrics
        self.train_loss(loss)
        self.train_mAP(preds, targets)

        self.log("train/mean_loss", self.train_loss.compute(), prog_bar=True)
        self.log("train/loss", reduced_loss, prog_bar=True)
        self.log("train/class_error", reduced_losses["class_error"], prog_bar=True)
        self.log("train/loss_ce", reduced_losses["loss_ce"], prog_bar=True)
        self.log("train/loss_bbox", reduced_losses["loss_bbox"], prog_bar=True)
        self.log("train/loss_giou", reduced_losses["loss_giou"], prog_bar=True)

        return loss
    
    def postprocess(self, preds: torch.Tensor, targets: dict) -> None:
        target_sizes = torch.tensor([[720, 720] for t in targets], device=self.device)
        preds = self.postprocessor(preds, target_sizes)

        return preds
    
    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds, targets, _, reduced_loss, _ = self.model_step(batch)

        # # update and log metrics
        self.val_loss(reduced_loss)
        self.val_mAP(preds, targets)

        self.log("val/loss", self.val_loss, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # self.log("val/acc_best", self.val_acc_best.compute(),
        #          sync_dist=True, prog_bar=True)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds, targets, _, reduced_loss, _ = self.model_step(batch)

        # update and log metrics
        self.test_loss(reduced_loss)
        self.test_mAP(preds, targets)

        metrics = self.test_mAP.compute()

        self.log("test/loss", self.test_loss, on_epoch=True, prog_bar=True)
        self.log("test/mAP", metrics['map'], on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        # for name, param in self.model.named_parameters():
        #     if "backbone" in name or "transformer" in name:
        #         param.requires_grad_(False)

        optimizer = self.hparams.optimizer(params=self.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
