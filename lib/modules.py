from datetime import datetime
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchmetrics
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchgeo.datasets.geo import GeoDataset
from torchgeo.datasets.utils import BoundingBox, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler
from torchgeo.samplers.single import GeoSampler

from .constants import CROP_INDICES
from .datasets import CDLMask, NDVIDataset


class CropDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        img_size: Tuple[int, int],
        epoch_size: int,
        batch_size: int,
        num_workers: int,
        val_ratio: float = 0.2,
        positive_indices: List[int] = CROP_INDICES,
        train_years: List[int] = [2019],
        val_years: List[int] = [2020],
    ):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_size = epoch_size
        self.val_ratio = val_ratio
        self.positive_indices = positive_indices
        self.train_years = train_years
        self.val_years = val_years
        self.years = list(set(self.train_years) | set(self.val_years))

    def prepare_data(self) -> None:
        # Download CDL data if necessary
        CDLMask(self.root_dir, positive_indices=self.positive_indices, years=self.years)

    def setup(self, stage=None):
        input_dataset = NDVIDataset(self.root_dir)
        target_dataset = CDLMask(
            self.root_dir, positive_indices=self.positive_indices, years=self.years, download=False
        )
        self.train_dataset = input_dataset & target_dataset  # Intersection dataset
        # Use the same dataset for training and validation, use different RoIs
        self.val_dataset = self.train_dataset
        self.test_dataset = self.train_dataset

    def _get_dataloader(self, dataset: GeoDataset, sampler: GeoSampler) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            prefetch_factor=5 if self.num_workers else 2,
            collate_fn=stack_samples,
        )

    def _get_split_roi(self, ref_dataset: GeoDataset):
        minx, maxx, miny, maxy, mint, maxt = ref_dataset.bounds
        width = ref_dataset.bounds.maxx - ref_dataset.bounds.minx
        height = ref_dataset.bounds.maxy - ref_dataset.bounds.miny
        if height > width:
            train_x = maxx
            val_x = minx
            train_y = maxy - self.val_ratio * height
            val_y = maxy - self.val_ratio * height
        else:
            train_x = maxx - self.val_ratio * width
            val_x = maxx - self.val_ratio * width
            train_y = maxy
            val_y = miny
        train_mint = datetime(min(self.train_years), 1, 1).timestamp()
        train_maxt = datetime(max(self.train_years) + 1, 1, 1).timestamp() - 1
        val_mint = datetime(min(self.val_years), 1, 1).timestamp()
        val_maxt = datetime(max(self.val_years) + 1, 1, 1).timestamp() - 1

        train_roi = BoundingBox(minx, train_x, miny, train_y, train_mint, train_maxt)
        val_roi = BoundingBox(val_x, maxx, val_y, maxy, val_mint, val_maxt)
        return train_roi, val_roi

    def train_dataloader(self) -> DataLoader:
        # Use the first dataset as index source
        train_roi, _ = self._get_split_roi(self.train_dataset)
        sampler = RandomGeoSampler(
            self.train_dataset,
            [i * self.train_dataset.res for i in self.img_size],
            self.epoch_size,
            roi=train_roi,
        )
        return self._get_dataloader(self.train_dataset, sampler)

    def val_dataloader(self) -> DataLoader:
        _, val_roi = self._get_split_roi(self.val_dataset)
        sampler = GridGeoSampler(
            self.val_dataset,
            [i * self.val_dataset.res for i in self.img_size],
            [i * self.val_dataset.res for i in self.img_size],
            roi=val_roi,
        )
        return self._get_dataloader(self.val_dataset, sampler)

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int):
        batch["bbox"] = [(a for a in b) for b in batch["bbox"]]
        return batch


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 37,
        classes: int = 1,
        num_epochs: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        self.model = smp.Unet(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=in_channels,
            classes=self.classes,
        )
        self.loss = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        metrics = torchmetrics.MetricCollection(
            {
                "ap": torchmetrics.BinnedAveragePrecision(num_classes=1, thresholds=100),
                "acc": torchmetrics.Accuracy(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        lr_scheduler = {
            "scheduler": scheduler,
            "name": "lr_scheduler",
        }
        return [optimizer], [lr_scheduler]

    def _shared_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        pred = self(batch["image"])
        for t in pred, batch["mask"]:
            assert torch.all(torch.isfinite(t))
        loss = self.loss(pred, batch["mask"])

        return {"loss": loss, "preds": pred.detach(), "target": batch["mask"]}

    def _shared_step_end(
        self, outputs: Dict[str, Any], metrics: torchmetrics.MetricCollection, prefix: str
    ) -> None:
        m = metrics(outputs["preds"].sigmoid().flatten(), outputs["target"].flatten().to(torch.int))
        self.log(f"{prefix}_loss", outputs["loss"])
        self.log_dict(m)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self._shared_step(batch, batch_idx)

    def training_step_end(self, outputs: Dict[str, Any]) -> None:
        self._shared_step_end(outputs, self.train_metrics, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self._shared_step(batch, batch_idx)

    def validation_step_end(self, outputs: Dict[str, Any]) -> None:
        return self._shared_step_end(outputs, self.val_metrics, "val")
