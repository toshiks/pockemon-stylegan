from typing import Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from src.datamodules.datasets.lmdb_dataset import MultiResolutionDataset


class StyleGanDataModule(LightningDataModule):
    def __init__(
            self,
            path_to_dataset: str,
            resolution: int = 128,
            batch_size: int = 64,
            num_workers: int = 0,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if not self.train:
            self.train = MultiResolutionDataset(self.hparams.path_to_dataset, self.hparams.resolution)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=False,
            shuffle=True,
        )
