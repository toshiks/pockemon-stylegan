from typing import Optional

from pytorch_lightning import LightningDataModule
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

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.path_to_dataset = path_to_dataset

        self.train: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if not self.train:
            self.train = MultiResolutionDataset(self.path_to_dataset, self.resolution)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
        )
