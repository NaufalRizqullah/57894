import torch
import lightning as L
import torchvision.transforms as T
import os

from config.core import config
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utility.helper import PadToSquare

class ShoeSandalBoot(L.LightningDataModule):
    def __init__(
        self, 
        dataset_directory, 
        image_size=config.image_size, 
        batch_size=config.image_size,
        max_samples=None
    ):
        super().__init__()

        self.data_dir = dataset_directory
        self.bs = batch_size
        self.max_samples = max_samples # to limit dataset.

        self.transforms = T.Compose([
            # T.Resize(size=(image_size, image_size)),
            PadToSquare(image_size),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        self.ssb = None

    def prepare_data(self):
        pass

    def setup(self, stage):
        if stage == "fit":
            dataset = ImageFolder(
                root=self.data_dir,
                transform=self.transforms
            )
            
            # To Limit Dataset
            if self.max_samples:
                print(f"[INFO] Dataset is Limited to {self.max_samples} Samples")
                self.ssb = torch.utils.data.Subset(dataset, range(min(len(dataset), self.max_samples)))
            else:
                self.ssb = dataset

    def train_dataloader(self):
        return DataLoader(self.ssb, batch_size=self.bs, num_workers=os.cpu_count(), shuffle=True)