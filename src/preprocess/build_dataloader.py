import lightning as L
import os

from torch.utils.data import DataLoader
from src.preprocess.build_dataset import YosemiteDataset

class Summer2Winter(L.LightningDataModule):
    def __init__(
        self, 
        dataset_summer,
        dataset_winter,
        batch_size,
        transform,
    ):
        super().__init__()

        self.dataset_summer = dataset_summer
        self.dataset_winter = dataset_winter
        
        self.bs = batch_size
        self.transforms = transform
        
        self.s2w = None

    def prepare_data(self):
        pass

    def setup(self, stage):
        if stage == "fit":
            self.s2w = YosemiteDataset(
                summer_path=self.dataset_summer,
                winter_path=self.dataset_winter,
                transforms=self.transforms
            )

    def train_dataloader(self):
        return DataLoader(self.s2w, batch_size=self.bs, num_workers=os.cpu_count(), shuffle=True)