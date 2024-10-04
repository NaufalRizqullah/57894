import lightning as L
import torchvision.transforms as T
import os

from torch.utils.data import DataLoader, Subset
from data.dataset import FaceToComicDataset

class FaceToComicDataModule(L.LightningDataModule):
    def __init__(
        self, 
        face_path, 
        comic_path, 
        image_size=(128, 128), 
        batch_size=32, 
        max_samples=None
    ):
        super().__init__()

        self.face_dir = face_path
        self.comic_dir = comic_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.max_samples = max_samples

        self.transform_face = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.transform_comic = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.face2comic = None

    def prepare_data(self):
        # No need to download or prepare data, as it's already present in the directories
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset = FaceToComicDataset(
                face_path=self.face_dir,
                comic_path=self.comic_dir,
                transform_face=self.transform_face,
                transform_comic=self.transform_comic
            )
            
            # To Limit Dataset
            if self.max_samples:
                print(f"[INFO] Dataset is Limited to {self.max_samples} Samples")
                self.face2comic = Subset(dataset, range(min(len(dataset), self.max_samples)))
            else:
                self.face2comic = dataset

    def train_dataloader(self):
        return DataLoader(self.face2comic, batch_size=self.batch_size, num_workers=os.cpu_count(), shuffle=True)

    def val_dataloader(self):
        # Implement if you need validation during training
        pass

    def test_dataloader(self):
        # Implement if you need testing after training
        pass
