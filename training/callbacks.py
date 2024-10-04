from lightning.pytorch.callbacks import Callback
from utility.helper import update_version_kaggle_dataset

class MyCustomSavingCallback(Callback):
    def __init__(self):
        super().__init__()
        
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        update_version_kaggle_dataset()
