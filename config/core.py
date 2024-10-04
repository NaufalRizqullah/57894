from pydantic_settings import BaseSettings

class Config(BaseSettings):
    PATH_FACE: str = "/kaggle/input/comic-faces-paired-synthetic-v2/face2comics_v2.0.0_by_Sxela/face2comics_v2.0.0_by_Sxela/faces"
    PATH_COMIC: str = "/kaggle/input/comic-faces-paired-synthetic-v2/face2comics_v2.0.0_by_Sxela/face2comics_v2.0.0_by_Sxela/comics"
    PATH_OUTPUT: str ="/kaggle/working/generates"

    FEATURE_DISCRIMINATOR: list = [64, 128, 256, 512]
    FEATURE_GENERATOR: int = 64

    IMAGE_SIZE: int = 256
    BATCH_SIZE: int = 128
    DISPLAY_STEP: int = 500
    MAX_SAMPLES: int = 5000

    LEARNING_RATE: float = 2e-4
    L1_LAMBDA: int = 100
    NUM_EPOCH: int = 500

    LOAD_CHECKPOINT: bool = False
    CKPT_PATH: str = ""


config = Config()