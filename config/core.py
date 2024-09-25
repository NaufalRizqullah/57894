from pydantic_settings import BaseSettings

class Config(BaseSettings):
    IMAGE_CHANNEL: int = 3
    LABEL_CHANNEL: int = 3
    NUM_CLASSES: int = 3
    IMAGE_SIZE: int = 128
    FEATURES_DISCRIMINATOR: int = 64 * 2
    FEATURES_GENERATOR: int = 64 * 2
    EMBED_SIZE: int = 30 + 20
    INPUT_Z_DIM: int = 64 * 2
    BATCH_SIZE: int = 20
    DISPLAY_STEP: int = 500
    MAX_SAMPLES: int = 2500

    LEARNING_RATE: float = 0.0002
    BETA_1: float = 0.5
    BETA_2: float = 0.999
    C_LAMBDA: int = 10

    NUM_EPOCH: int = 200 * 5

    CRITIC_REPEAT: int = 3

    LOAD_CHECKPOINT: bool = True
    PATH_DATASET: str = "/kaggle/input/shoe-vs-sandal-vs-boot-dataset-15k-images/Shoe vs Sandal vs Boot Dataset"
    CKPT_PATH: str = "./weights/epoch=299-step=450000.ckpt"

    OPTIONS_MAPPING: dict = {
        "Boot": 0,
        "Sandal": 1,
        "Shoe": 2
    }

config = Config()