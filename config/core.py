from pydantic_settings import BaseSettings

class Config(BaseSettings):
    IMAGE_CHANNEL: int = 3
    NUM_CLASSES: int = 3
    IMAGE_SIZE: int = 128
    FEATURES_DISCRIMINATOR: int = 64
    FEATURES_GENERATOR: int = 64
    EMBED_SIZE: int = 64
    INPUT_Z_DIM: int = 64
    BATCH_SIZE: int = 128
    DISPLAY_STEP: int = 500
    MAX_SAMPLES: int = 3000

    LEARNING_RATE: float = 0.0002
    BETA_1: float = 0.5
    BETA_2: float = 0.999
    C_LAMBDA: int = 10

    NUM_EPOCH: int = 200 * 5

    CRITIC_REPEAT: int = 3

    LOAD_CHECKPOINT: bool = True
    PATH_DATASET: str = ""
    CKPT_PATH: str = "./weights/epoch=999-step=96000.ckpt"

    OPTIONS_MAPPING: dict = {
        "Boot": 0,
        "Sandal": 1,
        "Shoe": 2
    }

config = Config()