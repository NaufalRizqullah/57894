from pydantic import BaseSettings

class Config(BaseSettings):
    LEARNING_RATE: float = 0.0002
    BETA_1: float = 0.5
    BETA_2: float = 0.999
    C_LAMBDA: int = 10
    IMAGE_SIZE: int = 100
    BATCH_SIZE: int = 64
    NUM_EPOCH: int = 600
    Z_DIM: int = 128
    FEATURES_GENERATOR: int = 128
    FEATURES_CRITIC: int = 128
    CRITIC_REPEATS: int = 3
    SSB_SHAPE: tuple = (3, 100, 100)
    N_CLASSES: int = 3
    DISPLAY_STEP: int = 250
    CHECKPOINT_PATH: str = './weights/epoch=999-step=96000.ckpt'

config = Config()