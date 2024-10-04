import argparse
import lightning as L

from config.core import config
from training.model import Pix2Pix
from training.callbacks import MyCustomSavingCallback
from data.dataloader import FaceToComicDataModule


# Add argparser for config params
parser = argparse.ArgumentParser()
parser.add_argument("--load_checkpoint", type=bool, default=config.LOAD_CHECKPOINT)
parser.add_argument("--ckpt_path", type=str, default=config.CKPT_PATH)
parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE)
parser.add_argument("--l1_lambda", type=int, default=config.L1_LAMBDA)
parser.add_argument("--features_discriminator", type=int, nargs='+', default=config.FEATURE_DISCRIMINATOR)
parser.add_argument("--features_generator", type=int, default=config.FEATURE_GENERATOR)
parser.add_argument("--display_step", type=int, default=config.DISPLAY_STEP)
parser.add_argument("--num_epoch", type=int, default=config.NUM_EPOCH)
parser.add_argument("--path_face", type=str, default=config.PATH_FACE)
parser.add_argument("--path_comic", type=str, default=config.PATH_COMIC)
parser.add_argument("--image_size", type=int, default=config.IMAGE_SIZE)
parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
parser.add_argument("--max_samples", type=int, default=config.MAX_SAMPLES)

args = parser.parse_args()

config.LOAD_CHECKPOINT = args.load_checkpoint
config.CKPT_PATH = args.ckpt_path
config.LEARNING_RATE = args.learning_rate
config.L1_LAMBDA = args.l1_lambda
config.FEATURE_DISCRIMINATOR = args.features_discriminator
config.FEATURE_GENERATOR = args.features_generator
config.DISPLAY_STEP = args.display_step
config.NUM_EPOCH = args.num_epoch
config.PATH_FACE = args.path_face
config.PATH_COMIC = args.path_comic
config.IMAGE_SIZE = args.image_size
config.BATCH_SIZE = args.batch_size
config.MAX_SAMPLES = args.max_samples

# Initialize the Model Lightning
model = Pix2Pix(
    in_channels=3,
    learning_rate=config.LEARNING_RATE,
    l1_lambda=config.L1_LAMBDA,
    features_discriminator=config.FEATURE_DISCRIMINATOR,
    features_generator=config.FEATURE_GENERATOR,
    display_step=config.DISPLAY_STEP,
)

# Setup Trainer
n_log = None

trainer = L.Trainer(
    accelerator="auto",
    devices="auto",
    strategy="auto",
    log_every_n_steps=n_log,
    max_epochs=config.NUM_EPOCH,
    callbacks=[MyCustomSavingCallback()],
    default_root_dir="/kaggle/working/",
    precision="16-mixed",
    fast_dev_run=True
)

# Lightning DataModule
dm = FaceToComicDataModule(
    face_path=config.PATH_FACE, 
    comic_path=config.PATH_COMIC, 
    image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE), 
    batch_size=config.BATCH_SIZE,
    max_samples=None
)

# Training set
if config.LOAD_CHECKPOINT:
    trainer.fit(model, datamodule=dm, ckpt_path=config.CKPT_PATH)
else:
    trainer.fit(model, datamodule=dm)