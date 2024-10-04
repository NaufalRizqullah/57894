import argparse
import lightning as L

from config.core import config
from training.model import Pix2Pix
from training.callbacks import MyCustomSavingCallback
from data.dataloader import FaceToComicDataModule


# Add argparser for config params
parser = argparse.ArgumentParser()
parser.add_argument("--load_checkpoint", action='store_true', help="Load checkpoint if this flag is set. If not set, start training from scratch.")
parser.add_argument("--no_load_checkpoint", action='store_false', dest='load_checkpoint', help="Do not load checkpoint. If set, start training from scratch.")

parser.add_argument("--ckpt_path", type=str, default=config.CKPT_PATH, help="Path to checkpoint file. If load_checkpoint is set, this path will be used to load the checkpoint.")
parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE, help="Learning rate for Adam optimizer.")
parser.add_argument("--l1_lambda", type=int, default=config.L1_LAMBDA, help="Scale factor for L1 loss.")
parser.add_argument("--features_discriminator", type=int, nargs='+', default=config.FEATURE_DISCRIMINATOR, help="List of feature sizes for the discriminator network.")
parser.add_argument("--features_generator", type=int, default=config.FEATURE_GENERATOR, help="Feature size for the generator network.")
parser.add_argument("--display_step", type=int, default=config.DISPLAY_STEP, help="Interval of epochs to display loss and save examples.")
parser.add_argument("--num_epoch", type=int, default=config.NUM_EPOCH, help="Number of epochs to train for.")
parser.add_argument("--path_face", type=str, default=config.PATH_FACE, help="Path to folder containing face images.")
parser.add_argument("--path_comic", type=str, default=config.PATH_COMIC, help="Path to folder containing comic images.")
parser.add_argument("--image_size", type=int, default=config.IMAGE_SIZE, help="Size of input images.")
parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size for training.")
parser.add_argument("--max_samples", type=int, default=config.MAX_SAMPLES, help="Maximum number of samples to use for training. If set to None, all samples will be used.")

args = parser.parse_args()

config.LOAD_CHECKPOINT = args.load_checkpoint if args.load_checkpoint is not None else config.LOAD_CHECKPOINT
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
    # fast_dev_run=True
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