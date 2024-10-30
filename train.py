import argparse
import lightning as L

import src.common.tools as tools
from src.models.train_model import CycleGAN
from src.utils.callbacks import MyCustomSavingCallback
from src.preprocess.build_dataloader import Summer2Winter
from src.preprocess.transformation import get_transformation

def train(config):

    # Initialize the Model Lightning
    model = CycleGAN(
        image_channels=config["image_channels"],
        learning_rate=config["learning_rate"],
        lambda_cycle=config["lambda_cycle"],
        lambda_identity=config["lambda_identity"]
    )

    # Setup Trainer
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        log_every_n_steps=None,
        max_epochs=config["epoch"],
        callbacks=[MyCustomSavingCallback()],
        default_root_dir="/kaggle/working/",
        precision="16-mixed",
        fast_dev_run=config["run_dev"]
    )

    # Lightning DataModule
    dm = Summer2Winter(
        dataset_summer=config["summer_path"],
        dataset_winter=config["winter_path"],
        batch_size=config["batch_size"],
        transform=get_transformation(),
    )

    # Training set
    if config["load_checkpoint"]:
        trainer.fit(model, datamodule=dm, ckpt_path=config["ckpt_path"])
    else:
        trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    config = tools.load_config("./config.yaml")
    
    # Add argparser for config params
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_checkpoint", action='store_true', help="Load checkpoint if this flag is set. If not set, start training from scratch.")
    parser.add_argument("--no_load_checkpoint", action='store_false', dest='load_checkpoint', help="Do not load checkpoint. If set, start training from scratch.")

    parser.add_argument("--num_epoch", type=int, default=config["epoch"], help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=config["batch_size"], help="Batch size for training.")
    parser.add_argument("--image_channels", type=int, default=config["image_channels"], help="Number of channels inputs.")
    parser.add_argument("--learning_rate", type=float, default=config["learning_rate"], help="Learning rate for Adam optimizer.")
    parser.add_argument("--lambda_cycle", type=int, default=config["lambda_cycle"], help="Weight for cycle consistency loss.")
    parser.add_argument("--lambda_identity", type=int, default=config["lambda_identity"], help="Weight for identity loss.")

    parser.add_argument("--ckpt_path", type=str, default=config["ckpt_path"], help="Path to checkpoint file. If load_checkpoint is set, this path will be used to load the checkpoint.")
    parser.add_argument("--summer_path", type=str, default=config["summer_path"], help="Path to summer dataset directory.")
    parser.add_argument("--winter_path", type=str, default=config["winter_path"], help="Path to winter dataset directory.")
    parser.add_argument("--folder_output", type=str, default=config["folder_output"], help="Path to Output results directory.")
    
    parser.add_argument("--display_step", type=int, default=config["display_step"], help="Interval of epochs to display loss and save examples.")
    
    parser.add_argument("--run_dev", action='store_true', help="Run in development mode if this flag is set. If not set, run in production mode.")
    parser.add_argument("--no_run_dev", action='store_false', dest='run_dev', help="Run in production mode if this flag is set. If not set, run in development mode.")

    args = parser.parse_args()

    config["epoch"] = args.num_epoch
    config["batch_size"] = args.batch_size

    config["image_channels"] = args.image_channels
    config["learning_rate"] = args.learning_rate
    config["lambda_cycle"] = args.lambda_cycle
    config["lambda_identity"] = args.lambda_identity    

    config["summer_path"] = args.summer_path
    config["winter_path"] = args.winter_path
    config["folder_output"] = args.folder_output

    config["display_step"] = args.display_step

    config["load_checkpoint"] = args.load_checkpoint if args.load_checkpoint is not None else config["load_checkpoint"]
    config["ckpt_path"] = args.ckpt_path
    
    config["run_dev"] = args.run_dev

    # do some training
    train(config)