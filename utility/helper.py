import torch

from config.core import config
from models.generator import Generator

def load_model_weights(checkpoint_path, model, device, prefix):
    """
    Load specific weights from a PyTorch Lightning checkpoint into a model.

    Parameters:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model instance to load weights into.
        prefix (str): The prefix in the checkpoint's state_dict keys to filter by and remove.

    Returns:
        model (torch.nn.Module): The model with loaded weights.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract and modify the state_dict keys to match the model's keys
    model_weights = {k.replace(f"{prefix}.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith(f"{prefix}.")}

    # Load the weights into the model
    model.load_state_dict(model_weights)

    return model

def load_latent_space(checkpoint_path):
    pass

def init_generator_model():
    """
    Initializes and returns the Generator model.

    Args:
        None.

    Returns:
        Generator: The initialized Generator model.
    """
    model = Generator(
        embed_size=config.EMBED_SIZE,
        num_classes=config.NUM_CLASSES,
        image_size=config.IMAGE_SIZE,
        features_generator=config.FEATURES_GENERATOR,
        input_dim=config.INPUT_Z_DIM,
        image_channel=config.IMAGE_CHANNEL
    )
    return model

def get_selected_value(label):
    """
    Get the selected value based on the display label.

    Args:
        label (str): The display label.

    Returns:
        int: The selected value corresponding to the display label.
    """
    # Get the selected value from the options mapping based on the display label.
    return config.OPTIONS_MAPPING[label]
