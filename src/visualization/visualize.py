import torch
import os

from torchvision.utils import save_image

def save_some_examples(generator_model, batch, epoch, folder_path, num_images=15):
    """
    Save some examples of the generator's output.

    Parameters:
        generator_model (nn.Module): The generator model.
        batch (tuple): The batch of input and target images as a tuple of tensors.
        epoch (int): The current epoch.
        folder_path (str): The folder path to save the examples to. Defaults to config.PATH_OUTPUT.
        num_images (int): The number of images to save. Defaults to 15.
    """
    
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    x, y = batch  # Unpack the batch
    
    # Limit the number of images to the specified num_images
    x = x[:num_images]
    y = y[:num_images]

    generator_model.eval()

    with torch.inference_mode():
        y_fake = generator_model(x)
        y_fake = y_fake * 0.5 + 0.5  # Remove normalization by tanh

        # Create 3x5 grid for generated images
        save_image(y_fake, folder_path + f"/y_gen_{epoch}.png", nrow=5)  # Save Generated Image

        # Create 3x5 grid for input images
        save_image(x * 0.5 + 0.5, folder_path + f"/input_{epoch}.png", nrow=5)  # Save Real Image

    generator_model.train()