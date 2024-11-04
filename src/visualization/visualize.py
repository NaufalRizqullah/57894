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
        save_image(y_fake, folder_path + f"/output_{epoch}.png", nrow=5)  # Save Generated Image

        # Create 3x5 grid for input images
        save_image(x * 0.5 + 0.5, folder_path + f"/input_{epoch}.png", nrow=5)  # Save Real Image

        # Create 3x5 grid for input images
        save_image(y * 0.5 + 0.5, folder_path + f"/label_{epoch}.png", nrow=5)  # Save Label Image


    generator_model.train()

def save_cycle_consistency_examples(generator_G, generator_F, batch, epoch, folder_path, num_images=15):
    """
    Save examples of the CycleGAN's cycle consistency output.
    
    Parameters:
        generator_G (nn.Module): The generator model G (e.g., input to output domain).
        generator_F (nn.Module): The generator model F (e.g., output to input domain).
        batch (tuple): The batch of input images (x, y).
        epoch (int): The current epoch number.
        folder_path (str): The directory where images will be saved.
        num_images (int): The number of images to process and save.
    """
    
    os.makedirs(folder_path, exist_ok=True)
    
    x, _ = batch  # Unpack the batch; y can be ignored for this example.
    
    x = x[:num_images]  # Limit to the specified number of images
    
    generator_G.eval()
    generator_F.eval()

    with torch.inference_mode():
        # Generate G(x) and F(G(x))
        G_x = generator_G(x)
        F_G_x = generator_F(G_x)

        # Unnormalize images for visualization (assuming the images were normalized with mean=0.5 and std=0.5)
        x = x * 0.5 + 0.5
        G_x = G_x * 0.5 + 0.5
        F_G_x = F_G_x * 0.5 + 0.5

        # Concatenate the input, output, and reconstruction along the batch dimension
        combined = torch.cat((x, G_x, F_G_x), dim=0)

        # Save the combined grid of images (3 rows: input, output, reconstruction)
        save_image(combined, folder_path + f"/cycle_consistency_{epoch}.png", nrow=num_images)

    generator_G.train()
    generator_F.train()