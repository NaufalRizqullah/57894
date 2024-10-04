import torch
import torch.nn as nn
import cv2
import imageio
import os
import subprocess

from config.core import config
from torchvision.utils import save_image


def save_some_examples(generator_model, batch, epoch, folder_path=config.PATH_OUTPUT, num_images=15):
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

def update_version_kaggle_dataset():
    # Make Metadata json
    subprocess.run(['kaggle', 'datasets', 'init'], check=True)

    # Write new metadata
    with open('/kaggle/working/dataset-metadata.json', 'w') as json_fid:
        json_fid.write(f'{{\n  "title": "Update Logs Pix2Pix",\n  "id": "muhammadnaufal/pix2pix",\n  "licenses": [{{"name": "CC0-1.0"}}]}}')

    # Push new version
    subprocess.run(['kaggle', 'datasets', 'version', '-m', 'Updated Dataset', '--quiet', '--dir-mode', 'tar'], check=True)



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


def initialize_weights(model):
    """
    Initializes the weights of a model using a normal distribution.

    Args:
        model: The model to be initialized.

    Returns:
        None
    """
    
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def create_video(image_folder, video_name, fps, appearance_duration=None):
    """
    Creates a video from a sequence of images with customizable appearance duration.

    Args:
        image_folder (str): The path to the folder containing the images.
        video_name (str): The name of the output video file.
        fps (int): The frames per second of the video.
        appearance_duration (int, optional): The desired appearance duration for each image in milliseconds.
            If None, the default duration based on frame rate is used.

    Example:
        image_folder = '/path/to/image/folder' \n
        video_name = 'output_video.mp4' \n
        fps = 12 \n
        appearance_duration = 200  # Appearance duration of 200ms for each image \n
        
        create_video(image_folder, video_name, fps, appearance_duration)
    """

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    # Sort the image files based on the step number
    image_files = sorted(image_files, key=lambda x: int(x.split('-')[1].split('.')[0]))

    # Load the first image to get the video size
    image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, layers = image.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the video codec
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Write each image to the video with customizable appearance duration
    for image_file in image_files:
        image = cv2.imread(os.path.join(image_folder, image_file))
        video.write(image)

        if appearance_duration is not None:
            # Calculate the number of frames for the desired appearance duration
            num_frames = appearance_duration * fps // 1000
            for _ in range(num_frames):
                video.write(image)

    # Release the video writer
    video.release()

def create_gif(image_folder, gif_name, fps, appearance_duration=None):
    """
    Creates a GIF from a sequence of images sorted by step number, with customizable appearance duration.

    Args:
        image_folder (str): The path to the folder containing the images.
        gif_name (str): The name of the output GIF file.
        fps (int): The frames per second of the GIF.
        appearance_duration (int, optional): The desired appearance duration for each image in milliseconds.
            If None, the default duration based on frame rate is used.

    Example:
        image_folder = '/path/to/image/folder'
        gif_name = 'output_animation.gif'
        fps = 12
        appearance_duration = 300  # Appearance duration of 300ms for each image

        create_gif(image_folder, gif_name, fps, appearance_duration)
    """

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    # Sort the image files based on the step number
    image_files = sorted(image_files, key=lambda x: int(x.split('-')[1].split('.')[0]))

    # Load the images into a list
    images = []
    for file in image_files:
        images.append(imageio.imread(os.path.join(image_folder, file)))

    # Create a list to store the repeated images
    repeated_images = []

    # Repeat each image for the desired duration
    if appearance_duration is not None:
        for image in images:
            repeated_images.extend([image] * (appearance_duration * fps // 1000))
    else:
        repeated_images = images  # Default appearance duration (based on fps)

    # Save the repeated images as a GIF
    imageio.mimsave(gif_name, repeated_images, fps=fps)
