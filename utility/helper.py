import torch
import torch.nn as nn
import cv2
import imageio
import os
import matplotlib.pyplot as plt

from config.core import config
from models.generator import Generator
from PIL import Image
from torchvision.utils import make_grid


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

def plot_images_from_tensor(image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True, save_path=None):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close()

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

class PadToSquare:
    """Pad an image to a square of the given size with a white background.

    Args:
        size (int): The target size for the output image.
    """
    
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        """Pad the input image to the target size with a white background.

        Args:
            img (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The padded image.
        """
        # Create a white canvas
        white_canvas = Image.new('RGB', (self.size, self.size), (255, 255, 255))

        # Calculate the position to paste the image onto the white canvas
        left = (self.size - img.width) // 2
        top = (self.size - img.height) // 2

        # Paste the image onto the canvas
        white_canvas.paste(img, (left, top))

        return white_canvas
    