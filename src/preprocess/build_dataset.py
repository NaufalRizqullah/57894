import os
import random
import numpy as np

from PIL import Image

from torch.utils.data import Dataset

class YosemiteDataset(Dataset):
    """Custom Dataset for loading paired summer and winter images.

    This dataset loads images from two folders: one containing summer images 
    and another containing winter images. The summer dataset typically has more 
    images than the winter dataset, so if the index exceeds the number of winter 
    images, a random winter image is loaded to match the length of the summer dataset.

    Attributes:
        root_summer (str): Path to the directory containing summer images.
        root_winter (str): Path to the directory containing winter images.
        transforms (callable, optional): A function/transform to apply to the images.
        files_summer (list): List of filenames in the summer directory.
        files_winter (list): List of filenames in the winter directory.
        summer_length (int): Number of images in the summer directory.
        winter_length (int): Number of images in the winter directory.
        max_length (int): Maximum length between the summer and winter datasets.
    """

    def __init__(self, summer_path, winter_path, transforms=None):
        """
        Args:
            summer_path (str): Path to the directory containing summer images.
            winter_path (str): Path to the directory containing winter images.
            transforms (callable, optional): A function/transform to apply to the images.
        """
        super().__init__()
        
        self.root_summer = summer_path
        self.root_winter = winter_path
        self.transforms = transforms
        
        # Get list of files in each directory
        self.files_summer = os.listdir(self.root_summer)
        self.files_winter = os.listdir(self.root_winter)
        
        self.summer_length = len(self.files_summer)
        self.winter_length = len(self.files_winter)
        
        # Set the maximum length to the length of the larger dataset
        self.max_length = max(self.summer_length, self.winter_length)
    
    def __getitem__(self, index):
        """Gets the summer and winter image pair at the specified index.

        If the index exceeds the number of images in either the summer or winter dataset,
        a random image from the respective dataset is selected to prevent out-of-bounds errors.

        Args:
            index (int): Index to retrieve the image pair.

        Returns:
            tuple: A tuple containing the summer image and the corresponding 
            winter image (or random images if the index is out of bounds).
        """
        # Load summer image, using a random index if out of bounds
        if index < self.summer_length:
            summer_image_path = os.path.join(self.root_summer, self.files_summer[index])
        else:
            random_index = random.randint(0, self.summer_length - 1)
            summer_image_path = os.path.join(self.root_summer, self.files_summer[random_index])

        summer_image = Image.open(summer_image_path).convert('RGB')

        # Load winter image, using a random index if out of bounds
        if index < self.winter_length:
            winter_image_path = os.path.join(self.root_winter, self.files_winter[index])
        else:
            random_index = random.randint(0, self.winter_length - 1)
            winter_image_path = os.path.join(self.root_winter, self.files_winter[random_index])

        winter_image = Image.open(winter_image_path).convert('RGB')
        
        # Convert images from PIL to NumPy arrays (cause Albumanetation accept np.arrays)
        summer_image = np.array(summer_image)
        winter_image = np.array(winter_image)

        # Apply Albumentations transforms if provided
        if self.transforms:
            augmentations = self.transforms(image=summer_image, image0=winter_image)
            summer_image = augmentations["image"]
            winter_image = augmentations["image0"]

        return summer_image, winter_image

    
    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            int: The maximum length between the summer and winter datasets.
        """
        return self.max_length