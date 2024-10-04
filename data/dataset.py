import os
from PIL import Image

from torch.utils.data import Dataset

class FaceToComicDataset(Dataset):
    def __init__(self, face_path, comic_path, transform_face=None, transform_comic=None):
        super().__init__()
        self.face_dir = face_path
        self.comic_dir = comic_path
        
        self.face_list_files = os.listdir(self.face_dir)
        self.comic_list_files = os.listdir(self.comic_dir)
        
        # Create a dictionary for quick lookup of comic files
        self.comic_dict = {comic_file: idx for idx, comic_file in enumerate(self.comic_list_files)}
        
        # Filter out files that don't have a corresponding pair (find only have pair)
        self.face_list_files = [f for f in self.face_list_files if f in self.comic_list_files]
        
        self.transform_face = transform_face
        self.transform_comic = transform_comic

    def __getitem__(self, index):
        face_file = self.face_list_files[index]
        comic_file = self.comic_list_files[self.comic_dict[face_file]]

        face_image = Image.open(os.path.join(self.face_dir, face_file))
        comic_image = Image.open(os.path.join(self.comic_dir, comic_file))

        if self.transform_face:
            face_image = self.transform_face(face_image)
        if self.transform_comic:
            comic_image = self.transform_comic(comic_image)

        return face_image, comic_image

    def __len__(self):
        return len(self.face_list_files)
