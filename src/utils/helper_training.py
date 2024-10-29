import torch
import torch.nn as nn
import cv2
import imageio
import os
import subprocess

def update_version_kaggle_dataset():
    # Make Metadata json
    subprocess.run(['kaggle', 'datasets', 'init'], check=True)

    # Write new metadata
    with open('/kaggle/working/dataset-metadata.json', 'w') as json_fid:
        json_fid.write(f'{{\n  "title": "Update Logs CycleGAN",\n  "id": "muhammadnaufal/cyclegan",\n  "licenses": [{{"name": "CC0-1.0"}}]}}')

    # Push new version
    subprocess.run(['kaggle', 'datasets', 'version', '-m', 'Updated Dataset', '--quiet', '--dir-mode', 'tar'], check=True)