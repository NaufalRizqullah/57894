import torch
import numpy as np
import gradio as gr
import os

from PIL import Image
import torchvision.transforms as T
from config.core import config
from utility.helper import load_model_weights, init_generator_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = init_generator_model()
model = load_model_weights(
    model=model,
    checkpoint_path=config.CKPT_PATH,
    device=device,
    prefix="gen",
)

# Transformation
transform_face = T.Compose([
        T.CenterCrop(config.IMAGE_SIZE),
        T.Resize(config.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def inference(image: Image):
    # transforms the target image and add a batch dimension
    img = transform_face(image)
    img = img.unsqueeze(0)

    # Inference the image
    model.eval()
    with torch.inference_mode():
        c2f = model(img)
    
    c2f = c2f * 0.5 + 0.5 # Normalize from Tanh
    image_unflat = c2f.detach().cpu().squeeze(0)  # Remove batch dimension
    image = image_unflat.permute(1, 2, 0)  # Permute to (H, W, C)

    # Convert image to numpy array
    image_array = image.numpy()
    
    # Scale values to 0-255 range
    image_array = (image_array * 255).astype(np.uint8)
    
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array)
    
    return image

demo = gr.Interface(
    fn=inference,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(),
    title="Pix2Pix Face to Comic",
    description="A implementation Pix2Pix from Scratch Pytorch",
    examples=[f"data/examples/{i}" for i in os.listdir("data/examples") if i.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
)

demo.launch()
    