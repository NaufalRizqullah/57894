import torch
from PIL import Image
import numpy as np
import gradio as gr

from config.core import config
from utility.helper import load_model_weights, init_generator_model, get_selected_value

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = init_generator_model()
model = load_model_weights(config.CKPT_PATH, model, DEVICE, "generator")
model.eval()

def inference(choice):
    z = torch.randn(1, config.INPUT_Z_DIM, 1, 1).to(DEVICE)
    label = torch.tensor([get_selected_value(choice)], device=DEVICE)

    image_tensor = model(z, label)

    image_tensor = (image_tensor + 1) / 2  # Shift and scale to 0 to 1
    image_unflat = image_tensor.detach().cpu().squeeze(0)  # Remove batch dimension
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
    inputs=gr.Dropdown(choices=list(config.OPTIONS_MAPPING.keys()), label="Select an option to Generates Images"),
    outputs=gr.Image(),
    title="Shoe, Sandal, Boot - Conditional GAN",
    description="Conditional WGAN-GP",
)

demo.launch()