import torch
import gradio as gr


from config.core import config
# from models.lightning import ConditionalWGAN_GP

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model = ConditionalWGAN_GP.load_from_checkpoint(config.CHECKPOINT_PATH, map_location=DEVICE)

# Define the mapping between display labels and values
options_mapping = {
    "Boot": 0,
    "Sandal": 1,
    "Shoe": 2
}

# Create a list of display labels for the dropdown
options = list(options_mapping.keys())

# Function to get the selected value based on the display label
def get_selected_value(label):
    return options_mapping[label]

def inference(choice):
    z = torch.randn(1, config.Z_DIM, 1, 1).to(DEVICE)
    label = torch.tensor([get_selected_value(choice)], device=DEVICE)

    print(f"Choice: {choice} => {label}")
    print(z.shape)

demo = gr.Interface(
    fn=inference,
    inputs=gr.Dropdown(options, label="Select an option to Generates Images"),
    outputs=gr.Image(shape=(128, 128)),
    title="Shoe, Sandal, Boot - Conditional GAN",
    description="Conditional WGAN-GP",
)

demo.launch()