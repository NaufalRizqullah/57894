import torch
import torch.nn as nn

from .base import WSConv2d, ConvBlock, PixelNorm
from config.core import config


class Generator(nn.Module):
    def __init__(self,  embed_size=128, num_classes=3, image_size=128, features_generator=128, input_dim=128, image_channel=3):
        super().__init__()

        self.gen = nn.Sequential(
           self._block(input_dim + embed_size, features_generator*2, first_double_up=True),
           self._block(features_generator*2, features_generator*4, first_double_up=False, final_layer=False,),
           self._block(features_generator*4, features_generator*4, first_double_up=False, final_layer=False,),
           self._block(features_generator*4, features_generator*4, first_double_up=False, final_layer=False,),
           self._block(features_generator*4, features_generator*2, first_double_up=False, final_layer=False,),
           self._block(features_generator*2, features_generator, first_double_up=False, final_layer=False,),
           self._block(features_generator, image_channel, first_double_up=False, use_double=False, final_layer=True,),
        )
        
        self.image_size = image_size
        self.embed_size = embed_size
        
        self.embed = nn.Embedding(num_classes, embed_size)
        self.embed_linear = nn.Linear(embed_size, embed_size)

    def forward(self, noise, labels):
        embedding_label = self.embed(labels)
        linear_embedding_label = self.embed_linear(embedding_label).unsqueeze(2).unsqueeze(3)
        
        noise = noise.view(noise.size(0), noise.size(1), 1, 1)
        
        x = torch.cat([noise, linear_embedding_label], dim=1)
        
        return self.gen(x)


    def _block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
           first_double_up=False, use_double=True, final_layer=False):
        layers = []

        if not final_layer:
            layers.append(ConvBlock(in_channels, out_channels))
        else:
            layers.append(WSConv2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.Tanh())

        if use_double:
            if first_double_up:
                layers.append(nn.ConvTranspose2d(out_channels, out_channels, 4, 1, 0))
            else:
                layers.append(nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1))

            layers.append(PixelNorm())
            layers.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*layers)

def test():
    sample = torch.randn(1, config.INPUT_Z_DIM, 1, 1)
    label = torch.tensor([1])

    model = Generator(
                embed_size=config.EMBED_SIZE,
                num_classes=config.NUM_CLASSES,
                image_size=config.IMAGE_SIZE,
                features_generator=config.FEATURES_GENERATOR,
                input_dim=config.INPUT_Z_DIM,
            )

    preds = model(sample, label)
    print(preds.shape)