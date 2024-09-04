import torch
import torch.nn as nn

from .base import WSConv2d, ConvBlock, PixelNorm


class Generator(nn.Module):
    def __init__(self, embed_size=128, num_classes=3, image_size=128, features_generator=128, input_dim=128, image_channel=3):
        super().__init__()

        self.gen = nn.Sequential(
            self._block(input_dim + embed_size, features_generator * 2, first_double_up=True),
            self._block(features_generator * 2, features_generator * 4, first_double_up=False, final_layer=False, ),
            self._block(features_generator * 4, features_generator * 4, first_double_up=False, final_layer=False, ),
            self._block(features_generator * 4, features_generator * 4, first_double_up=False, final_layer=False, ),
            self._block(features_generator * 4, features_generator * 2, first_double_up=False, final_layer=False, ),
            self._block(features_generator * 2, features_generator, first_double_up=False, final_layer=False, ),
            self._block(features_generator, image_channel, first_double_up=False, use_double=False, final_layer=True, ),
        )

        self.image_size = image_size
        self.embed_size = embed_size

        self.embed = nn.Embedding(num_classes, embed_size)

    def forward(self, noise, labels):
        embedding_label = self.embed(labels).unsqueeze(2).unsqueeze(
            3)  # Add height and width channel; N x Noise_dim x 1 x 1

        # Noise is 4 channel, or 2 channel. later will decide
        noise = noise.view(noise.size(0), noise.size(1), 1, 1)  # Reshape to (batch_size, z_dim, 1, 1)

        x = torch.cat([noise, embedding_label], dim=1)

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

