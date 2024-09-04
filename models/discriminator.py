import torch
import torch.nn as nn

from .base import WSConv2d, ConvBlock


class Discriminator(nn.Module):
    def __init__(self, num_classes=3, image_size=128, features_discriminator=128, image_channel=3):
        super().__init__()

        self.num_classes = num_classes
        self.image_size = image_size
        label_channel = 1

        self.disc = nn.Sequential(
            self._block_discriminator(image_channel + label_channel, features_discriminator, kernel_size=4, stride=2,
                                      padding=1),
            self._block_discriminator(features_discriminator, features_discriminator, kernel_size=4, stride=2,
                                      padding=1),
            self._block_discriminator(features_discriminator, features_discriminator * 2, kernel_size=4, stride=2,
                                      padding=1),
            self._block_discriminator(features_discriminator * 2, features_discriminator * 4, kernel_size=4, stride=2,
                                      padding=1),
            self._block_discriminator(features_discriminator * 4, features_discriminator * 4, kernel_size=4, stride=2,
                                      padding=1),
            self._block_discriminator(features_discriminator * 4, 1, kernel_size=4, stride=1, padding=0,
                                      final_layer=True)
        )

        self.embed = nn.Embedding(num_classes, image_size * image_size)

    def forward(self, image, label):
        embedding = self.embed(label)
        embedding = embedding.view(
            label.shape[0],
            1,
            self.image_size,
            self.image_size
        )

        data = torch.cat([image, embedding], dim=1)

        x = self.disc(data)

        return x.view(len(x), -1)

    def _block_discriminator(
            self,
            input_channels,
            output_channels,
            kernel_size=3,
            stride=2,
            padding=0,
            final_layer=False
    ):
        if not final_layer:
            return nn.Sequential(
                ConvBlock(input_channels, output_channels),
                WSConv2d(output_channels, output_channels, kernel_size, stride, padding)
            )
        else:
            return WSConv2d(input_channels, output_channels, kernel_size, stride, padding)


