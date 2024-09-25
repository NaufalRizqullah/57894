import torch
import torch.nn as nn

from .base import WSConv2d, ConvBlock
from config.core import config


class Discriminator(nn.Module):
    def __init__(self, num_classes=3, embed_size=128, image_size=128, features_discriminator=128, image_channel=3, label_channel=3):
        super().__init__()
        
        self.num_classes = num_classes
        self.image_size = image_size
        
        self.embed_size = embed_size
        self.label_channel = label_channel
        
        self.disc = nn.Sequential(
            self._block_discriminator(image_channel + label_channel, features_discriminator, kernel_size=4, stride=2, padding=1),
            self._block_discriminator(features_discriminator, features_discriminator, kernel_size=4, stride=2, padding=1),
            self._block_discriminator(features_discriminator, features_discriminator * 2, kernel_size=4, stride=2, padding=1),
            self._block_discriminator(features_discriminator * 2, features_discriminator * 4, kernel_size=4, stride=2, padding=1),
            self._block_discriminator(features_discriminator * 4, features_discriminator  *4 , kernel_size=4, stride=2, padding=1),
            self._block_discriminator(features_discriminator * 4, 1, kernel_size=4, stride=1, padding=0, final_layer=True)
        )
        
        self.embed = nn.Embedding(num_classes, embed_size)
        self.embed_linear = nn.Linear(embed_size, label_channel*image_size*image_size)

    def forward(self, image, label):
        embedding = self.embed(label)
        
        linear_embedding = self.embed_linear(embedding)
        
        embedding_layer = linear_embedding.view(
            label.shape[0],
            self.label_channel,
            self.image_size,
            self.image_size
        )
        
        data = torch.cat([image, embedding_layer], dim=1)
        
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
                WSConv2d(output_channels, output_channels, kernel_size, stride, padding),
            )
        else:
            return WSConv2d(input_channels, output_channels, kernel_size, stride, padding)
        
def test():
    sample = torch.randn(1, 3, 128, 128)
    label = torch.tensor([1])

    model = Discriminator(
        num_classes=config.NUM_CLASSES,
        embed_size=config.EMBED_SIZE,
        image_size=config.IMAGE_SIZE,
        features_discriminator=config.FEATURES_DISCRIMINATOR,
        image_channel=config.IMAGE_CHANNEL,
        label_channel=config.LABEL_CHANNEL
    )

    preds = model(sample, label)
    print(preds.shape)