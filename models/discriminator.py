import torch
import torch.nn as nn

from models.base import BlockCNN


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], kernel_size=4, activation_slope=0.2, ):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(activation_slope),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                BlockCNN(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=kernel_size, stride=1, padding=1, padding_mode="reflect"
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)
        
def test():
    # Test Case for Discriminator Model
    x = torch.randn((1, 3, 256, 256))
    disc = Discriminator()
    print(f"Discriminator Output Shape: {disc(x, x).shape}")