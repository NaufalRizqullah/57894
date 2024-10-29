import torch
import torch.nn as nn

from src.features.base import Block

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            custom_stride = 1 if feature == features[-1] else 2
            layers.append(Block(in_channels, feature, custom_stride))
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1,
                      padding=1, padding_mode="reflect")
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        return torch.sigmoid(x)


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    preds = model(x)
    print(preds.shape)
