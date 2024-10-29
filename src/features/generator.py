import torch
import torch.nn as nn

from src.features.base import ConvBlock, ResidualBlock


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residual=9):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7,
                        stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )

        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features * 2,
                            kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features * 2, num_features * 4,
                            kernel_size=3, stride=2, padding=1),
            ]
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residual)]
        )

        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features * 4, num_features * 2, down=False,
                            kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features * 2, num_features * 1, down=False,
                            kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )

        self.last = nn.Conv2d(num_features * 1, img_channels,
                                kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        # initial layer
        x = self.initial(x)

        # 2 layer down-sampling
        for layer in self.down_blocks:
            x = layer(x)

        # residual block
        x = self.residual_blocks(x)

        # 2 layer up-sampling
        for layer in self.up_blocks:
            x = layer(x)

        # last layer (convert to rgb)
        x = self.last(x)

        return torch.tanh(x)



if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = Generator(3)
    preds = model(x)
    print(preds.shape)