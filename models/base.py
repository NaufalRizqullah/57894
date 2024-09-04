import torch
import torch.nn as nn


class WSConv2d(nn.Module):
    """
    A 2D convolutional layer with weight scaling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolving kernel. Default is 3.
        stride (int, optional): Stride of the convolution. Default is 1.
        padding (int, optional): Zero-padding added to both sides of the input. Default is 1.
        gain (float, optional): Gain factor for weight scaling. Default is 2.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # Initialize Conv Layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Forward pass of the WSConv2d layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying convolution, weight scaling, and bias addition.
        """
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    """
    Pixel normalization layer.

    Args:
        eps (float, optional): Small value to avoid division by zero. Default is 1e-8.
    """

    def __init__(self, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = eps

    def forward(self, x):
        """
        Forward pass of the PixelNorm layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    """
    A block of two convolutional layers, with optional pixel normalization and LeakyReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        use_pixelnorm (bool, optional): Whether to apply pixel normalization. Default is True.
    """

    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        """
        Forward pass of the ConvBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after two convolutional layers, optional pixel normalization, and LeakyReLU activation.
        """
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x

        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x
