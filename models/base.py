import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """Discriminator model for Conditional GAN.

    Args:
        num_classes (int): Number of classes in the dataset.
        image_size (int): Size of the input images (assumes square images).
        features_discriminator (int): Number of feature maps in the first layer of the discriminator.
        image_channel (int): Number of channels in the input image.

    Attributes:
        disc (nn.Sequential): The sequential layers that define the discriminator.
        embed (nn.Embedding): Embedding layer to encode labels into image-like format.
    """

    def __init__(self, num_classes=3, image_size=128, features_discriminator=128, image_channel=3):
        super().__init__()
        
        self.num_classes = num_classes
        self.image_size = image_size
        
        label_channel = 1
        
        self.disc = nn.Sequential(
            self._block_discriminator(image_channel + label_channel, features_discriminator, kernel_size=4, stride=2, padding=1),
            self._block_discriminator(features_discriminator, features_discriminator, kernel_size=4, stride=2, padding=1),
            self._block_discriminator(features_discriminator, features_discriminator * 2, kernel_size=4, stride=2, padding=1),
            self._block_discriminator(features_discriminator * 2, features_discriminator * 4, kernel_size=4, stride=2, padding=1),
            self._block_discriminator(features_discriminator * 4, features_discriminator * 4, kernel_size=4, stride=2, padding=1),
            self._block_discriminator(features_discriminator * 4, 1, kernel_size=4, stride=1, padding=0, final_layer=True)
        )
        
        self.embed = nn.Embedding(num_classes, image_size * image_size)

    def forward(self, image, label):
        """Forward pass for the discriminator.

        Args:
            image (torch.Tensor): Batch of input images.
            label (torch.Tensor): Corresponding labels for the images.

        Returns:
            torch.Tensor: Discriminator output.
        """
        # Embed label into an image-like format
        embedding = self.embed(label)
        embedding = embedding.view(
            label.shape[0],
            1,
            self.image_size,
            self.image_size
        )  # Reshape into 1-channel image
        
        data = torch.cat([image, embedding], dim=1)  # Concatenate image with the label channel
        
        x = self.disc(data)
        
        return x.view(len(x), -1)

    def _block_discriminator(self, input_channels, output_channels, kernel_size=3, stride=2, padding=0, final_layer=False):
        """Creates a convolutional block for the discriminator.

        Args:
            input_channels (int): Number of input channels for the convolutional layer.
            output_channels (int): Number of output channels for the convolutional layer.
            kernel_size (int): Size of the kernel for the convolutional layer.
            stride (int): Stride of the convolutional layer.
            padding (int): Padding for the convolutional layer.
            final_layer (bool): If True, this is the final layer, which doesn't include normalization or activation.

        Returns:
            nn.Sequential: Sequential block for the discriminator.
        """
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.InstanceNorm2d(output_channels, affine=True),
                nn.LeakyReLU(0.2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
            )

class Generator(nn.Module):
    """Generator model for Conditional GAN.

    Args:
        embed_size (int): Size of the embedding vector for the labels.
        num_classes (int): Number of classes in the dataset.
        image_size (int): Size of the output images (assumes square images).
        features_generator (int): Number of feature maps in the first layer of the generator.
        input_dim (int): Dimensionality of the noise vector.
        image_channel (int): Number of channels in the output image.

    Attributes:
        gen (nn.Sequential): The sequential layers that define the generator.
        embed (nn.Embedding): Embedding layer to encode labels.
    """

    def __init__(self, embed_size=128, num_classes=3, image_size=128, features_generator=128, input_dim=128, image_channel=3):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
           self._block(input_dim + embed_size, features_generator * 2, first_double_up=True),
           self._block(features_generator * 2, features_generator * 4, first_double_up=False, final_layer=False),
           self._block(features_generator * 4, features_generator * 4, first_double_up=False, final_layer=False),
           self._block(features_generator * 4, features_generator * 4, first_double_up=False, final_layer=False),
           self._block(features_generator * 4, features_generator * 2, first_double_up=False, final_layer=False),
           self._block(features_generator * 2, features_generator, first_double_up=False, final_layer=False),
           self._block(features_generator, image_channel, first_double_up=False, use_double=False, final_layer=True),
        )
        
        self.image_size = image_size
        self.embed_size = embed_size
        
        self.embed = nn.Embedding(num_classes, embed_size)

    def forward(self, noise, labels):
        """Forward pass for the generator.

        Args:
            noise (torch.Tensor): Batch of input noise vectors.
            labels (torch.Tensor): Corresponding labels for the noise vectors.

        Returns:
            torch.Tensor: Generated images.
        """
        embedding_label = self.embed(labels).unsqueeze(2).unsqueeze(3)  # Reshape to (batch_size, embed_size, 1, 1)
        
        noise = noise.view(noise.size(0), noise.size(1), 1, 1)  # Reshape to (batch_size, z_dim, 1, 1)

        x = torch.cat([noise, embedding_label], dim=1)
        
        return self.gen(x)

    def _block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
               first_double_up=False, use_double=True, final_layer=False):
        """Creates a convolutional block for the generator.

        Args:
            in_channels (int): Number of input channels for the convolutional layer.
            out_channels (int): Number of output channels for the convolutional layer.
            kernel_size (int): Size of the kernel for the convolutional layer.
            stride (int): Stride of the convolutional layer.
            padding (int): Padding for the convolutional layer.
            first_double_up (bool): If True, the first layer uses a different upsampling strategy.
            use_double (bool): If True, the block includes an upsampling layer.
            final_layer (bool): If True, this is the final layer, which uses Tanh activation.

        Returns:
            nn.Sequential: Sequential block for the generator.
        """
        layers = []

        if not final_layer:
            # Add first convolutional layer
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))

            # Add second convolutional layer
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.Tanh())

        if use_double:
            if first_double_up:
                layers.append(nn.ConvTranspose2d(out_channels, out_channels, 4, 1, 0))
            else:
                layers.append(nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1))
        
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*layers)
