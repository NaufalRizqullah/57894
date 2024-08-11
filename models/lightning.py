import torch
import torch.optim as optim
import lightning as L
from .base import Discriminator, Generator

class ConditionalWGAN_GP(L.LightningModule):
    """Conditional WGAN-GP implementation using PyTorch Lightning.

    Attributes:
        image_size: Size of the generated images.
        critic_repeats: Number of critic iterations per generator iteration.
        c_lambda: Gradient penalty lambda hyperparameter.
        generator: The generator model.
        critic: The discriminator (critic) model.
        critic_losses: List to store critic loss values.
        generator_losses: List to store generator loss values.
        curr_step: The current training step.
        fixed_latent_space: Fixed latent vectors for generating consistent images.
        fixed_label: Fixed labels corresponding to the latent vectors.
    """

    def __init__(self, image_size, learning_rate, z_dim, embed_size, num_classes,
                 critic_repeats, feature_gen, feature_critic, c_lambda, beta_1,
                 beta_2, display_step):
        """Initializes the Conditional WGAN-GP model.

        Args:
            image_size: Size of the generated images.
            learning_rate: Learning rate for the optimizers.
            z_dim: Dimension of the latent space.
            embed_size: Size of the embedding for the labels.
            num_classes: Number of classes for the conditional generation.
            critic_repeats: Number of critic iterations per generator iteration.
            feature_gen: Number of features for the generator.
            feature_critic: Number of features for the critic.
            c_lambda: Gradient penalty lambda hyperparameter.
            beta_1: Beta1 parameter for the Adam optimizer.
            beta_2: Beta2 parameter for the Adam optimizer.
            display_step: Step interval for displaying generated images.
        """
        super().__init__()

        self.automatic_optimization = False

        self.image_size = image_size
        self.critic_repeats = critic_repeats
        self.c_lambda = c_lambda

        self.generator = Generator(
            embed_size=embed_size,
            num_classes=num_classes,
            image_size=image_size,
            features_generator=feature_gen,
            input_dim=z_dim,
        )

        self.critic = Discriminator(
            num_classes=num_classes,
            image_size=image_size,
            features_discriminator=feature_critic,
        )

        self.critic_losses = []
        self.generator_losses = []
        self.curr_step = 0

        self.fixed_latent_space = torch.randn(25, z_dim, 1, 1)
        self.fixed_label = torch.tensor([i % num_classes for i in range(25)])

        self.save_hyperparameters()

    def configure_optimizers(self):
        """Configures the optimizers for the generator and critic.

        Returns:
            A tuple of two Adam optimizers, one for the generator and one for the critic.
        """
        optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta_1, self.hparams.beta_2),
        )
        optimizer_c = optim.Adam(
            self.critic.parameters(),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta_1, self.hparams.beta_2),
        )

        return optimizer_g, optimizer_c

    def on_load_checkpoint(self, checkpoint):
        """Loads necessary variables from a checkpoint.

        Args:
            checkpoint: The checkpoint dictionary.
        """
        if self.current_epoch != 0:
            self.critic_losses = checkpoint['critic_losses']
            self.generator_losses = checkpoint['generator_losses']
            self.curr_step = checkpoint['curr_step']
            self.fixed_latent_space = checkpoint['fixed_latent_space']
            self.fixed_label = checkpoint['fixed_label']

    def on_save_checkpoint(self, checkpoint):
        """Saves necessary variables to a checkpoint.

        Args:
            checkpoint: The checkpoint dictionary.
        """
        checkpoint['critic_losses'] = self.critic_losses
        checkpoint['generator_losses'] = self.generator_losses
        checkpoint['curr_step'] = self.curr_step
        checkpoint['fixed_latent_space'] = self.fixed_latent_space
        checkpoint['fixed_label'] = self.fixed_label

    def forward(self, noise, labels):
        """Generates an image given noise and labels.

        Args:
            noise: Latent noise vector.
            labels: Class labels for conditional generation.

        Returns:
            Generated image tensor.
        """
        return self.generator(noise, labels)

    def predict_step(self, noise, labels):
        """Predicts an image given noise and labels.

        Args:
            noise: Latent noise vector.
            labels: Class labels for conditional generation.

        Returns:
            Generated image tensor.
        """
        return self.generator(noise, labels)
