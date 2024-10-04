import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim

from models.generator import Generator
from models.discriminator import Discriminator
from utility.helper import save_some_examples


class Pix2Pix(L.LightningModule):
    def __init__(self, in_channels, learning_rate, l1_lambda, features_generator, features_discriminator, display_step):
        super().__init__()

        self.automatic_optimization = False

        self.gen = Generator(
            in_channels=in_channels,
            features=features_generator
        )
        self.disc = Discriminator(
            in_channels=in_channels,
            features=features_discriminator
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.discriminator_losses = []
        self.generator_losses = []
        self.curr_step = 0

        self.bce = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

        self.save_hyperparameters()


    def configure_optimizers(self):
        optimizer_G = optim.Adam(self.gen.parameters(), lr=self.hparams.learning_rate, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.disc.parameters(), lr=self.hparams.learning_rate, betas=(0.5, 0.999))

        return optimizer_G, optimizer_D

    def on_load_checkpoint(self, checkpoint):
        # List of keys that you expect to load from the checkpoint
        keys_to_load = ['discriminator_losses', 'generator_losses', 'curr_step']

        # Iterate over the keys and load them if they exist in the checkpoint
        for key in keys_to_load:
            if key in checkpoint:
                setattr(self, key, checkpoint[key])

    def on_save_checkpoint(self, checkpoint):
        # Save the current state of the model
        checkpoint['discriminator_losses'] = self.discriminator_losses
        checkpoint['generator_losses'] = self.generator_losses
        checkpoint['curr_step'] = self.curr_step

    def training_step(self, batch, batch_idx):
        # Get the Optimizers
        opt_generator, opt_discriminator = self.optimizers()
        
        X, y = batch

        # Train Discriminator
        y_fake = self.gen(X)
        D_real = self.disc(X, y)
        D_fake = self.disc(X, y_fake.detach())

        D_real_loss = self.loss_fn(D_real, torch.ones_like(D_real))
        D_fake_loss = self.loss_fn(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        opt_discriminator.zero_grad()
        self.manual_backward(D_loss)
        opt_discriminator.step()

        self.log("D_loss", D_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.discriminator_losses.append(D_loss.item())

        # Train Generator
        D_fake = self.disc(X, y_fake)
        G_fake_loss = self.bce(D_fake, torch.ones_like(D_fake))

        L1 = self.l1_loss(y_fake, y) * self.hparams.l1_lambda
        G_loss = G_fake_loss + L1

        opt_generator.zero_grad()
        self.manual_backward(G_loss)
        opt_generator.step()

        self.log("G_loss", G_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.generator_losses.append(G_loss.item())
        
        self.log("Current_Step", self.curr_step, on_step=False, on_epoch=True, prog_bar=True)
        
        # Visualize
        if self.curr_step % self.hparams.display_step == 0 and self.curr_step > 0:
            save_some_examples(self.gen, batch, self.current_epoch)
        
        self.curr_step += 1

