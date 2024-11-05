import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L


from src.features.discriminator import Discriminator
from src.features.generator import Generator
from src.visualization.visualize import save_cycle_consistency_examples

class CycleGAN(L.LightningModule):
    def __init__(self, image_channels, learning_rate, lambda_cycle, lambda_identity, folder_output, display_step):
        super().__init__()
        # to note: Horse/H = X, Zebra/Z = Y
        self.automatic_optimization = False
        
        self.discriminator_H = Discriminator(in_channels=image_channels)
        self.discriminator_Z = Discriminator(in_channels=image_channels)
        
        self.generator_H = Generator(img_channels=image_channels)
        self.generator_Z = Generator(img_channels=image_channels)
        
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        
        self.curr_step = 0
        self.discriminator_losses = []
        self.generator_losses = []
        
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        optimizer_discriminator = optim.Adam(
            list(self.discriminator_H.parameters()) + list(self.discriminator_Z.parameters()),
            lr=self.hparams.learning_rate,
            betas=(0.5, 0.999)
        )
        optimizer_generator = optim.Adam(
            list(self.generator_H.parameters()) + list(self.generator_Z.parameters()),
            lr=self.hparams.learning_rate,
            betas=(0.5, 0.999)
        )

        return optimizer_discriminator, optimizer_generator
    
    def on_load_checkpoint(self, checkpoint):
        # List of keys that you expect to load from the checkpoint
        keys_to_load = [
            'curr_step',
            'generator_losses',
            'discriminator_losses',
        ]

        # Iterate over the keys and load them if they exist in the checkpoint
        for key in keys_to_load:
            if key in checkpoint:
                setattr(self, key, checkpoint[key])
    
    def on_save_checkpoint(self, checkpoint):
        # Save necessary variable to checkpoint
        checkpoint['curr_step'] = self.curr_step
        checkpoint['generator_losses'] = self.generator_losses
        checkpoint['discriminator_losses'] = self.discriminator_losses
    
    def training_step(self, batch, batch_idx):
        # Get the Optimizers
        opt_discriminator, opt_generator = self.optimizers()
        
        # Get X and Y
        X, Y = batch
        
        ##################################
        # Train Discriminator X and Y ####
        ##################################
        # to note: Horse/H = X, Zebra/Z = Y
        fake_X = self.generator_H(Y)
        
        discriminator_X_real = self.discriminator_H(X)
        discriminator_X_fake = self.discriminator_H(fake_X.detach())
        discriminator_X_real_loss = self.mse(discriminator_X_real, torch.ones_like(discriminator_X_real))
        discriminator_X_fake_loss = self.mse(discriminator_X_fake, torch.zeros_like(discriminator_X_fake))
        discriminator_X_loss = discriminator_X_real_loss + discriminator_X_fake_loss
        
        fake_Y = self.generator_Z(X)
        
        discriminator_Y_real = self.discriminator_Z(Y)
        discriminator_Y_fake = self.discriminator_Z(fake_Y.detach())
        discriminator_Y_real_loss = self.mse(discriminator_Y_real, torch.ones_like(discriminator_Y_real))
        discriminator_Y_fake_loss = self.mse(discriminator_Y_fake, torch.zeros_like(discriminator_Y_fake))
        discriminator_Y_loss = discriminator_Y_real_loss + discriminator_Y_fake_loss
        
        # Discriminator Loss
        discriminator_loss = (discriminator_X_loss + discriminator_Y_loss) / 2
        
        opt_discriminator.zero_grad()
        self.manual_backward(discriminator_loss)
        opt_discriminator.step()
        
        ##################################
        # Train Generator X and Y ########
        ##################################
        # to note: Horse/H = X, Zebra/Z = Y
        
        discriminator_X_fake = self.discriminator_H(fake_X)
        discriminator_Y_fake = self.discriminator_Z(fake_Y)
        
        loss_generator_X = self.mse(discriminator_X_fake, torch.ones_like(discriminator_X_fake))
        loss_generator_Y = self.mse(discriminator_Y_fake, torch.ones_like(discriminator_Y_fake))
        
        # Cycle Loss
        cycle_Y = self.generator_Z(fake_X)
        cycle_X = self.generator_H(fake_Y)
        
        cycle_Y_loss = self.l1(Y, cycle_Y)
        cycle_X_loss = self.l1(X, cycle_X)
        
        # Identity Loss (not used)
        # identity_Y = self.generator_Z(Y)
        # identity_X = self.generator_H(X)
        
        # identity_Y_loss = self.l1(Y, identity_Y)
        # identity_X_loss = self.l1(X, identity_X)
        
        # Add All Together Generator Loss
        generator_loss = (
            loss_generator_X + loss_generator_Y
            + cycle_Y_loss * self.hparams.lambda_cycle
            + cycle_X_loss * self.hparams.lambda_cycle
            
            # loss identity only using on other dataset, like painting, color changing etc (need compare the same image, and only changin like color etc)
            # + identity_Y_loss * self.hparams.lambda_identity
            # + identity_X_loss * self.hparams.lambda_identity
        )
        
        opt_generator.zero_grad()
        self.manual_backward(generator_loss)
        opt_generator.step()
        
        ##############################
        # Logging ####################
        ##############################
        self.log("generator_loss", generator_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("discriminator_loss", discriminator_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        
        self.generator_losses.append(generator_loss.item())
        self.discriminator_losses.append(discriminator_loss.item())

        
        # Visualize the training
        if self.curr_step % self.hparams.display_step == 0 and self.curr_step > 0:
            save_cycle_consistency_examples(
                self.generator_H,
                self.generator_Z,
                batch,
                self.current_epoch,
                self.hparams.folder_output,
            )
        
        self.curr_step += 1