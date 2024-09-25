import torch
import lightning as L
import torch.optim as optim

from models.generator import Generator
from models.discriminator import Discriminator
from utility.helper import initialize_weights, plot_images_from_tensor
from utility.wgan_gp import gradient_penalty, calculate_generator_loss, calculate_critic_loss



class ConditionalWGAN_GP(L.LightningModule):
    def __init__(self, image_channel, label_channel, image_size, learning_rate, z_dim, embed_size, num_classes, critic_repeats, feature_gen, feature_critic, c_lambda, beta_1, beta_2, display_step):
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
            embed_size=embed_size,
            image_size=image_size,
            features_discriminator=feature_critic,
            image_channel=image_channel,
            label_channel=label_channel,
        )
        
        
        self.critic_losses = []
        self.generator_losses = []
        self.curr_step = 0
        
        self.fixed_latent_space = torch.randn(25, z_dim, 1, 1)
        self.fixed_label = torch.tensor([i % num_classes for i in range(25)])
        
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        # READ: https://lightning.ai/docs/pytorch/stable/common/optimization.html#use-multiple-optimizers-like-gans
        # READ: https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        # READ: https://lightning.ai/docs/pytorch/stable/model/build_model_advanced.html
        # READ: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.backward
        # READ: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#manual-backward
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.hparams.learning_rate, betas=(self.hparams.beta_1, self.hparams.beta_2))
        optimizer_C = optim.Adam(self.critic.parameters(), lr=self.hparams.learning_rate, betas=(self.hparams.beta_1, self.hparams.beta_2))

        return optimizer_G, optimizer_C
    
    def on_load_checkpoint(self, checkpoint):
        # List of keys that you expect to load from the checkpoint
        keys_to_load = ['critic_losses', 'generator_losses', 'curr_step', 'fixed_latent_space', 'fixed_label']

        # Iterate over the keys and load them if they exist in the checkpoint
        for key in keys_to_load:
            if key in checkpoint:
                setattr(self, key, checkpoint[key])
    
    def on_save_checkpoint(self, checkpoint):
        # Save necessary variable to checkpoint
        checkpoint['critic_losses'] = self.critic_losses
        checkpoint['generator_losses'] = self.generator_losses
        checkpoint['curr_step'] = self.curr_step
        checkpoint['fixed_latent_space'] = self.fixed_latent_space
        checkpoint['fixed_label'] = self.fixed_label
    
    def on_train_start(self):
        if self.current_epoch == 0:
            self.generator.apply(initialize_weights)
            self.critic.apply(initialize_weights)
    
    def training_step(self, batch, batch_idx):
        # Get the Optimizers
        opt_generator, opt_critic = self.optimizers()
        
        # Get Data and Label
        X, labels = batch
        
        # Get the current batch size
        batch_size = X.shape[0]
        
        ##############################
        # Train Critic ###############
        ##############################
        mean_critic_loss_for_this_iteration = 0
        
        for _ in range(self.critic_repeats):
            # Clean the Gradient
            opt_critic.zero_grad()
            
            # Generate the noise.
            noise = torch.randn(batch_size, self.hparams.z_dim, device=self.device)
            
            # Generate fake image.
            fake = self.generator(noise, labels)
            
            # Get the Critic's prediction on the reals and fakes
            critic_fake_pred = self.critic(fake.detach(), labels)
            critic_real_pred = self.critic(X, labels)
            
            # Calculate the Critic loss using WGAN
            
            # Generate epsilon for interpolate image.
            epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device, requires_grad=True)
            
            # Calculate Gradient Penalty Critic model
            gp = gradient_penalty(self.critic, labels, X, fake.detach(), epsilon)
            
            # calculate full of WGAN-GP loss for Critic
            critic_loss = calculate_critic_loss(
                critic_fake_pred, critic_real_pred, gp, self.c_lambda
            )
            
            # Keep track of the average critic loss in this batch
            mean_critic_loss_for_this_iteration += critic_loss.item() / self.critic_repeats
            
            # Update the gradients Criticz
            # self.manual_backward(critic_loss, retain_graph=True)
            self.manual_backward(critic_loss) # no need retain graph cause, already detach() on the image, so it will cut from backpropagate. use that retain_graph=True if not using detach()
            
            # Update the optimizer
            opt_critic.step()            
        
        ##############################
        # Train Generator ############
        ##############################
        
        # Clean the gradient
        opt_generator.zero_grad()
        
        # Generate the noise.
        noise = torch.randn(batch_size, self.hparams.z_dim, device=self.device)
        
        # Generate fake image.
        fake = self.generator(noise, labels)
        
        # Get the Critic's prediction on the fakes by generator
        generator_fake_predictions = self.critic(fake, labels)
        
        # Calculate loss for Generator
        generator_loss = calculate_generator_loss(generator_fake_predictions)
        
        # update the gradient generator
        self.manual_backward(generator_loss)
        
        # Update the optimizer
        opt_generator.step()
        
        ##############################
        # Visualization ##############
        ##############################
        
        if self.curr_step % self.hparams.display_step == 0 and self.curr_step > 0:
            VISUALIZE = True
            if VISUALIZE:
                with torch.no_grad():
                    fake_images_fixed = self.generator(
                        self.fixed_latent_space.to(self.device),
                        self.fixed_label.to(self.device)
                    )
                    
            path_save = f"/kaggle/working/generates/generated-{self.curr_step}-step.png"
            plot_images_from_tensor(fake_images_fixed, size=(3, self.image_size, self.image_size), show=False, save_path=path_save)
            plot_images_from_tensor(X, size=(3, self.image_size, self.image_size), show=False)
            
            print(f" ==== Critic Loss: {mean_critic_loss_for_this_iteration} ==== ")
            print(f" ==== Generator Loss: {generator_loss.item()} ==== ")
         
        self.curr_step += 1
        
        ##############################
        # Logging ####################
        ##############################
        # Store the loss Critic into Log
        self.log("critic_loss", mean_critic_loss_for_this_iteration, on_step=False, on_epoch=True, prog_bar=True)
        self.log("generator_loss", generator_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        
        # store into list, so can used later for visualization
        self.critic_losses.append(mean_critic_loss_for_this_iteration)
        self.generator_losses.append(generator_loss.item())
    
    def forward(self, noise, labels):
        return self.generator(noise, labels)
    
    def predict_step(self, noise, labels):
        return self.generator(noise, labels)