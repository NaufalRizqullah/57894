import torch

# Gradient Penalty Calculation - Calculate Gradient of Critic Score
def gradient_penalty(critic, labels, real_images, fake_images, epsilon):
    """
    This function calculates the gradient penalty for the WGAN-GP loss function

    Parameters:
    critic (nn.Module): The critic model
    labels (torch.tensor): The labels for the images
    real_images (torch.tensor): The real images
    fake_images (torch.tensor): The fake images
    epsilon (torch.tensor): The interpolation parameter

    Returns:
    gradient_penalty (torch.tensor): The gradient penalty for the critic model
    """

    # Create the interpolated images as a weighted combination of real and fake images
    interpolated_images = real_images * epsilon + fake_images * (1 - epsilon)

    mixed_scores = critic(interpolated_images, labels)
    
    create_real_label = torch.ones_like(mixed_scores)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=create_real_label,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Reshape each image in the batch into a 1D tensor (flatten the images)
    gradient = gradient.view(len(gradient), -1)

    # Calculate the L2 norm of the gradients
    gradient_norm = gradient.norm(2, dim=1)

    # Calculate the penalty as the mean squared distance of the norms from 1
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    
    return gradient_penalty

# Critic Loss Calculation
def calculate_critic_loss(critic_fake_prediction, critic_real_prediction, gradient_penalty, critic_lambda):
    """
    Calculates the critic loss, which is the difference between the mean of the real scores and mean of the fake scores plus the gradient penalty.

    Parameters:
    critic_fake_prediction (torch.tensor): The critic predictions for the fake images
    critic_real_prediction (torch.tensor): The critic predictions for the real images
    gradient_penalty (torch.tensor): The gradient penalty for the critic model
    critic_lambda (float): The coefficient for the gradient penalty

    Returns:
    critic_loss (torch.tensor): The critic loss
    """
    critic_loss = (
        -(torch.mean(critic_real_prediction) - torch.mean(critic_fake_prediction))  + critic_lambda * gradient_penalty
    )
    
    return critic_loss

# Generator Loss Calculation
def calculate_generator_loss(critic_fake_prediction):
    """
    Calculates the generator loss, which is the mean of the critic predictions for the fake images with a negative sign.

    Parameters:
    critic_fake_prediction (torch.tensor): The critic predictions for the fake images

    Returns:
    generator_loss (torch.tensor): The generator loss
    """
    generator_loss = -1.0 * torch.mean(critic_fake_prediction)
    return generator_loss