# Conditional GAN Shoe, Sandal, Boot
=====================================

## Project Overview
---------------

This project implements a Conditional Generative Adversarial Network (CGAN) using the Conditional WGAN-GP architecture to generate images of shoes, sandals, and boots.

### Key Features
------------

* **Conditional WGAN-GP architecture**: for generating images of shoes, sandals, and boots
* **Trained on a limited dataset**: due to VRAM constraints
* **Model architecture and hyperparameters optimized**: for Kaggle environment with 15GB VRAM
* **Embedding used instead of one-hot encoding**: for training labels to avoid using 0 on labels
* **Implemented using PyTorch Lightning framework**

## Training Details
---------------

### Training Epochs
----------------

* Approximately 300 epochs

### Model Architecture Compromises
-----------------------------

* **Reduced Size of latent space (z dim) and Embedding**: due to VRAM limitations
* **Limited features for generator and critic networks**: due to VRAM limitations
* **Image size limitations**: due to VRAM limitations
* **Dataset used**: [Shoe vs Sandal vs Boot Image Dataset (15K Images)](https://www.kaggle.com/datasets/hasibalmuzdadid/shoe-vs-sandal-vs-boot-dataset-15k-images)

## Results
---------

Despite the architectural compromises, the model produces reasonable results. However, the quality of the generated images may not be optimal due to the limited dataset and VRAM constraints.

Here is an animated demonstration of the training progress.
![CWGAN-GP Animation](data/CWGAN-GP_animated.gif)

## Demo
-------------
The demo of this project is deployed on Hugging Face's model hub and uses the Gradio framework to provide a user-friendly interface for interacting with the model. You can try out the demo by visiting [this link](https://huggingface.co/spaces/SkylarWhite/57894).


## Future Work
-------------

* **Experiment with larger datasets and more complex model architectures**
* **Investigate alternative optimization techniques to improve model performance**
* **Explore other applications of Conditional GANs in computer vision**
