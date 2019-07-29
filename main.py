# reference code https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py
# https://github.com/martinarjovsky/WassersteinGAN/blob/master/main.py
# GAN tips : https://github.com/soumith/ganhacks
import argparse
import os
import numpy as np
import math
import sys
from torch.autograd import Variable
import torch
from argument import add_argument
from utils import load_Anime, load_mnist
from model.wgangp import compute_gradient_penalty
from utils import save_model, save_imgs, save_mnist_imgs # MLDS default save_imgs function
from logger import TensorboardLogger
parser = add_argument(argparse.ArgumentParser()) # argument.py
opt = parser.parse_args()
print(opt)

# Device
cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda') if cuda else torch.device('cpu')
print("using device:", device)
# Initialize generator and discriminator
if opt.model_use == "WGAN":
    from model.wgan import Generator, Discriminator, initialize_weights
    print("using WGAN")
elif opt.model_use == "WGANGP":
    pass
generator = Generator(opt).to(device)
discriminator = Discriminator(opt).to(device)

generator.apply(initialize_weights)
discriminator.apply(initialize_weights)

# Optimizers
# Tip 9 : use Adam
if opt.model_use == "WGAN":
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)
    # exp 4 , use Adam
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001,betas=(0,0.9))
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001,betas=(0,0.9))    
elif opt.model_use == "WGANGP": # WGANGP paper default parameters
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001,betas=(0,0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001,betas=(0,0.9))

# Load data
dataloader = load_Anime()
# dataloader = load_mnist()

# log
if opt.model_use == "WGAN":
    log_dir = './wgan'
elif opt.model_use == "WGANGP":
    log_dir = './wgangp'
tensorboard = TensorboardLogger(log_dir)
# ----------
#  Training
# ----------

batches_done = 0
epoch_s = 0
one = (torch.FloatTensor([1])).cuda()
mone = (one * -1).cuda()
for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure input
        real_imgs = imgs.to(device)
        # ---------------------
        #  Train Discriminator
        # ---------------------
        for p in discriminator.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in discriminator update
        for p in generator.parameters():
            p.requires_grad = False

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = np.expand_dims(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)),axis=2)
        z = np.expand_dims(z, axis=3)
        z = torch.from_numpy(z).float().to(device)
        # Generate a batch of images
        fake_imgs = generator(z).detach()

        # Adversarial loss
        if opt.model_use == "WGAN":
            loss_D_real = torch.mean(discriminator(real_imgs))
            loss_D_real.backward(one)

            loss_D_fake = torch.mean(discriminator(fake_imgs))
            loss_D_fake.backward(mone)

            #loss = loss_D_real + loss_D_fake
            d_loss = loss_D_fake - loss_D_real
            wd = -d_loss
            optimizer_D.step()

            # Clip weights of discriminator(WGAN weight clipping)
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)


        elif opt.model_use == "WGANGP":
            penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs)
            loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs)) + \
                     torch.tensor([10])*penalty
        


        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:
            for p in discriminator.parameters():
                p.requires_grad = False # to avoid computation
            for p in generator.parameters():
                p.requires_grad = True
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)

            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader),(loss_D_real+loss_D_fake).item(), loss_G.item())
            )

        if batches_done % opt.sample_interval == 0:
            # save_mnist_imgs(batches_done, generator, device)
            print("saving... images and model")
            save_imgs(batches_done, generator,device)
            save_model(batches_done, generator, discriminator)
            

        tensorboard.scalar_summary("batch_D_loss",(d_loss).item(),batches_done)
        tensorboard.scalar_summary("batch_D_real_loss", loss_D_real.item(), batches_done)
        tensorboard.scalar_summary("batch_D_fake_loss", loss_D_fake.item(), batches_done)
        tensorboard.scalar_summary("batch_G_loss",loss_G.item(),batches_done)
        batches_done += 1
        
    tensorboard.scalar_summary("epoch_D_loss",d_loss.item(),epoch_s)
    tensorboard.scalar_summary("epoch_D_real_loss", loss_D_real.item(), epoch_s)
    tensorboard.scalar_summary("epoch_D_fake_loss", loss_D_fake.item(), epoch_s)
    tensorboard.scalar_summary("epoch_G_loss",loss_G.item(),epoch_s)
    epoch_s += 1
