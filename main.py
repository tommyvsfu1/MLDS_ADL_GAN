# reference code https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py

import argparse
import os
import numpy as np
import math
import sys

from torch.autograd import Variable
import torch
from argument import add_argument
from utils import load_Anime
from model.wgan import Generator, Discriminator
from utils import save_imgs # MLDS default save_imgs function

parser = add_argument(argparse.ArgumentParser()) # argument.py
opt = parser.parse_args()
print(opt)

# Device
cuda = True if torch.cuda.is_available() else False
# Initialize generator and discriminator
generator = Generator(opt)
discriminator = Discriminator(opt)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

# Datatype
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


dataloader = load_Anime()

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(1):

    for i, imgs in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

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
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

        if batches_done % opt.sample_interval == 0:
            save_imgs(generator)
        batches_done += 1

        