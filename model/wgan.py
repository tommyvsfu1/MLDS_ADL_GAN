# reference code https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py
import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


def initialize_weights(m):
    if (isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d)) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)

class Generator(nn.Module):
    def __init__(self,opt=None):
        super(Generator, self).__init__()
        ngf = 64
        d = 32
        self.model = nn.Sequential(
            # # input is Z, going into a convolution
            # nn.ConvTranspose2d(opt.latent_dim, ngf * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.LeakyReLU(0.2,True),
            # # state size. (ngf*8) x 4 x 4
            # nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 4),
            # nn.LeakyReLU(0.2,True),
            # # state size. (ngf*4) x 8 x 8
            # nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            # nn.LeakyReLU(0.2,True),
            # # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.LeakyReLU(0.2, True),
            # # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            # nn.Tanh()
            # # state size. (nc) x 64 x 64
            nn.ConvTranspose2d(100, d * 8, 4, 1, 0),
            nn.BatchNorm2d(d * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1),
            nn.BatchNorm2d(d * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1),
            nn.BatchNorm2d(d * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(d * 2, 1, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, z):
        img = self.model(z)
        return img


    def predict(self, z):
        img = self.model(z)
        return img

class Discriminator(nn.Module):
    def __init__(self, opt=None):
        super(Discriminator, self).__init__()
        # self.img_shape = (opt.channels, opt.img_size, opt.img_size)
        d = 32
        ndf = 64
        self.model = nn.Sequential(
            # # input is (nc) x 64 x 64
            # nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf) x 32 x 32
            # nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*2) x 16 x 16
            # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*4) x 8 x 8
            # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 4 x 4
            # # Flatten(),
            # # nn.Linear(512*4*4,1),
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Conv2d(1, d, 4, 2, 1),
            nn.InstanceNorm2d(d),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d, d * 2, 4, 2, 1),
            nn.InstanceNorm2d(d * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d * 2, d * 4, 4, 2, 1),
            nn.InstanceNorm2d(d * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d * 4, 1, 4, 1, 0),
        )        
    def forward(self, img):
        validity = self.model(img)
        return validity


# test_g_gan = Generator()
# test_g_gan.apply(initialize_weights)#
# test_d_gan = Discriminator()
# fake_seed = torch.randn(16, 100,1,1)
# fake_images = test_g_gan.forward(fake_seed)
# print(fake_images.size())
# print("fake image", fake_images[0])
# label = test_d_gan(fake_images)
# print(label.size())
# print("label",label.size())