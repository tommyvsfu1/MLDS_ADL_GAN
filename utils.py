from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from scipy.misc import imread
from scipy.stats import norm
from PIL import Image
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


def save_imgs(idx, generator, device):
    r, c = 2, 2
    noise = np.expand_dims(np.random.normal(0, 1, (r * c, 100)),axis=2)
    noise = np.expand_dims(noise, axis=3)
    noise = torch.from_numpy(noise).float().to(device)

    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = generator.predict(noise).detach()
    gen_imgs = gen_imgs.cpu().numpy()
    
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            gen_imgs = gen_imgs.reshape((gen_imgs.shape[0],3,64,64))
            img = (gen_imgs[cnt,:,:,:]).transpose((1, 2, 0)) # C,H,W -> H,W,C
            img = ((img + 1)*127.5).astype(np.uint8)
            axs[i,j].imshow(img)
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("./model/wgan_checkpoint/output/" + "WGAN_" + "output_" + str(idx) + ".png")
    plt.close()


class AnimeDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file_name = str(self.frame.iloc[idx, 0]) + ".jpg"
        img_name = os.path.join(self.root_dir,file_name)
        image = imread(img_name) # read as np.array
        image = Image.fromarray(image) # convert to PIL image(Pytorch default image datatype)
        if self.transform:
            image = self.transform(image)

        return image

def load_Anime(dataset_filepath='image/', opt=None):

    if opt is None:
        img_size = 64
        batch_size = 32
    else :
        img_size = opt.img_size
        batch_size = opt.batch_size

    data_transform = transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) # tip 1 : normalize the images between -1 and 1
    ])

    dataset = AnimeDataset(csv_file=dataset_filepath + 'tags_clean.csv',
                                            root_dir=dataset_filepath + 'faces/',
                                            transform=data_transform
                                            )

    dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4)

    return dataloader


def save_model(idx, G, D, save_path='./model/wgan_checkpoint/model_dict/'):
    print('save model to', save_path)
    torch.save(G.state_dict(), save_path + "WGAN_G" + str(idx) + '.cpt')
    torch.save(D.state_dict(), save_path + "WGAN_D" + str(idx) + '.cpt')

