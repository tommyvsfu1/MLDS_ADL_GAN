

## exp1
structure

G:
```python
def block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers
self.model = nn.Sequential(
    *block(opt.latent_dim, 128, normalize=False),
    *block(128, 256),
    *block(256, 512),
    *block(512, 1024),
    nn.Linear(1024, int(np.prod(self.img_shape))),
    nn.Tanh()
)
```
D:
```python
self.model = nn.Sequential(
    nn.Linear(int(np.prod(self.img_shape)), 512),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(512, 256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(256, 1),
)
```




structure 2
使用pytorch tutorial DCGAN 架構

60000 updates
![](https://i.imgur.com/07U4qQd.png)

發生很嚴重的noise 問題，不知道哪裡出錯了


然後我回去看李老師的lecture，我看他們training 50000 update就差不多了，所以大概50 epochs (1 hours多一點就可以開始debug，不用到6 hours)

exp3  
Leaky ReLU  
exp4  
Adam + Conv2d initialization  
exp5  
D updates times : 5 -> 1  
exp6  
Dense 架構  
exp7  
前面的實驗算是白做了，因為Image我做了normalization to [-1,1]  
但是，在save_image 中我denormalize是img*255 而非(img+1)*127.5  



## Reference for Debug
https://github.com/pavitrakumar78/Anime-Face-GAN-Keras/blob/master/GAN_Trainingv3.py
https://github.com/nashory/gans-awesome-applications (find some useful repo, read their code for debugging)