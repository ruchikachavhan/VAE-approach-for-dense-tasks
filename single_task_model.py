import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

from create_dataset import *
from utils import *

class param_net(nn.Module):
    def __init__(self, latent_dim):
        super(param_net, self).__init__()
        self.conv_mean = nn.Sequential(nn.Conv2d(latent_dim, latent_dim, 3, 1, 1), 
                                    nn.BatchNorm2d(latent_dim), nn.ReLU(), 
                                    )
        self.conv_var = nn.Sequential(nn.Conv2d(latent_dim, latent_dim, 3, 1, 1), 
                                    nn.BatchNorm2d(latent_dim), nn.LeakyReLU(0.2), 
                                )

    def forward(self, x):
        return self.conv_mean(x), self.conv_var(x)


class Net(nn.Module):
    def __init__(self, n_channels, n_classes, mode):
        super(Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mode = mode
        filters = [64, 128, 256, 512, 1024]

        self.conv1 = nn.Sequential(nn.Conv2d(n_channels, filters[0], 4, 2, 1), 
                                    nn.BatchNorm2d(filters[0]), nn.LeakyReLU(0.2), 
                                    nn.Conv2d(filters[0], filters[0], 3, 1, 1), 
                                    nn.BatchNorm2d(filters[0]), nn.LeakyReLU(0.2), 
                                    nn.Conv2d(filters[0], filters[0], 3, 1, 1), 
                                    )

        self.conv2 = nn.Sequential(nn.Conv2d(filters[0], filters[1], 4, 2, 1), 
                                    nn.BatchNorm2d(filters[1]), nn.LeakyReLU(0.2), 
                                    nn.Conv2d(filters[1], filters[1], 3, 1, 1), 
                                    nn.BatchNorm2d(filters[1]), nn.LeakyReLU(0.2), 
                                    nn.Conv2d(filters[1], filters[1], 3, 1, 1), 
                                    )

        self.conv3 = nn.Sequential(nn.Conv2d(filters[1], filters[2], 4, 2, 1), 
                                    nn.BatchNorm2d(filters[2]), nn.LeakyReLU(0.2), 
                                    nn.Conv2d(filters[2], filters[2], 3, 1, 1), 
                                    nn.BatchNorm2d(filters[2]), nn.LeakyReLU(0.2), 
                                    nn.Conv2d(filters[2], filters[2], 3, 1, 1), 
                                    )

        self.conv4 = nn.Sequential(nn.Conv2d(filters[2], filters[3], 4, 2, 1), 
                                    nn.BatchNorm2d(filters[3]), nn.LeakyReLU(0.2), 
                                    nn.Conv2d(filters[3], filters[3], 3, 1, 1), 
                                    nn.BatchNorm2d(filters[3]), nn.LeakyReLU(0.2), 
                                    nn.Conv2d(filters[3], filters[3], 3, 1, 1), 
                                    )

        self.bottleneck = nn.Sequential(nn.Conv2d(filters[3], filters[4], 3, 1, 1), 
                                        nn.BatchNorm2d(filters[4]), nn.LeakyReLU(0.2), 
                                        nn.Conv2d(filters[4], filters[3], 3, 1, 1), 
                                        # nn.BatchNorm2d(filters[3]), nn.LeakyReLU(0.2), 
                                        )

        if(mode == "vae"):
            self.param_net = param_net(filters[3])


        self.upconv1 = nn.Sequential(nn.Conv2d(filters[4], filters[3], 3, 1, 1), 
                                        nn.BatchNorm2d(filters[3]), nn.LeakyReLU(0.2), 
                                        nn.Conv2d(filters[3], filters[3], 3, 1, 1), 
                                        nn.BatchNorm2d(filters[3]), nn.LeakyReLU(0.2), 
                                        nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, stride=2, padding=1, output_padding=1)) 


        self.upconv2 = nn.Sequential(nn.Conv2d(filters[2] + filters[3], filters[2], 3, 1, 1), 
                                        nn.BatchNorm2d(filters[2]), nn.LeakyReLU(0.2), 
                                        nn.Conv2d(filters[2], filters[2], 3, 1, 1), 
                                        nn.BatchNorm2d(filters[2]), nn.LeakyReLU(0.2), 
                                        nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, stride=2, padding=1, output_padding=1)) 

        self.upconv3 = nn.Sequential(nn.Conv2d(filters[1] + filters[2], filters[1], 3, 1, 1), 
                                        nn.BatchNorm2d(filters[1]), nn.LeakyReLU(0.2), 
                                        nn.Conv2d(filters[1], filters[1], 3, 1, 1), 
                                        nn.BatchNorm2d(filters[1]), nn.LeakyReLU(0.2), 
                                        nn.ConvTranspose2d(filters[1], filters[1], kernel_size=3, stride=2, padding=1, output_padding=1)) 

        self.upconv4 = nn.Sequential(nn.Conv2d(filters[1] + filters[0], filters[0], 3, 1, 1), 
                                        nn.BatchNorm2d(filters[0]), nn.LeakyReLU(0.2), 
                                        nn.Conv2d(filters[0], filters[0], 3, 1, 1), 
                                        nn.BatchNorm2d(filters[0]), nn.LeakyReLU(0.2), 
                                        nn.ConvTranspose2d(filters[0], self.n_classes, kernel_size=3, stride=2, padding=1, output_padding=1)) 

        self.out = nn.Sequential(nn.Conv2d(filters[3], filters[3], 3, 1, 1), nn.ReLU(), nn.Sigmoid())
# 

    def forward(self, x):
        x1 = self.conv1(x)
        x2  = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.bottleneck(x4)

        if(self.mode=="vae"):
            mu, log_var = self.param_net(x5)
            x5 = self.sample(x5, mu, log_var)
        
        x6 = torch.cat((x5, x4), dim = 1)
        x7 = self.upconv1(x6)
        
        x8 = torch.cat((x7, x3), dim = 1)
        x9 = self.upconv2(x8)
        x10 = torch.cat((x9, x2), dim = 1)
        x11 = self.upconv3(x10)
        x12 = torch.cat((x11, x1), dim=1)
        x14 = self.upconv4(x12)

        if(self.mode == "vae"):

            return x14, mu, log_var, self.out(x5)
        else:
            return x14

    def sample(self, out, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(out)
        eps = eps.mul(std).add_(mu)

        return eps

        

        

# class Net(nn.Module):
#     def __init__(self, input_nc, latent_dim, output_nc, mode, sample_num):
#         super(Net, self).__init__()
#         self.encoder = Encoder(input_nc)
#         self.decoder = Decoder(latent_dim, output_nc)
#         self.net_mean = VAE_fcn(latent_dim)
#         self.net_var = VAE_fcn(latent_dim)
#         self.mode = mode
#         self.sample_num = sample_num
#         self.conv_vae = nn.Conv2d(latent_dim, latent_dim, 1, 1)
#         self.bn = nn.BatchNorm2d(latent_dim)
#         self.relu = nn.LeakyReLU(0.2)
#         self.att = nn.Sequential( nn.Sigmoid())

#     def forward(self, img):
        
#         x = self.encoder(img)
#         if(self.mode == "vae"):

#             x, mu, log_var = self.sample(x, mu, log_var)

#         out = self.decoder(x)
#         if(self.mode== "normal"):
#             return out
#         else:
#             return out, mu, log_var, self.att(x)


        

