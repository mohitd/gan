import numpy as np
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision.utils import save_image


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        flattened_img = x.view(x.size(0), -1)
        return self.model(flattened_img)


class GAN(object):
    def __init__(self, dataset, **kwargs):
        self.data_loader = dataset
        self.img_shape = kwargs['img_shape']
        self.latent_dim = kwargs['latent_dim']
        self.save_dir = kwargs['save_dir']
        self.model_name = kwargs['model_name']
        self.num_epochs = kwargs['num_epochs']
        self.save_interval = kwargs['save_interval']

        self.model_dir = os.path.join(self.save_dir, self.model_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.G = Generator(latent_dim=self.latent_dim, img_shape=self.img_shape)
        self.D = Discriminator(self.img_shape)
        self.loss = nn.BCELoss()

        print(self.G)
        print(self.D)

        self.G_optim = optim.Adam(self.G.parameters())
        self.D_optim = optim.Adam(self.D.parameters())

        if torch.cuda.is_available():
            self.D.cuda()
            self.G.cuda()
            self.loss.cuda()

    def train(self):
        # notify model we're training, e.g., dropout or batch norm
        self.G.train()
        self.D.train()

        for epoch in range(self.num_epochs):
            for i, (x, _) in enumerate(self.data_loader):
                # input image and random vector
                x = Variable(x)
                batch_size = x.size(0)

                if torch.cuda.is_available():
                    y_real = Variable(torch.ones(batch_size, 1).cuda(), requires_grad=False)
                    y_fake = Variable(torch.zeros(batch_size, 1).cuda(), requires_grad=False)
                else:
                    y_real = Variable(torch.ones(batch_size, 1), requires_grad=False)
                    y_fake = Variable(torch.zeros(batch_size, 1), requires_grad=False)

                z = Variable(torch.rand((batch_size, self.latent_dim)))

                """
                Train the discriminator
                """
                self.D_optim.zero_grad()

                # run discriminator against real images
                D_real = self.D(x)
                # one-sided label smoothing
                D_real_loss = self.loss(D_real, y_real * 0.9)

                # run the discriminator against the fake images
                G = self.G(z)
                D_fake = self.D(G)
                D_fake_loss = self.loss(D_fake, y_fake)

                # combine both losses and weight update
                D_loss = D_real_loss + D_fake_loss
                D_loss.backward()
                self.D_optim.step()

                """
                Train the generator
                """
                self.G_optim.zero_grad()

                # generate fake examples and treat them as "real"
                G = self.G(z)
                D_fake = self.D(G)
                G_loss = self.loss(D_fake, y_real)

                # weight update
                G_loss.backward()
                self.G_optim.step()

                if ((i + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (i + 1), len(self.data_loader.dataset) // batch_size,
                           D_loss.item(), G_loss.item()))

            if (epoch + 1) % self.save_interval == 0:
                save_image(G[:32], os.path.join(self.model_dir, 'images.png'), normalize=True)
                self.save()

    def save(self):
        torch.save(self.G.state_dict(), os.path.join(self.model_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(self.model_dir, self.model_name + '_D.pkl'))

    def load(self):
        self.G.load_state_dict(torch.load(os.path.join(self.model_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(self.model_dir, self.model_name + '_D.pkl')))
