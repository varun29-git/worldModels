import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------

class Encoder(nn.Module):

    def __init__(self, input_shape):
        super().__init__()

        c, h, w = input_shape

        # conv layers
        self.conv1 = nn.Conv2d(c, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)

    def forward(self, x):
        
        # activation = RELU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # flatten
        x = torch.flatten(x, start_dim=1)

        return x


# --------------------------------------------------

class Decoder(nn.Module):

    def __init__(self, latent_dim, output_shape):
        super().__init__()

        c, h, w = output_shape

        # fc layer
        self.fc = nn.Linear(latent_dim, 1024)

        # deconv layers
        self.deconv1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, c, kernel_size=6, stride=2)

    def forward(self, z):

        x = self.fc(z)

        # reshape
        x = x.view(-1, 1024, 1, 1)

        # activation = RELU
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))

        # activation = SIGMOID
        x = torch.sigmoid(self.deconv4(x))

        return x


# --------------------------------------------------

class VAE(nn.Module):

    def __init__(self, input_shape=(3, 64, 64), latent_dim=32):
        super().__init__()

        # encoder
        self.encoder = Encoder(input_shape)

        # fc layers
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # decoder
        self.decoder = Decoder(latent_dim, input_shape)

    def encode(self, x):

        # activation = RELU
        h = self.encoder(x)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        z = mu + eps * std

        return z

    def decode(self, z):

        return self.decoder(z)

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        recon = self.decode(z)

        return recon, mu, logvar


# --------------------------------------------------
