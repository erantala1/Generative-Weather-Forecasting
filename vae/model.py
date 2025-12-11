# vae/model.py
# implementation of VAE model

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device=None):
        super(VAE, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input_height = 91
        self.input_width = 180
        self.encoder = nn.Sequential(
            # 91×180 -> 46×90
            nn.Conv2d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2),

            # 46×90 -> 23×45
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(0.2),

            # 23×45 -> 12×23
            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.LeakyReLU(0.2),

            # 12×23 -> 6×12
            nn.Conv2d(hidden_dim*4, hidden_dim*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim*8),
            nn.LeakyReLU(0.2),
        ).to(self.device)

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, self.input_height, self.input_width).to(self.device)
            encoded_output = self.encoder(dummy_input)
            self.encoded_h = encoded_output.size(2)
            self.encoded_w = encoded_output.size(3)
            self.encoded_channels = encoded_output.size(1)
            self.flattened_dim = encoded_output.view(1, -1).size(1)
            print(f"Encoded shape: {encoded_output.shape}, Flattened: {self.flattened_dim}")

        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim).to(self.device)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim).to(self.device)
        self.fc_decode = nn.Linear(latent_dim, self.flattened_dim).to(self.device)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.encoded_channels, hidden_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.LeakyReLU(0.2),

            # 6×12 -> 12×24 (upscale)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.LeakyReLU(0.2),

            # 12×24 -> 23×45 (approximately, will be exact with final resize)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim*4, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(0.2),

            # 23×45 -> 46×90 (approximately)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2),

            # 46×90 -> 91×180 (approximately)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2),

            # Final adjustment to exact output size and channels
            nn.Conv2d(hidden_dim, input_dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).to(self.device)

        self.to(self.device)

    def encode(self, x):
        x = x.to(self.device)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-4, max=4)
        std = torch.exp(logvar / 2) + 1e-5
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std

    def decode(self, z):
        z = z.to(self.device)
        x = self.fc_decode(z)
        batch_size = z.size(0)
        x = x.view(batch_size, self.encoded_channels, self.encoded_h, self.encoded_w)
        x = self.decoder(x)

        if x.size(2) != self.input_height or x.size(3) != self.input_width:
            x = F.interpolate(x, size=(self.input_height, self.input_width),
                             mode='bilinear', align_corners=False)

        return x

    def forward(self, x):
        x = x.to(self.device)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar