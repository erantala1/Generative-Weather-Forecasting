# diffusion/unet_diffusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Diffusion(nn.Module):
    def __init__(self, out_channels=2, time_embed_dim=128):
        super(Diffusion, self).__init__()

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU()
        )

        self.cond_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU()
        )

        self.init_conv = nn.Conv2d(2 + 32, 32, kernel_size=5, padding=2)
        self.attn1 = SimpleResBlock(64)
        self.attn2 = SimpleResBlock(128)
        self.attn3 = SimpleResBlock(256)

        self.down1 = ResConvBlock(32, 64, time_embed_dim)
        self.down2 = ResConvBlock(64, 128, time_embed_dim)
        self.down3 = ResConvBlock(128, 256, time_embed_dim)

        self.mid1 = ResConvBlock(256, 256, time_embed_dim)
        self.mid_attn = SimpleResBlock(256)
        self.mid2 = ResConvBlock(256, 256, time_embed_dim)

        self.up3 = ResConvBlock(256 + 256, 128, time_embed_dim)
        self.up2 = ResConvBlock(128 + 128, 64, time_embed_dim)
        self.up1 = ResConvBlock(64 + 64, 32, time_embed_dim)

        self.final_block = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU()
        )

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x, t, cond):
        cond_features = self.cond_encoder(cond)
        x_cond = torch.cat([x, cond_features], dim=1)
        t_emb = self.time_embed(t)
        h = self.init_conv(x_cond)

        h1 = self.down1(h, t_emb)
        h1 = self.attn1(h1)
        h1_shape = h1.shape[-2:]

        h2 = self.down2(self.down_sample(h1), t_emb)
        h2 = self.attn2(h2)
        h2_shape = h2.shape[-2:]

        h3 = self.down3(self.down_sample(h2), t_emb)
        h3 = self.attn3(h3)
        h3_shape = h3.shape[-2:]

        h = self.mid1(self.down_sample(h3), t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        h = F.interpolate(h, size=h3_shape, mode='bilinear', align_corners=False)
        h = torch.cat([h, h3], dim=1)
        h = self.up3(h, t_emb)

        h = F.interpolate(h, size=h2_shape, mode='bilinear', align_corners=False)
        h = torch.cat([h, h2], dim=1)
        h = self.up2(h, t_emb)

        h = F.interpolate(h, size=h1_shape, mode='bilinear', align_corners=False)
        h = torch.cat([h, h1], dim=1)
        h = self.up1(h, t_emb)

        h = self.final_block(h)
        return self.final_conv(h)

class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=None):
        super(ResConvBlock, self).__init__()
        self.time_mlp = nn.Linear(time_dim, out_channels) if time_dim else None

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()

        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t=None):
        residual = self.residual(x)

        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        if t is not None and self.time_mlp is not None:
            time_emb = self.time_mlp(t)
            h = h + time_emb.unsqueeze(-1).unsqueeze(-1)

        h = self.conv2(h)
        h = self.norm2(h)
        h = h + residual
        h = self.act2(h)

        return h


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super(SimpleResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act2 = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x + residual