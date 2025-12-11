# vae/train_vae.py

from model import VAE
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from data.data_loader import load_data, corrupt_data 

def main():
    z_data_79, z_data_80, z_data_81, z_data_83, z_data_85 = load_data()

    train_data = torch.cat([
        torch.tensor(z_data_79, dtype=torch.float32),
        torch.tensor(z_data_80, dtype=torch.float32),
        torch.tensor(z_data_81, dtype=torch.float32),
        torch.tensor(z_data_83, dtype=torch.float32)
    ], dim=0)
    k = int(train_data.shape[0] * 0.9)
    test_data = torch.tensor(z_data_85, dtype=torch.float32)

    data_mean = train_data.mean(dim=(0, 2, 3), keepdim=True)
    data_std = train_data.std(dim=(0, 2, 3), keepdim=True)
    train_data_norm = (train_data - data_mean) / (data_std + 1e-6)
    test_data_norm = (test_data - data_mean) / (data_std + 1e-6)


    train_z = corrupt_data(train_data_norm, k)
    input_train_z = torch.tensor(train_z, dtype=torch.float32)
    label_train_z = torch.tensor(train_data_norm, dtype=torch.float32)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    batch_size = 40
    epochs = 30
    net = VAE(input_dim=2, hidden_dim=64, latent_dim=128)
    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=0.0001, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    input_train_z = input_train_z.to(device)
    label_train_z = label_train_z.to(device)
    trainN = input_train_z.shape[0]
    kl_weight_start = 0.0001
    kl_weight_end = 0.001
    kl_annealing_epochs = 15

    for epoch in range(epochs):
        net.train()
        indices = np.random.permutation(trainN)
        epoch_loss = 0

        if epoch < kl_annealing_epochs:
            kl_weight = kl_weight_start + (kl_weight_end - kl_weight_start) * (epoch / kl_annealing_epochs)
        else:
            kl_weight = kl_weight_end

        for step in range(0, trainN, batch_size):
            batch_indices = indices[step:step + batch_size]
            input_batch = input_train_z[batch_indices]
            label_batch = label_train_z[batch_indices]

            optimizer.zero_grad()
            output, mu, logvar = net(input_batch)
            logvar = torch.clamp(logvar, min = -4, max = 4)
            recon_loss = loss_fn(output, label_batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / input_batch.size(0)

            #kl_loss = torch.clamp(kl_loss, min=0.1)

            loss_value = recon_loss + kl_weight*kl_loss

            loss_value.backward()
            optimizer.step()

            epoch_loss += loss_value.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss/trainN:.4f}, KL Loss: {kl_loss.item():.4f}")

    torch.save(net.state_dict(), "net_weights.pth")
    torch.save(net, "net_full.pth")