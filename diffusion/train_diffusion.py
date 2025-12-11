# diffusion/train_diffusion.py

import torch
from scheduler import LinearNoiseScheduler
from unet_diffusion import Diffusion
import torch.nn as nn
from data.data_loader import load_data, normalize_data
def main():
    _, z_data_80, z_data_81, z_data_83, z_data_85 = load_data()

    model = Diffusion(out_channels=2, time_embed_dim=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    scheduler = LinearNoiseScheduler(beta_start=1e-4, beta_end=0.02, timesteps=100, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()
    epochs = 30
    batch_size = 25
    classifier_free_guidance_prob = 0.1
    spatial_loss_weight = 0.1
    model.to(device)
    train_data = torch.cat([z_data_80, z_data_81, z_data_83], dim=0)
    train_data_norm, mean, std = normalize_data(train_data)
    num_timesteps = train_data.shape[0] - 1

    def spatial_continuity_loss(pred):
        dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        return (dx.pow(2).mean() + dy.pow(2).mean()) * spatial_loss_weight


    training_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        batch_indices = torch.randperm(num_timesteps)

        for i in range(0, num_timesteps, batch_size):
            optimizer.zero_grad()
            indices = batch_indices[i:min(i+batch_size, num_timesteps)]
            batch_size_actual = len(indices)

            x_t = train_data_norm[indices].to(device)
            x_t_plus_1 = train_data_norm[indices + 1].to(device)

            if torch.rand(1).item() < classifier_free_guidance_prob:
                zero_mask = torch.zeros_like(x_t)
                use_zeros = torch.rand(batch_size_actual) < 0.5

                for idx, use_zero in enumerate(use_zeros):
                    if use_zero:
                        x_t[idx] = zero_mask[idx]

            t = torch.randint(0, scheduler.timesteps, (batch_size_actual,), device=device).long()
            noise = torch.randn_like(x_t_plus_1).to(device)
            noisy_x_t_plus_1 = scheduler.add_noise(x_t_plus_1, noise, t)
            noise_pred = model(noisy_x_t_plus_1, t, x_t)
            recon_loss = loss_fn(noise_pred, noise)
            spatial_loss = spatial_continuity_loss(noise_pred)
            loss = recon_loss + spatial_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * batch_size_actual

        avg_loss = total_loss / num_timesteps
        training_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'weather_diffusion_model_epoch{epoch+1}.pth')

    torch.save(model.state_dict(), 'weather_diffusion_model.pth')
