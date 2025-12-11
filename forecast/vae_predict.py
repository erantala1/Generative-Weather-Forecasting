# forecast/vae_predict.py

import matplotlib as plt
import torch
from data.data_loader import corrupt_data, load_data
from vae.model import VAE

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
    test_data_norm = (test_data - data_mean) / (data_std + 1e-6)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    net_loaded = VAE(input_dim=2, hidden_dim=64, latent_dim=128)
    net_loaded.load_state_dict(torch.load("net_weights.pth", map_location=device))
    net_loaded = net_loaded.to(device)
    net_loaded.eval()

    input_85 = corrupt_data(test_data_norm, k)
    input_85 = input_85.float().to(device)

    with torch.no_grad():
        predicted_85, _, _ = net_loaded(input_85)
        predicted_85 = predicted_85.detach().cpu()

    predicted_85_denorm = predicted_85 * data_std + data_mean

    spatial_points = test_data.shape[2] * test_data.shape[3]
    sparse_percent = (k / spatial_points) * 100

    time_steps = [1, 3, 7]

    fig, axes = plt.subplots(4, len(time_steps), figsize=(15, 12))
    fig.suptitle(f'VAE Results with 90% sparsity', fontsize=16)

    for i, t in enumerate(time_steps):
        ax = axes[0, i]
        cs = ax.contourf(predicted_85_denorm[t, 0].numpy(), cmap='coolwarm')
        plt.colorbar(cs, ax=ax)
        ax.set_title(f'Predicted Ch1 - Step {t}')

        ax = axes[1, i]
        cs = ax.contourf(test_data[t, 0].numpy(), cmap='coolwarm')
        plt.colorbar(cs, ax=ax)
        ax.set_title(f'Ground Truth Ch1 - Step {t}')

        ax = axes[2, i]
        cs = ax.contourf(predicted_85_denorm[t, 1].numpy(), cmap='coolwarm')
        plt.colorbar(cs, ax=ax)
        ax.set_title(f'Predicted Ch2 - Step {t}')

        ax = axes[3, i]
        cs = ax.contourf(test_data[t, 1].numpy(), cmap='coolwarm')
        plt.colorbar(cs, ax=ax)
        ax.set_title(f'Ground Truth Ch2 - Step {t}')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    main()