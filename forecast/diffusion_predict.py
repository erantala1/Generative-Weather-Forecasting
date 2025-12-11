# forecast/diffusion_predict.py

import torch
import torch.nn.functional as F
import matplotlib as plt
from data.data_loader import denormalize_data, load_data, normalize_data
from utils import apply_directional_consistency, apply_multi_scale_smoothing
from diffusion.unet_diffusion import Diffusion
from diffusion.scheduler import LinearNoiseScheduler

def make_prediction(model, scheduler, x_0, num_steps, device, mean, std):
    model.eval()
    x_0 = x_0.unsqueeze(0).to(device)
    predictions = [x_0.cpu()]
    current_x = x_0
    guidance_scales = [1.0, 2.5, 3.5, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
    denoising_strength = 0.4
    temporal_consistency = 0.8
    smoothing_sizes = [5, 7, 9, 11, 13, 13, 13, 13, 13, 13]
    smoothing_sigmas = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.0, 2.0, 2.0, 2.0]

    history = [current_x]

    for step in range(num_steps):
        print(f"Generating prediction for step {step+1}/{num_steps}")

        current_x = apply_multi_scale_smoothing(current_x, step, smoothing_sizes, smoothing_sigmas, device)
        steps_to_use = max(100 - step * 8, 50)
        step_indices = torch.linspace(0, scheduler.timesteps-1, steps_to_use).long()

        noise_scale = max(0.9 - step * 0.1, 0.3)
        x = torch.randn_like(current_x).to(device) * noise_scale

        if step > 0:
            prev_pred = predictions[-1].to(device)
            warm_start_weight = min(0.5 + step * 0.05, 0.8)
            x = x * (1 - warm_start_weight) + prev_pred * warm_start_weight

            if step > 1:
                prev_prev_pred = predictions[-2].to(device)
                x = apply_directional_consistency(x, prev_pred, prev_prev_pred)

        guidance = guidance_scales[min(step, len(guidance_scales)-1)]

        for i in reversed(step_indices):
            t = torch.full((1,), i, device=device, dtype=torch.long)

            with torch.no_grad():
                noise_pred_cond = model(x, t, current_x)
                empty_cond = torch.zeros_like(current_x)
                noise_pred_uncond = model(x, t, empty_cond)
                noise_pred = noise_pred_uncond + guidance * (noise_pred_cond - noise_pred_uncond)
                x, _ = scheduler.sample_prev_timestep(x, noise_pred, i)
                x = torch.clamp(x, -2.0, 2.0)
                smoothing_frequency = max(20 - step * 2, 5)
                if i % smoothing_frequency == 0:
                    x = apply_multi_scale_smoothing(x, step, smoothing_sizes, smoothing_sigmas, device)

        x = apply_multi_scale_smoothing(x, step, smoothing_sizes, smoothing_sigmas, device)
        predictions.append(x.cpu())
        history.append(x)

        if step > 0:
            blend_factor = temporal_consistency + min(step * 0.02, 0.1)
            current_x = x * (1 - blend_factor) + history[-2] * blend_factor
        else:
            current_x = x
        if len(history) > 3:
            history = history[-3:]

    predictions = torch.cat(predictions, dim=0).squeeze(1)
    predictions_denorm = denormalize_data(predictions, mean, std)

    return predictions_denorm

def plot_predictions(predictions, ground_truth, steps_to_show=[0, 3, 7]):
    n_steps = len(steps_to_show)
    fig, axes = plt.subplots(4, n_steps, figsize=(n_steps*5, 12))

    for i, step in enumerate(steps_to_show):

        pred_data_ch1 = predictions[step, 0].numpy()
        pred_data_ch2 = predictions[step, 1].numpy()
        truth_data_ch1 = ground_truth[step, 0].numpy()
        truth_data_ch2 = ground_truth[step, 1].numpy()

        ax = axes[0, i]
        im = ax.contourf(pred_data_ch1, cmap='coolwarm')
        ax.set_title(f'Predicted Ch1 - Step {step}')
        plt.colorbar(im, ax=ax)

        ax = axes[1, i]
        im = ax.contourf(truth_data_ch1, cmap='coolwarm')
        ax.set_title(f'Ground Truth Ch1 - Step {step}')
        plt.colorbar(im, ax=ax)

        ax = axes[2, i]
        im = ax.contourf(pred_data_ch2, cmap='coolwarm')
        ax.set_title(f'Predicted Ch2 - Step {step}')
        plt.colorbar(im, ax=ax)

        ax = axes[3, i]
        im = ax.contourf(truth_data_ch2, cmap='coolwarm')
        ax.set_title(f'Ground Truth Ch2 - Step {step}')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

def plot_predictions(predictions, ground_truth, steps_to_show=[0, 3, 7]):
    n_steps = len(steps_to_show)
    fig, axes = plt.subplots(4, n_steps, figsize=(n_steps*5, 12))

    for i, step in enumerate(steps_to_show):

        pred_data_ch1 = predictions[step, 0].numpy()
        pred_data_ch2 = predictions[step, 1].numpy()
        truth_data_ch1 = ground_truth[step, 0].numpy()
        truth_data_ch2 = ground_truth[step, 1].numpy()

        ax = axes[0, i]
        im = ax.contourf(pred_data_ch1, cmap='coolwarm')
        ax.set_title(f'Predicted Ch1 - Step {step}')
        plt.colorbar(im, ax=ax)

        ax = axes[1, i]
        im = ax.contourf(truth_data_ch1, cmap='coolwarm')
        ax.set_title(f'Ground Truth Ch1 - Step {step}')
        plt.colorbar(im, ax=ax)

        ax = axes[2, i]
        im = ax.contourf(pred_data_ch2, cmap='coolwarm')
        ax.set_title(f'Predicted Ch2 - Step {step}')
        plt.colorbar(im, ax=ax)

        ax = axes[3, i]
        im = ax.contourf(truth_data_ch2, cmap='coolwarm')
        ax.set_title(f'Ground Truth Ch2 - Step {step}')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

def main():
    z_data_79, z_data_80, z_data_81, z_data_83, z_data_85 = load_data()
    
    train_data = torch.cat([
        torch.tensor(z_data_79, dtype=torch.float32),
        torch.tensor(z_data_80, dtype=torch.float32),
        torch.tensor(z_data_81, dtype=torch.float32),
        torch.tensor(z_data_83, dtype=torch.float32)
    ], dim=0)

    train_data = torch.cat([z_data_80, z_data_81, z_data_83], dim=0)
    test_data = z_data_85
    train_data_norm, mean, std = normalize_data(train_data)
    test_data_norm = (test_data - mean) / (std + 1e-8)
    x_0 = test_data_norm[0]
    num_steps = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = Diffusion(out_channels=2, time_embed_dim=128)
    model.load_state_dict(torch.load('weather_diffusion_model.pth'))
    model.to(device)
    scheduler = LinearNoiseScheduler(beta_start=1e-4, beta_end=0.02, timesteps=100, device=device)
    predictions_denorm = make_prediction(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        num_steps=num_steps,
        device=device,
        mean=mean,
        std=std
    )
    ground_truth = test_data[:11]
    plot_predictions(predictions_denorm, ground_truth, steps_to_show=[0, 3, 7])

    mse_ch_1 = []
    mse_ch_2 = []

    for i in range(ground_truth.shape[0]):
        mse_1 = torch.mean((predictions_denorm[i, 0] - ground_truth[i, 0]) ** 2).item()
        mse_2 = torch.mean((predictions_denorm[i, 1] - ground_truth[i, 1]) ** 2).item()
        mse_ch_1.append(mse_1)
        mse_ch_2.append(mse_2)


    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mse_ch_1)), mse_ch_1, 'b-', label='Channel 1')
    plt.plot(range(len(mse_ch_2)), mse_ch_2, 'r-', label='Channel 2')
    plt.xlabel('Time step')
    plt.ylabel('MSE')
    plt.title('MSE over time')
    plt.legend()
    plt.grid(True)
    plt.show()