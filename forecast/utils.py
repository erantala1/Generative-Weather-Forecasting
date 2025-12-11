# diffusion/utils.py

import torch
import torch.nn.functional as F

def apply_directional_consistency(x, prev_pred, prev_prev_pred=None):
        if prev_prev_pred is None:
            return x

        motion = prev_pred - prev_prev_pred
        momentum_factor = 0.3
        return x + motion * momentum_factor

def create_gaussian_kernel(size, sigma, device):
    coords = torch.arange(size).float() - (size - 1) / 2
    coords = coords.view(1, -1).expand(size, -1)
    coords_y = coords.t()

    kernel = torch.exp(-(coords**2 + coords_y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, size, size).repeat(2, 1, 1, 1)
    return kernel.to(device)

def apply_multi_scale_smoothing(x, step, smoothing_sizes, smoothing_sigmas, device):
    size = smoothing_sizes[min(step, len(smoothing_sizes)-1)]
    sigma = smoothing_sigmas[min(step, len(smoothing_sigmas)-1)]
    kernel = create_gaussian_kernel(size, sigma)
    padded = F.pad(x, (size//2, size//2, size//2, size//2), mode='reflect')
    smoothed = F.conv2d(padded, kernel, groups=2)
    if step > 1:
        large_kernel = create_gaussian_kernel(size*2-1, sigma*2, device)
        large_padded = F.pad(x, (size-1, size-1, size-1, size-1), mode='reflect')
        large_smoothed = F.conv2d(large_padded, large_kernel, groups=2)
        smoothed = smoothed * 0.7 + large_smoothed * 0.3
    blend_factor = min(0.3 + step * 0.05, 0.6)
    return x * (1 - blend_factor) + smoothed * blend_factor