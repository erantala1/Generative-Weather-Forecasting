# diffusion/scheduler.py
# linear noise scheduler used in unet_diffusion.py

import torch

class LinearNoiseScheduler():
  def __init__(self, beta_start, beta_end, timesteps, device):
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.timesteps = timesteps
    self.betas = torch.linspace(beta_start,beta_end,timesteps, device=device)
    self.alphas = 1. - self.betas
    self.alpha_product = torch.cumprod(self.alphas,dim=0)
    self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_product)
    self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1-self.alpha_product)
    self.device = device

  def add_noise(self,data,noise,t):
    data_shape = data.shape
    batch_size = data_shape[0]
    sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)
    sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size)
    for _ in range(len(data_shape) - 1):
        sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
    return (sqrt_alpha_cum_prod * data + sqrt_one_minus_alpha_cum_prod * noise)

  def sample_prev_timestep(self, xt, noise_pred, t):

        alpha_cum_prod = self.alpha_product

        x0 = ((xt - (self.sqrt_one_minus_alpha_cum_prod[t].to(xt.device) * noise_pred)) /
              torch.sqrt(alpha_cum_prod[t]))
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - ((self.betas[t].to(xt.device)) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t].to(xt.device))
        mean = mean / torch.sqrt(self.alphas[t].to(xt.device))

        if t == 0:
            return mean, x0
        else:
            variance = (1 - alpha_cum_prod[t - 1]) / (1.0 - alpha_cum_prod[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0