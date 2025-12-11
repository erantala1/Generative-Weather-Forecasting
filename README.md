# Generative Weather Forecasting
Evan Rantala

Generative models for reconstructing and forecasting weather using NetCDF-4 reanalysis data. 
This project implements two models:

- **Denoising Variational Autoencoder (VAE)**  
- **U-Net Diffusion Model for autoregressive forecasting**

## Files Used for VAE:

- vae/
    - `model.py`                - Denoising VAE implementation
    - `train_vae.py`            - Main training loop

- forecast/
    - `vae_predict.py`          - Predict from corrupted data

## Files used for U-Net Diffusion:

- diffusion/
    - `scheduler.py`            - Linear noise scheduler module
    - `unet_diffusion.py`       - U-Net Diffusion module
    - `train_diffusion.py`      - Main training loop

- forecast/
    - `diffusion_predict.py`    - Sampling/prediction 
    - `utils.py`                - Common helper functions

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/erantala1/Generative-Weather-Forecasting
cd Generative-Weather-Forecasting
```
### 2. Create a virtual environment
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
### To train a model
```bash
cd vae
python train_vae.py
```
```bash
cd diffusion
python train_diffusion.py
```