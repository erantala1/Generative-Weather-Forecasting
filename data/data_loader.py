# data/data_loader.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
import netCDF4 as nc

def load_data():
    drive.mount('/content/drive')
    data_79 = nc.Dataset('/content/drive/MyDrive/Copy of z1979.nc')
    data_80 = nc.Dataset('/content/drive/MyDrive/Copy of z1980.nc')
    data_81 = nc.Dataset('/content/drive/MyDrive/Copy of z1981.nc')
    data_85 = nc.Dataset('/content/drive/MyDrive/Copy of z1985.nc')
    data_83 = nc.Dataset('/content/drive/MyDrive/Copy of z1983.nc')

    z_data_79 = data_79.variables['z'][:]
    z_data_80 = data_80.variables['z'][:]
    z_data_81 = data_81.variables['z'][:]
    z_data_85 = data_85.variables['z'][:]
    z_data_83 = data_83.variables['z'][:]

    time_steps = [0, 30, 50]
    fig, axes = plt.subplots(1, len(time_steps), figsize=(15, 5))

    for i, t in enumerate(time_steps):
        spatial_data = z_data_85[t, 0, :, :]
        ax = axes[i]
        cs = ax.contourf(spatial_data, cmap='coolwarm')
        plt.colorbar(cs, ax=ax)
        ax.set_title(f'Time step {t}')

    plt.show()
    '''
    print(data_79.variables['z'][:].shape)
    print(data_80.variables['z'][:].shape)
    print(data_81.variables['z'][:].shape)
    print(data_83.variables['z'][:].shape)
    print(data_85.variables['z'][:].shape)
    '''
    return z_data_79, z_data_80, z_data_81, z_data_83, z_data_85

def normalize_data(data):
    mean = data.mean(dim=(0, 2, 3), keepdim=True)
    std = data.std(dim=(0, 2, 3), keepdim=True)
    return (data - mean) / (std + 1e-8), mean, std

def denormalize_data(data, mean, std):
    return data * (std + 1e-8) + mean

def corrupt_data(data, k):
  sparse_data = data.clone()
  time, channel, latitude, longitude = sparse_data.shape
  for t in range(time):
    indices1 = np.random.randint(0,latitude,k)
    indices2 = np.random.randint(0,longitude,k)
    for i in range(k):
      sparse_data[t,:,indices1[i],indices2[i]] = 0
  return torch.tensor(sparse_data,dtype=torch.float32)