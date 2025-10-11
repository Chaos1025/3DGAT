import os
import sys
import yaml

import torch
from torch.fft import fftn, ifftn, fftshift, ifftshift
import tifffile as tiff
import numpy as np
from scipy.io import loadmat

from opt import get_opts
from datasets import dataset_dict


if __name__ == "__main__":

    hparams = get_opts()
    config_file = os.path.join(hparams.root_dir, "deconvolve.yaml")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        setattr(hparams, key, value)
    hparams.output_file = "wiener_recon"

    device = torch.device("cuda:" + str(hparams.device) if torch.cuda.is_available() else "cpu")
    sample_path = os.path.join(hparams.root_dir, hparams.data_name)
    psf_path = os.path.join(hparams.root_dir, hparams.psf_name)
    lenslet_pos_path = os.path.join(hparams.root_dir, hparams.lenslet_idxs)

    print("Start loading data...")
    sample = tiff.imread(sample_path).astype(np.float32)
    sample = sample / sample.max()
    print("Finish Sample Loading...")
    lenslet_idxs = loadmat(lenslet_pos_path)["MLcenter_idxes"].astype(np.int16)
    print("Finish Index Loading...")
    psf_sparse = loadmat(psf_path)["H"]
    psf_sparse = psf_sparse / (psf_sparse.sum().sum() / len(lenslet_idxs))

    sample = torch.tensor(sample, device=device).unsqueeze(0)
    I_hat = fftshift(fftn(ifftshift(sample, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))

    print("Start Wiener filtering...")
    noise_power = 1e-4
    m, n = psf_sparse[0, 0, 1].todense().shape
    p = psf_sparse.shape[-1]
    M, N = hparams.sensor_field_size
    sample_r = torch.zeros([p, M, N])
    for layer in range(p):
        psf_layer = psf_sparse[0, 0, layer].todense()
        psf_layer = psf_layer[(m - M) // 2 : (m - M) // 2 + M, (n - N) // 2 : (n - N) // 2 + N]
        psf_tensor = torch.tensor(psf_layer, device=device).unsqueeze(0)
        H_hat = fftshift(fftn(ifftshift(psf_tensor, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))
        I_r = (I_hat * torch.conj(H_hat)) / (torch.abs(H_hat) ** 2 + noise_power)
        sample_r[layer] = fftshift(ifftn(ifftshift(I_r, dim=(1, 2)), dim=(1, 2)), dim=(1, 2)).abs()

    sample_recon = sample_r.cpu().numpy()
    print("Full reconstruction shape:", sample_recon.shape)
    D, H, W = sample_recon.shape
    d, h, w = hparams.object_voxel_size
    recon_crop = sample_recon[
        (D - d) // 2 : (D + d) // 2, (H - h) // 2 : (H + h) // 2, (W - w) // 2 : (W + w) // 2
    ]
    print("Crop it to:", recon_crop.shape)
    recon_crop = recon_crop / recon_crop.max()
    recon_save = np.uint16(recon_crop * 65535)

    wiener_file = os.path.join(hparams.root_dir, hparams.output_file + ".tif")
    tiff.imwrite(wiener_file, recon_save)
    print(f"Writing Wiener Filtering result to {wiener_file}")
    print("One-step Wiener filtering done!")
