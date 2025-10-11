# python file for Fourier Light-field Microscopy Deconvolution
# Using Richardson-Lucy Deconvolution (RLD) Algorithm

import sys
from os.path import join
import os
import time
import yaml

import torch
from torch.fft import fftn, ifftn, fftshift, ifftshift
from torchvision import transforms
import tifffile as tiff
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt

from opt import get_opts


def img2psnr(img1, img2):
    img1 = img1 / img1.max()
    img2 = img2 / img2.max()
    mse = torch.mean(torch.square(img1 - img2))
    return -10 * torch.log10(mse).item()


class DeconvolutionRL:
    def __init__(self, psf, psf_t, hparams, device="cpu"):
        data_shape = hparams.object_voxel_size
        self.device = device
        self.psf_shape = psf.shape
        self.otf = self._psf2otf(psf, [i // 2 for i in data_shape])
        self.otf_t = self._psf2otf(psf_t, [i // 2 for i in data_shape])
        print("OTF shape: ", self.otf.shape)

        whole_shape = psf.shape
        crop_size = []
        for i in range(len(data_shape)):
            crop_size.append((whole_shape[i] - data_shape[i]) // 2)
            crop_size.append((whole_shape[i] + data_shape[i]) // 2)
        self.crange = crop_size
        self.file_dir = join("results", hparams.dataset_name, hparams.exp_name)
        print("Reconstruction shape after crop: ", data_shape)

        del psf, psf_t
        torch.cuda.empty_cache()

    def _psf2otf(self, psf, pad_size):
        otf = ifftshift(fftn(fftshift(psf, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))
        return otf

    def _forward(self, x):
        ft = ifftshift(fftn(fftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))
        ft.mul_(self.otf)
        fpj = ifftshift(
            ifftn(fftshift(torch.sum(ft, dim=0), dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2)
        ).real
        return fpj

    def _backward(self, x):
        ft = ifftshift(fftn(fftshift(x, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))
        bpj = ifftshift(
            ifftn(fftshift(self.otf_t * ft, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2)
        ).real
        return bpj

    def _crop_file(self, file):
        return file[
            self.crange[0] : self.crange[1],
            self.crange[2] : self.crange[3],
            self.crange[4] : self.crange[5],
        ]

    def _write_file(self, file, iter, file_name):
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)
        with open(f"{self.file_dir}/log.txt", "a") as f:
            print(f"[file number]: {file_name.split('seq')[-1]}", file=f)
            print(f"[iter number]: {iter}", file=f)
            print(f"[max value]: {file.max()}", file=f)
        filepath = join(self.file_dir, f"RL-{file_name}-{iter}.tif")
        file = file / file.max() * 65535
        tiff.imwrite(filepath, file.astype(np.uint16))
        print("Write file to", filepath)

    def deconvolve(self, data, iter, file_name, ref=None):
        assert isinstance(iter, (int, list))
        if isinstance(iter, int):
            iter = [iter]

        self.val_list = []
        validation = ref is not None
        recon = torch.ones(self.psf_shape, device=self.device)  # Initialize in-place
        print("Start deconvolution...")

        tbar = tqdm(range(max(iter)), desc="RL Deconvolution: ")
        for i in tbar:
            temp = time.time()
            fpj = self._forward(recon)
            error = data / fpj
            error[torch.isnan(error)] = 0
            error[torch.isinf(error)] = 0
            error = torch.nn.functional.relu(error)
            del fpj
            torch.cuda.empty_cache()

            bpj = self._backward(error)

            recon.mul_(bpj)
            ttime = time.time() - temp
            # Manually release variables and empty cache
            del error, bpj
            torch.cuda.empty_cache()

            reconCrop = self._crop_file(recon)
            if validation:
                psnr = img2psnr(reconCrop, ref)
                self.val_list.append(psnr)

            if i + 1 in iter:
                recon_save = reconCrop.cpu().numpy()
                self._write_file(recon_save, i + 1, file_name)

            basic_fstr = f"iter={i}, ttime={ttime:.2f}s, maxv={reconCrop.max().item():.3f}"
            fstr = basic_fstr + f", validation psnr={psnr:.2f}dB" if validation else basic_fstr
            tbar.set_postfix_str(fstr)

        if validation:
            self._log()
        return True

    def _log(self):
        data = {"PSNR": self.val_list}
        df = pd.DataFrame(data)
        df.to_csv(join(self.file_dir, "val_RL.csv"), index=True)

        plt.figure()
        plt.plot(self.val_list, label="PSNR")
        plt.legend()
        plt.title("VALIDATION for RL DECONVOLUTION")
        plt.grid()
        plt.savefig(join(self.file_dir, "psnr.png"))
        return


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    iter = [20, 50, 100]
    hparams = get_opts()
    config_file = os.path.join(hparams.root_dir, "deconvolve.yaml")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        setattr(hparams, key, value)

    hparams.exp_name = os.path.join("RL", hparams.exp_name)

    print("RL deconvolution starting!!!")
    hparams.device = 4
    device = torch.device(
        "cuda:" + str(hparams.device)
        if torch.cuda.is_available() and hparams.device != "cpu"
        else "cpu"
    )
    start_time = time.time()
    psf_path = join(hparams.root_dir, hparams.psf_name)
    lenslet_pos_path = join(hparams.root_dir, hparams.lenslet_idxs)
    data_shape = hparams.object_voxel_size
    print("real data shape: ", data_shape)

    lenslet_idxs = loadmat(lenslet_pos_path)["MLcenter_idxes"].astype(np.int16)
    print("Processing sparse PSF...")
    psf_sparse = loadmat(psf_path)["H"]
    m, n = psf_sparse[0, 0, 1].todense().shape
    psf_stack = []
    M, N = hparams.sensor_field_size
    for layer in psf_sparse[0, 0, :]:
        dense_layer = layer.todense()
        dense_layer = dense_layer[(m - M) // 2 : (m - M) // 2 + M, (n - N) // 2 : (n - N) // 2 + N]
        psf_stack.append(np.array(dense_layer))
    psf = np.stack(psf_stack, axis=0)  # Shape: (Z, Y, X)
    psf = psf / (psf.sum() / len(lenslet_idxs))

    print("Convert to tensor and move to device...")
    psf = torch.tensor(psf, device=device).requires_grad_(False)
    psf_t = transforms.functional.rotate(psf, 180)

    DeconvolerRL = DeconvolutionRL(psf, psf_t, hparams, device=device)
    del psf, psf_t, psf_sparse, psf_stack
    torch.cuda.empty_cache()

    sample = tiff.imread(join(hparams.root_dir, hparams.data_name)).astype(np.float32)
    sample = sample / sample.max()
    sample = torch.tensor(sample, device=device).unsqueeze(0).requires_grad_(False)  # [1, Y, X]

    ref_path = join(hparams.root_dir, hparams.ref_name)
    if os.path.exists(ref_path):
        ref = tiff.imread(ref_path).astype(np.float32)
        ref = ref / ref.max()
        ref = torch.tensor(ref, device=device).requires_grad_(False)
    else:
        ref = None

    DeconvolerRL.deconvolve(sample, iter, hparams.output_file, ref)

    total_time = time.time() - start_time
    print(f"Total time consumption: {total_time:.2f}")
