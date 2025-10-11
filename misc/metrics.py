"""File to implement SSIM and mSSIM metrics for 3D images."""

import torch
import torch.nn.functional as F
from math import exp
import numpy as np
import lpips
import torch.nn as nn
from typing import Union

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def create_3d_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(-1)
    _3D_window = (_2D_window * _1D_window.t().unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, depth, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, depth, height, width)
        window = create_3d_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv3d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv3d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs.mean(1).mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(
            img1,
            img2,
            window_size=window_size,
            size_average=size_average,
            full=True,
            val_range=val_range,
        )

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool3d(img1, (2, 2, 2))
        img2 = F.avg_pool3d(img2, (2, 2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs**weights
    pow2 = ssims**weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(
            img1, img2, window=window, window_size=self.window_size, size_average=self.size_average
        )


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


def ssim_d(img1, img2):
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    mu1_mu2 = mu1 * mu2

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = torch.mean(img1 * img1) - mu1_sq
    sigma2_sq = torch.mean(img2 * img2) - mu2_sq
    sigma12 = torch.mean(img1 * img2) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    loss = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / torch.clamp(
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2), min=1e-8, max=1e8
    )

    return loss


def layer_wise_PSNR(gt: torch.Tensor, img: torch.Tensor):
    assert gt.shape == img.shape, "Two images must have the same shape"
    d = gt.shape[0]
    psnr_list = torch.zeros(d)
    for i in range(d):
        mse = torch.mean((gt[i, ...] - img[i, ...]) ** 2)
        psnr_list[i] = -10 * torch.log10(mse)
    return psnr_list


class lpips_metric:
    def __init__(self, device=DEVICE):
        self.lpips_vgg = lpips.LPIPS(net="vgg").to(device)
        self.lpips_alex = lpips.LPIPS(net="alex").to(device)

    def compute(self, img1: torch.Tensor, img2: torch.Tensor):
        try:
            d = img1.shape[0]
            lpips_list = np.zeros((d, 2))
            for i in range(d):
                vgg = self.lpips_vgg(img1[i, ...], img2[i, ...]).mean().item()
                alex = self.lpips_alex(img1[i, ...], img2[i, ...]).mean().item()
                lpips_list[i] = np.array([vgg, alex])
            return np.array(lpips_list)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory")
            else:
                raise e

    def get_metric(self, gt: torch.Tensor, img: torch.Tensor, img_name: str = "", layer_range=None):
        assert (
            gt.shape == img.shape
        ), "Two images must have the same shape, but got {} and {}".format(gt.shape, img.shape)
        lpips_d = self.compute(gt, img)
        vgg_list = lpips_d[:, 0]
        alex_list = lpips_d[:, 1]
        if layer_range is not None:
            assert (
                layer_range[0] >= 0 and layer_range[1] <= gt.shape[0]
            ), "layer_range must be in [0, {}], but got {}".format(gt.shape[0], layer_range)
            vgg_list = vgg_list[layer_range[0] : layer_range[1]]
            alex_list = alex_list[layer_range[0] : layer_range[1]]

        print(
            f"{img_name} LPIPS vs GT: VGG={vgg_list.mean():.4e}, Alex={alex_list.mean():.4e}, the smaller the better"
        )
        return [vgg_list.mean(), alex_list.mean()]

    def get_slice_metric(self, gt: torch.Tensor, img: torch.Tensor, img_name: str = ""):
        assert (
            gt.shape == img.shape
        ), "Two images must have the same shape, but got {} and {}".format(gt.shape, img.shape)
        lpips_d = self.compute(gt, img)
        return lpips_d


class reconstruction_metric:
    def __init__(self, device=DEVICE):
        self.lpips = lpips_metric(device=device)
        self.mse = nn.MSELoss()
        self.layer_wise_PSNR = layer_wise_PSNR
        self.ssim = ssim_d
        self.device = device

    def get_metric(
        self,
        gt: Union[np.array, torch.Tensor],
        img: Union[np.array, torch.Tensor],
        img_name: str = "",
        layer_range=None,
    ):
        assert (
            gt.shape == img.shape
        ), "Two images must have the same shape, but got {} and {}".format(gt.shape, img.shape)
        if isinstance(gt, np.ndarray):
            gt = torch.tensor(gt).float()
        gt = gt.to(self.device)
        if isinstance(img, np.ndarray):
            img = torch.tensor(img).float().to(self.device)
        img = img.to(self.device)

        if len(gt.shape) == 3:
            gt = gt.unsqueeze(1)
            img = img.unsqueeze(1)
        if len(gt.shape) == 2:
            gt = gt.unsqueeze(0).unsqueeze(0)
            img = img.unsqueeze(0).unsqueeze(0)

        PSNR_layer_wise = self.layer_wise_PSNR(gt, img)
        if layer_range is not None:
            PSNR_layer_mean = PSNR_layer_wise[layer_range[0] : layer_range[1]].mean().item()
        else:
            PSNR_layer_mean = PSNR_layer_wise.mean().item()
        mse_d = self.mse(gt, img).item()
        ssim_d = self.ssim(gt, img).item()
        print(f"{img_name} PSNR vs GT: {-10*np.log10(mse_d):.4f}dB, the larger the better")
        print(f"{img_name} PSNR layer-wise vs GT: {PSNR_layer_mean:.4f}dB, the larger the better")
        print(f"{img_name} SSIM vs GT: {ssim_d:.4f}, the larger the better")
        lpips_d = self.lpips.get_metric(gt, img, img_name, layer_range=layer_range)
        return mse_d, ssim_d, lpips_d
