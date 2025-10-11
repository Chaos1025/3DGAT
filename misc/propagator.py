"""Module for simulating physical propagation in optical systems."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.fft import fftn, ifftn, fftshift, ifftshift

PropogationModality = ["FLFM"]


def get_half_even(x):
    """Compute half of x, rounding up if x is odd."""
    return x // 2 if x % 2 == 0 else (x + 1) // 2


class PhysicalPropagator:
    """
    A class for simulating physical propagation in optical systems.

    Parameters:
        psf (torch.Tensor): Point Spread Function (PSF) of the system.
        obj_shape (tuple): Shape of the object (e.g., the 3D volume to be propagated).
        psf_shape (tuple): Shape of the PSF.
        forward_mode (str, optional):
            Options are 'tile', 'patch', 'total'. Default is 'patchpatch'.
        sensor_field_size (tuple, optional): Size of the sensor field.
            Only required in 'patch' forward mode.
    """

    def __init__(
        self,
        psf: torch.Tensor,
        obj_shape: tuple,
        psf_shape: tuple,
        forward_mode="patch",
        sensor_field_size=None,
    ):
        self.obj_shape = obj_shape
        self.psf_shape = psf_shape
        self.forward_mode = forward_mode
        self.sensor_field_size = sensor_field_size
        self.patcher = Patcher(sensor_field_size)

        print("Given PSF shape: ", psf.shape, "; Provided PSF shape: ", psf_shape)
        print("Provided obj shape: ", obj_shape)

        # Construct padding functions based on modality
        self.flfm_dim = [-2, -1]
        self._construct_paddings_FLFM()
        # Convert PSF to OTF
        self.otf = self.psf2otf(psf)
        return None

    def _construct_paddings_FLFM(self):
        """Construct padding functions for Fourier Light Field Microscopy modality."""
        # Create circular mask
        self.circular_mask = self.create_circular_mask(*self.obj_shape[1:])

        # Check that object and PSF have the same number of axial slices
        assert (
            self.obj_shape[0] == self.psf_shape[0]
        ), f"Object ({self.obj_shape[0]} slices) and PSF ({self.psf_shape[0]} slices) should have the same number of axial slices"

        # Compute half sizes
        _, h_obj, w_obj = (get_half_even(i) for i in self.obj_shape)
        _, h_psf, w_psf = (get_half_even(i) for i in self.psf_shape)

        # Define padding functions
        pad_size_psf = (w_obj, w_obj, h_obj, h_obj, 0, 0)
        self.psf_padder = functools.partial(F.pad, pad=pad_size_psf, mode="constant", value=0)

        pad_size_obj = (w_psf, w_psf, h_psf, h_psf, 0, 0)
        self.obj_padder = functools.partial(F.pad, pad=pad_size_obj, mode="constant", value=0)

        # Define cropping function
        if self.forward_mode in ["total"]:
            self.obj_cropper = lambda x: x[:, h_obj:-h_obj, w_obj:-w_obj]
        elif self.forward_mode in ["tile"]:
            self.obj_cropper = lambda x: x[:, h_psf:-h_psf, w_psf:-w_psf]
        elif self.forward_mode in ["patch"]:
            self.obj_cropper = lambda x: x
        else:
            raise NotImplementedError(
                f"Forward mode '{self.forward_mode}' has not been implemented"
            )

    def create_circular_mask(self, h, w, center=None, radius=None):
        """Create a circular mask"""
        if center is None:
            center = (w // 2, h // 2)
        if radius is None:
            radius = min(center[0], center[1], w - center[0], h - center[1])
        Y, X = torch.meshgrid(torch.arange(h), torch.arange(w))
        dist_from_center = torch.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        mask = dist_from_center <= radius
        return mask.cuda()

    def psf2otf(self, psf: torch.Tensor) -> torch.Tensor:
        """
        Convert PSF from spatial domain to OTF in frequency domain.

        Parameters:
            psf (torch.Tensor): The Point Spread Function.

        Returns:
            torch.Tensor: The Optical Transfer Function.
        """
        psf_pad = self.psf_padder(psf)
        otf_stack = fftshift(
            fftn(ifftshift(psf_pad, dim=self.flfm_dim), dim=self.flfm_dim), dim=self.flfm_dim
        )  # [V, Z, kY, kX]
        return otf_stack

    def propagate(self, obj: torch.Tensor, y_max=1, psf_idxs=None, lenslet=None):
        """
        Propagate the object using the OTF.

        Parameters:
            obj (torch.Tensor): The object to be propagated.
            y_max (float, optional): Normalization factor for the output.
            psf_idxs (torch.Tensor, optional): Indices for selecting PSFs of FLFM.
            lenslet (torch.Tensor, optional): Lenslet positions for FLFM modality.

        Returns:
            torch.Tensor: The propagated result.
        """
        return self._propagate_FLFM(obj, y_max, psf_idxs, lenslet)

    def _propagate_FLFM(
        self, obj: torch.Tensor, y_max: float, psf_idxs: torch.Tensor, lenslet: torch.Tensor = None
    ):
        """Propagate using Fourier Light Field Microscopy modality."""
        if psf_idxs is None:
            raise ValueError("psf_idxs must be provided for FLFM modality")
        # Select the relevant OTFs
        otf = self.otf[psf_idxs, ...]  # Shape: [N, Z, kY, kX]
        obj = obj.unsqueeze(0)
        obj_padded = self.obj_padder(obj)
        obj_ft = fftshift(
            fftn(ifftshift(obj_padded, dim=self.flfm_dim), dim=self.flfm_dim), dim=self.flfm_dim
        )  # [Z, kY, kX]
        obj_ft_sum = torch.sum(obj_ft * otf, dim=1)  # [N, kX, kY]
        results = torch.abs(
            fftshift(
                ifftn(ifftshift(obj_ft_sum, dim=self.flfm_dim), dim=self.flfm_dim),
                dim=self.flfm_dim,
            )
        )
        result = self.obj_cropper(results)  # [N, Y, X]

        if self.forward_mode == "patch":
            # In patch mode, assemble sub-views into sensor field
            result, _ = self.patcher.patch(result, lenslet)
            return result.unsqueeze(0)  # [1, M, N]

        return result


class Patcher:
    """Only suitable for FLFM forward mode 'patch'"""

    def __init__(self, image_size, device="cuda"):
        self.image_size = image_size  # Shape: [H, W]
        self.zeros = torch.zeros(self.image_size, device=device)

    def patch(self, patches, positions):
        assert (
            patches.shape[0] == positions.shape[0]
        ), "Input patches number not match with initialization"
        image = self.zeros.clone()
        mask = self.zeros.clone()
        for i, l in enumerate(positions):
            image = self.single_patch(image, l, patches[i])  # repair each patch into their place
            mask = self.single_patch(mask, l, torch.ones_like(patches[i]))
        return image, (mask > 0)

    def single_patch(self, image, position, patch):
        # image: [H, W] size tensor
        # position: [2] tensor, (y, x)
        # patch: [pH, pW] size tensor, small than image
        h, w = image.shape
        ph, pw = patch.shape
        y, x = position
        # Compute the region of interest
        image_y_start = y - ph // 2
        patch_y_start = 0
        if image_y_start < 0:
            image_y_start = 0
            patch_y_start = ph // 2 - y
        image_y_end = y + ph // 2
        patch_y_end = ph
        if image_y_end > h:
            image_y_end = h
            patch_y_end = ph - (y + ph // 2 - h)

        image_x_start = x - pw // 2
        patch_x_start = 0
        if image_x_start < 0:
            image_x_start = 0
            patch_x_start = pw // 2 - x
        image_x_end = x + pw // 2
        patch_x_end = pw
        if image_x_end > w:
            image_x_end = w
            patch_x_end = pw - (x + pw // 2 - w)

        # Copy the patch to the image
        image[image_y_start:image_y_end, image_x_start:image_x_end] += patch[
            patch_y_start:patch_y_end, patch_x_start:patch_x_end
        ]

        return image
