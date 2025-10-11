import torch
import numpy as np
import os
from einops import rearrange
from scipy.io import loadmat
import tifffile as tiff
from .base import BaseDataset


class FLFMDataset(BaseDataset):
    """
    Fourier Light Field Microscopy (FLFM) dataset class.

    Parameters:
        root_dir (str): Root directory of the dataset.
        split (str): Dataset split ('train', 'test', etc.).
        data_name (str): Data file name.
        ref_name (str): Reference file name.
        psf_name (str): PSF file name.
        lenslet_idxes_name (str): Lenslet indices file name.
        init_name (str): Initialization file name.
        forward_mode (str): Forward mode ('tile', 'patch', 'total').

    Attributes:
        data (np.ndarray): Loaded data.
        ref (np.ndarray): Loaded reference data.
        psf (np.ndarray): Loaded PSF data.
        lenslet_idxes (torch.Tensor): Lenslet indices.
        psf_shape (tuple): Shape of the PSF.
        data_shape (tuple): Shape of the data.
        circular_mask (torch.Tensor): Circular mask for a single sub-view.
    """

    # Fourier Light Field Microscopy dataset
    # @forward_mode: both psf file name and sample file name are dependent on the forward_mode flag
    #   if True, the whole PSF with sparse matrix is used, psf_name should be "H.mat" and data_name should be "LFimage.tif"
    #    else the view-wise PSF is used, psf_name should be "psf.mat" and data_name should be "sample.tif"
    def __init__(
        self,
        root_dir,
        split="train",
        data_name="sample.tif",
        ref_name="data.tif",
        psf_name="psf.mat",
        lenslet_idxes_name="MLcenter_idxes.mat",
        init_name=None,
        forward_mode="tile",
        data_size=None,
        **kwargs,
    ):
        super().__init__(root_dir, split)
        self.data_name = data_name
        self.ref_name = ref_name
        self.psf_name = psf_name
        self.lenslet_idxes_name = lenslet_idxes_name
        self.forward_mode = forward_mode
        self.pretrain = init_name is not None
        self.init_name = init_name
        self.data_size = data_size

        self.read_intrinsics()
        self.read_meta()

    def __len__(self):
        if self.split.startswith("valid"):
            return 1
        return 1

    def __getitem__(self, idx):
        if self.split.startswith("train"):
            if self.forward_mode in ["total"]:
                img_idxs = np.arange(1)
            else:
                img_idxs = np.arange(self.N_frames)
            rays = self.rays if self.forward_mode == "patch" else self.rays[img_idxs]
            lenslet_pos = self.lenslet_idxes[img_idxs, ...]
            sample = {
                "img_idxs": img_idxs,
                "rays": rays,
                "psf_idxs": img_idxs,
                "lenslet_pos": lenslet_pos,
            }
            if self.ref is not None:  # if ground truth available
                sample["ref"] = self.ref
        elif self.split.startswith("pretrain"):
            sample = {"rays": self.init}
        else:
            if self.forward_mode in ["total"]:
                img_idxs = np.arange(1)
            else:
                img_idxs = np.arange(self.N_frames)

            rays = self.rays if self.forward_mode == "patch" else self.rays[img_idxs, ...]
            sample = {
                "img_idxs": img_idxs,
                "rays": rays,
            }
            if self.ref is not None:
                sample["ref"] = self.ref

        return sample

    def read_intrinsics(self):
        # Load reference data (if available)
        ref_path = os.path.join(self.root_dir, self.ref_name)
        if os.path.exists(ref_path):
            print("Reading reference from", ref_path)
            self.ref = tiff.imread(ref_path)
        else:
            self.ref = None

        # Load lenslet indices
        lenslet_path = os.path.join(self.root_dir, self.lenslet_idxes_name)
        print("Reading lenslet idxs from", lenslet_path)
        lenslet_idxes = loadmat(lenslet_path)["MLcenter_idxes"].astype(np.int16)
        self.lenslet_idxes = (
            torch.tensor(lenslet_idxes) - 1
        )  # Modification from matlab index mode to python

        if self.pretrain:
            init_path = os.path.join(self.root_dir, self.init_name)
            if os.path.exists(init_path):
                print("Reading init from", init_path)
                self.init = tiff.imread(os.path.join(self.root_dir, self.init_name))
                self.init = self.init / self.init.max()
            else:
                self.init = None

        # Load PSF and data based on forward mode
        psf_path = os.path.join(self.root_dir, self.psf_name)
        print("Reading psf from", psf_path)
        data_path = os.path.join(self.root_dir, self.data_name)
        print("Reading sample from", data_path)
        # Switch FLFM forward mode
        if self.forward_mode in ["total"]:
            sparse_psf = loadmat(psf_path)["H"]
            psf_stack = []
            for layer in sparse_psf[0, 0, :]:
                dense_layer = layer.todense()
                psf_stack.append(np.array(dense_layer))
            self.psf = np.stack(psf_stack, axis=0)[None, ...]  # Shape: (1, Z, Y, X)
            self.data = tiff.imread(data_path)[None, ...]  # Shape: (1, Y, X)
        elif self.forward_mode in ["patch"]:
            self.psf = loadmat(psf_path)["psf_splitted"].astype(np.float32)
            self.psf = rearrange(self.psf, "y x z v -> v z y x")
            self.data = tiff.imread(data_path)[None, ...]  # (1, Y, X)
        elif self.forward_mode in ["tile"]:
            self.psf = loadmat(psf_path)["psf_splitted"].astype(np.float32)
            self.psf = rearrange(self.psf, "y x z v -> v z y x")
            self.data = tiff.imread(data_path)  # (V, Y, X), V is the number of view points
            assert self.psf.shape[0] == lenslet_idxes.shape[0], "PSF and Lenslet number mismatch"
        else:
            raise NotImplementedError(
                f"Forward mode {self.forward_mode} is not implemented of FLFM"
            )

        self.psf_shape = self.psf.shape[1:]
        self.data_shape = self.data.shape

    def read_meta(self):
        self.N_frames = len(self.lenslet_idxes)
        self.rays = torch.FloatTensor(self.data / self.data.max())
        self.psf = torch.FloatTensor(self.psf)
        if self.pretrain and self.init is not None:
            self.init = torch.FloatTensor(self.init)

        # Normalize PSF energy
        self.psf = self.psf / (torch.sum(self.psf) / self.N_frames)
        if self.ref is not None:
            self.ref = self.ref / self.ref.max()
            self.ref = torch.FloatTensor(self.ref)

        # Create circular mask
        def create_circular_mask(h, w, center=None, radius=None):
            """Create a circular mask"""
            if center is None:
                center = (w // 2, h // 2)
            if radius is None:
                radius = min(center[0], center[1], w - center[0], h - center[1])
            Y, X = torch.meshgrid(torch.arange(h), torch.arange(w))
            dist_from_center = torch.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            mask = dist_from_center <= radius
            return mask

        self.circular_mask = create_circular_mask(self.data_size[-2], self.data_size[-1])


if __name__ == "__main__":
    dataset = FLFMDataset("data/flfm/dandelion")
    print(dataset.data_shape)
