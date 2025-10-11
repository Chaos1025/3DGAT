from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """

    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.split = split

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith("train"):
            return 1
        return self.data_shape[0]

    def __getitem__(self, idx):
        if self.split.startswith("train"):
            img_idxs = np.arange(self.data_shape[0])
            pix_idxs = [
                np.random.choice(self.data_shape[1], self.batch_size),
                np.random.choice(self.data_shape[2], self.batch_size),
            ]
            rays = self.rays[img_idxs, ...]
            sample = {"img_idxs": img_idxs, "pix_idxs": pix_idxs, "rays": rays, "psf": self.psf}
            if self.ref is not None:  # if ground truth available
                sample["ref"] = self.ref[img_idxs, ...]
        else:
            img_idxs = np.arange(self.data_shape[0])
            sample = {
                "img_idxs": img_idxs,
                "rays": self.rays[img_idxs, ...],
            }
            if self.ref is not None:  # if ground truth available
                sample["ref"] = self.ref[img_idxs, ...]

        return sample
