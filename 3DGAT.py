import torch
from torch import nn
from opt import get_opts
import os
import time
import tifffile as tiff
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

from datasets import dataset_dict
from misc.propagator import PhysicalPropagator as PP
import misc.visualize as vis
from misc.losses import *
from misc.metrics import reconstruction_metric

from gaussian_renderer import query
from scene import GaussianModel, Scene
from utils import effective_rank

from torchmetrics import PeakSignalNoiseRatio
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import warnings

warnings.filterwarnings("ignore")
npimg2uint16 = lambda x: np.uint16(x / x.max() * 65535)
npimg2unit8 = lambda x: np.uint8(x / x.max() * 255)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


imagej_meta = {
    "axes": "ZYX",
    "unit": "um",
}
tiff_dict = {"imagej": True, "metadata": imagej_meta}
MIP_dict = {"mode": "max", "cmap": "inferno"}


class GSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.loss = {"L2": nn.MSELoss(), "L1": nn.L1Loss()}[hparams.main_loss]
        fourier_loss = {
            "fft": FFT_loss(
                shape=tuple([1] + hparams.sensor_field_size), dim=[1, 2], device="cuda"
            ),
            "fft_window": FFT_loss(
                shape=tuple([1] + hparams.sensor_field_size),
                with_window=True,
                dim=[1, 2],
                device="cuda",
            ),
        }
        self.fourier_loss = fourier_loss[hparams.fourier_loss]
        print(
            "Fourier loss type",
            hparams.fourier_loss,
            "with weight",
            hparams.fourier_weight,
            "is used",
        )
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.bbox = torch.tensor(hparams.bbox)
        self.scale_min_bound = hparams.scale_min_bound
        self.scale_max_bound = hparams.scale_max_bound

        self.densify_scale_threshold = (
            hparams.densify_scale_threshold * float(np.array(hparams.object_voxel_size).max())
            if hparams.densify_scale_threshold
            else None
        )
        tv_vol_size = hparams.tv_vol_size
        if hparams.initial_file.startswith("wiener_recon"):
            sample_path = os.path.join(hparams.root_dir, hparams.initial_file + ".tif")
        else:
            sample_path = os.path.join(
                "results", hparams.dataset_name, hparams.exp_name, hparams.initial_file + ".tif"
            )
        print("Load Initialize File from: ", sample_path)
        if os.path.exists(sample_path):
            sample = tiff.imread(sample_path)
            sample = sample.astype(np.float32) / sample.max()
            sample = torch.tensor(sample)
        else:
            sample = None

        if hparams.pretrain:
            init = tiff.imread(os.path.join(hparams.root_dir, hparams.init_name))
            init = init.astype(np.float32) / init.max()
            init = torch.tensor(init)

        # Object voxel size that should be reconstructed
        self.tv_vol_nVoxel = torch.tensor(hparams.object_voxel_size)
        self.tv_vol_sVoxel = (
            torch.tensor(hparams.object_physical_size)
            if hparams.object_physical_size
            else self.tv_vol_nVoxel
        )

        hparams.bbox = self.bbox

        self.gaussians = GaussianModel([self.scale_min_bound, self.scale_max_bound])
        self.scene = Scene(
            hparams,
            self.gaussians,
            init if hparams.pretrain else sample,
            load_iteration=None,
            init_from=hparams.init_from,
            ply_path=hparams.init_ply_path,
        )
        hparams.position_lr_max_steps = hparams.num_epochs
        hparams.density_lr_max_steps = hparams.num_epochs
        hparams.rotation_lr_max_steps = hparams.num_epochs
        hparams.scaling_lr_max_steps = hparams.num_epochs
        self.gaussians.training_setup(hparams)

        if hparams.start_checkpoint:
            model_params, self.first_iter = torch.load(hparams.start_checkpoint)
            self.gaussians.restore(model_params, hparams)

    def forward(self, batch, split):
        # forward projection process
        frames = batch["img_idxs"]
        lenslet_pos = self.lenslet_pos[frames]

        self.tv_vol_center = (self.bbox[0] + self.tv_vol_sVoxel / 2) + (
            self.bbox[1] - self.tv_vol_sVoxel - self.bbox[0]
        ) * torch.rand(3)

        vol_pkg = query(
            self.scene.gaussians,
            self.tv_vol_center,
            self.tv_vol_nVoxel,
            self.tv_vol_sVoxel,
            self.hparams,
        )
        vol_pred, viewspace_point_tensor, visibility_filter, radii = (
            vol_pkg["vol"],
            vol_pkg["viewspace_points"],
            vol_pkg["visibility_filter"],
            vol_pkg["radii"],
        )

        if split == "pretrain":
            prediction = vol_pred
        else:
            prediction = self.PP.propagate(
                vol_pred,
                psf_idxs=frames,
                lenslet=lenslet_pos,
            )

        return prediction, vol_pred, viewspace_point_tensor, visibility_filter, radii

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {
            "root_dir": self.hparams.root_dir,
            "data_name": self.hparams.data_name,
            "ref_name": self.hparams.ref_name,
            "psf_name": self.hparams.psf_name,
            "init_name": self.hparams.init_name if self.hparams.pretrain else None,
            "forward_mode": self.hparams.flfm_forward_mode,
            "data_size": self.hparams.object_voxel_size,
        }
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.psf_shape = self.train_dataset.psf_shape
        self.test_dataset = dataset(split="test", **kwargs)

        self.PP = PP(
            torch.Tensor(self.train_dataset.psf).cuda(0),
            self.tv_vol_nVoxel,
            self.psf_shape,
            forward_mode=self.hparams.flfm_forward_mode,
            sensor_field_size=self.hparams.sensor_field_size,
        )

        self.log_bar = partial(self.log, on_step=False, on_epoch=True, prog_bar=True)
        self.log_nobar = partial(self.log, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opts = []
        self.net_opt = self.gaussians.optimizer
        opts += [self.net_opt]
        return opts

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=4,
            persistent_workers=True,
            batch_size=None,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=4, batch_size=None, pin_memory=True)

    def on_train_start(self):
        self.register_buffer("mask", self.train_dataset.circular_mask.to(self.device))
        self.register_buffer("lenslet_pos", self.train_dataset.lenslet_idxes.to(self.device))

        self.start_time = time.time()
        if not self.hparams.no_save_test:
            self.val_dir = f"results/{self.hparams.dataset_name}/{self.hparams.exp_name}"
            os.makedirs(self.val_dir, exist_ok=True)
            self.logfile_dir = os.path.join(self.val_dir, "log")
            os.makedirs(self.logfile_dir, exist_ok=True)

        # Perform a pretrain loop
        tbar = tqdm(range(self.hparams.pretrain_steps))
        if self.hparams.pretrain:
            print("Pretraining started....")
            for i in tbar:
                for batch in self.pretrain_dataloader():
                    log = self.pretrain_step(batch, i)
                    tbar.set_description(
                        f"lr: {log['lr']:.2e} | loss: {log['loss']:.2e}, max: {log['max']:.3f}, min: {log['min']:.3f}"
                    )
            print("Pretrain finished")

    def training_step(self, batch, batch_nb, *args):
        self.gaussians.update_learning_rate(self.global_step)
        (
            results,
            vol_pred,
            self.viewspace_point_tensor,
            self.visibility_filter,
            self.radii,
        ) = self(batch, split="train")
        measurement = batch["rays"]

        # loss calculation
        loss_d = self.loss(results, measurement)
        loss = loss_d.mean()
        fourier_loss = self.fourier_loss(results, measurement)
        fourier_weight = self.hparams.fourier_weight * (
            (1 - self.global_step / self.hparams.num_epochs) * 0.8 + 0.2
        )
        total_loss = loss + fourier_loss * fourier_weight
        total_loss += ssim_loss(results, measurement) * self.hparams.ssim_weight
        gaussians_erank = effective_rank(self.gaussians.get_scaling**2)
        total_loss += (
            torch.mean(torch.pow(torch.log(4 - gaussians_erank), 4)) * self.hparams.erank_weight
        )
        total_loss /= 1 + self.hparams.ssim_weight

        with torch.no_grad():
            self.train_psnr(results, batch["rays"])
            if self.train_dataset.ref is not None:
                self.val_psnr(vol_pred / vol_pred.max(), batch["ref"])

            if (
                self.global_step % self.hparams.figure_interval == self.hparams.figure_interval - 1
                or self.global_step == 0
                or self.global_step == self.hparams.num_epochs - 1
            ):
                self.logger.experiment.add_histogram(
                    "density", self.gaussians.get_density, global_step=self.global_step
                )
                self.logger.experiment.add_histogram(
                    "scaling_erank", gaussians_erank, global_step=self.global_step
                )
                rand_idx = torch.randint(0, results.shape[0], (1,))
                if self.train_dataset.ref is not None:
                    vis.display_multistack_MIP(
                        [
                            measurement[rand_idx, ...],
                            results[rand_idx, ...],
                            vol_pred,
                            batch["ref"],
                        ],
                        title="meas-conv-pred-ref_" + str(self.global_step),
                        save_dir=self.logfile_dir,
                        **MIP_dict,
                    )
                else:
                    vis.display_multistack_MIP(
                        [measurement[rand_idx, ...], results[rand_idx, ...], vol_pred],
                        title="meas-conv-pred_" + str(self.global_step),
                        save_dir=self.logfile_dir,
                        **MIP_dict,
                    )
            if self.global_step == 0:
                initial_prediction = vol_pred.detach().cpu().numpy()
                tiff.imsave(f"{self.val_dir}/initial.tif", npimg2uint16(initial_prediction))

            max_value = torch.max(vol_pred)
            min_value = torch.min(vol_pred)
            max_density = self.gaussians.get_density.max()
            min_density = self.gaussians.get_density.min()
            max_gsize = self.gaussians.get_scaling.max()
            min_gsize = self.gaussians.get_scaling.min()
        self.log("lr", self.net_opt.param_groups[0]["lr"])
        self.log_nobar("Gaussians number", self.gaussians.get_xyz.shape[0])
        self.log_nobar("max", max_value)
        self.log_nobar("min", min_value)
        self.log_nobar("Gaussian max density", max_density.item())
        self.log_nobar("Gaussian min density", min_density.item())
        self.log_nobar("Gaussian max size", max_gsize.item())
        self.log_nobar("Gaussian min size", min_gsize.item())
        self.log_bar("train/loss", loss.item())
        self.log_bar("train/fourier", fourier_loss.item())
        self.log_nobar("train/psnr", self.train_psnr)
        if self.train_dataset.ref is not None:
            self.log_bar("val/psnr", self.val_psnr)

        return total_loss

    def on_train_epoch_end(self):
        with torch.no_grad():
            if self.viewspace_point_tensor.grad is not None:
                self.gaussians.max_radii2D[self.visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[self.visibility_filter],
                    self.radii[self.visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    self.viewspace_point_tensor, self.visibility_filter
                )
                grads = self.gaussians.xyz_gradient_accum / self.gaussians.denom
                grads[grads.isnan()] = 0.0

                if (
                    self.global_step > self.hparams.densify_from_iter
                    and self.global_step < self.hparams.densify_until_iter
                    and self.global_step != self.trainer.max_epochs - 1
                ):
                    if self.global_step % self.hparams.densification_interval == 0:
                        self.gaussians.densify_and_prune(
                            grads,
                            self.hparams.densify_grad_threshold,
                            self.hparams.density_min_threshold,
                            self.hparams.max_screen_size,
                            self.hparams.max_scale,
                            self.hparams.max_num_gaussians,
                            self.densify_scale_threshold,
                            self.bbox,
                        )
                        self.logger.experiment.add_histogram(
                            "gradient", grads.cpu().numpy(), global_step=self.global_step
                        )
                        print(f"Number of gaussians: {self.gaussians.get_xyz.shape[0]}")

            # Prune nan
            prune_mask = torch.isnan(self.gaussians.get_density).squeeze()
            if prune_mask.sum() > 0:
                self.gaussians.prune_points(prune_mask)

            prune_mask = (torch.isnan(self.gaussians.get_density)).squeeze()
            if prune_mask.sum() > 0:
                self.gaussians.prune_points(prune_mask)

    def on_validation_start(self):
        pass

    def validation_step(self, batch, batch_nb):
        recon_metric = reconstruction_metric("cuda:0")
        result, volume, _, _, _ = self(batch, split="valid")

        if self.train_dataset.ref is not None:
            recon_metric.get_metric(
                self.train_dataset.ref, volume / volume.max(), "3DGAT FLFM solution"
            )

        volume = volume.cpu().numpy()
        try:
            tiff.imwrite(
                f"{self.val_dir}/{self.hparams.output_file}.tif", npimg2uint16(volume), **tiff_dict
            )
        except:
            tiff.imwrite(f"{self.val_dir}/volume_16bit.tif", npimg2uint16(volume), **tiff_dict)

        with open(f"{self.val_dir}/log.txt", "a") as f:
            print(f"[source file]: {self.hparams.data_name}", file=f)
            print(f"[restored file]: {self.hparams.output_file}.tif", file=f)
            print(f"[max value]: {volume.max()}", file=f)

    def on_validation_end(self):
        hparams = self.hparams
        os.makedirs(f"ckpts/{hparams.dataset_name}/{hparams.exp_name}", exist_ok=True)
        torch.save(
            (self.gaussians.capture(), self.current_epoch),
            f"ckpts/{hparams.dataset_name}/{hparams.exp_name}"
            + "/chkpnt"
            + str(self.current_epoch)
            + ".pth",
        )
        self.scene.save(
            f"ckpts/{hparams.dataset_name}/{hparams.exp_name}",
            self.global_step,
            query,
            self.tv_vol_center,
            self.tv_vol_nVoxel,
            self.tv_vol_sVoxel,
            self.hparams,
        )
        print(self.gaussians.get_xyz.shape[0])

        self.end_time = time.time()
        print(f"\n Time taken for training: {self.end_time - self.start_time}s")

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        print("Progress bar dict: ", items)
        return items


if __name__ == "__main__":
    setup_seed(777)
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError("You need to provide a @ckpt_path for validation!")

    system = GSystem(hparams)

    ckpt_cb = ModelCheckpoint(
        dirpath=f"ckpts/{hparams.dataset_name}/{hparams.exp_name}",
        filename="{epoch:d}",
        save_weights_only=True,
        every_n_epochs=hparams.num_epochs,
        save_on_train_epoch_end=True,
        save_top_k=-1,
    )
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(
        save_dir=f"logs/{hparams.dataset_name}",
        name=hparams.exp_name,
        default_hp_metric=False,
    )

    trainer = Trainer(
        max_epochs=hparams.num_epochs,
        check_val_every_n_epoch=hparams.num_epochs,
        callbacks=callbacks,
        logger=logger,
        enable_model_summary=False,
        accelerator="gpu",
        devices=hparams.num_gpus,
        strategy="ddp" if hparams.num_gpus > 1 else "auto",
        num_sanity_val_steps=-1 if hparams.val_only else 0,
        precision=32,
    )
    trainer.fit(system)
