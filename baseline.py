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


class VoxelModel(nn.Module):
    def __init__(self, hparams, initial=None, device="cuda"):
        super().__init__()
        self.object_voxel_size = hparams.object_voxel_size
        if initial is None:
            object_voxel = torch.ones(self.object_voxel_size, dtype=torch.float32, device=device)
        else:
            object_voxel = initial.to(device).float()

        self.object_voxel = nn.Parameter(object_voxel)
        self.object_voxel_size = tuple(self.object_voxel.shape)
        self.activation = nn.Softplus()
        self.device = device
        self.hparams = hparams

    def forward(self):
        return self.activation(self.object_voxel)

    def get_voxel(self):
        return self.activation(self.object_voxel).detach()


from torch.optim.lr_scheduler import _LRScheduler
from utils import get_expon_lr_func


class ExponAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, lr_init, lr_final, total_epochs, last_epoch=-1):
        self.total_epochs = total_epochs
        self.scheduler_args = get_expon_lr_func(lr_init, lr_final, max_steps=total_epochs)
        super(ExponAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.scheduler_args(self.last_epoch)]


class VoxelSystem(LightningModule):
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

        self.densify_scale_threshold = (
            hparams.densify_scale_threshold * float(np.array(hparams.object_voxel_size).max())
            if hparams.densify_scale_threshold
            else None
        )
        if hparams.initial_file.startswith("wiener_recon") or hparams.initial_file.startswith(
            "shift_and_multiply"
        ):
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

        self.voxel_size = hparams.object_voxel_size
        self.voxel_model = VoxelModel(hparams, initial=sample, device=self.device)

    def forward(self, batch, split):
        # forward projection process
        frames = batch["img_idxs"]
        lenslet_pos = self.lenslet_pos[frames]

        vol_pred = self.voxel_model()
        prediction = self.PP.propagate(
            vol_pred,
            psf_idxs=frames,
            lenslet=lenslet_pos,
        )
        return prediction, vol_pred

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
            self.voxel_size,
            self.psf_shape,
            forward_mode=self.hparams.flfm_forward_mode,
            sensor_field_size=self.hparams.sensor_field_size,
        )

        self.log_bar = partial(self.log, on_step=False, on_epoch=True, prog_bar=True)
        self.log_nobar = partial(self.log, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{"params": self.voxel_model.parameters(), "lr": self.hparams.lr}], eps=1e-15
        )
        scheduler = ExponAnnealingLR(
            optimizer,
            lr_init=self.hparams.lr,
            lr_final=1e-3,
            total_epochs=self.hparams.num_epochs,
        )
        return [optimizer], [scheduler]

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
        (results, vol_pred) = self(batch, "train")
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
                rand_idx = torch.randint(0, results.shape[0], (1,))
                # rand_idx = slice(0, 1)
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

        self.log_nobar("max", max_value)
        self.log_nobar("min", min_value)
        self.log_bar("train/loss", loss.item())
        self.log_bar("train/fourier", fourier_loss.item())
        self.log_nobar("train/psnr", self.train_psnr)
        self.log_bar("lr", self.optimizers().param_groups[0]["lr"])
        if self.train_dataset.ref is not None:
            self.log_bar("val/psnr", self.val_psnr)
        return total_loss

    def validation_step(self, batch, batch_nb):
        recon_metric = reconstruction_metric("cuda:0")
        result, volume = self(batch, split="valid")

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
        self.end_time = time.time()
        print(f"\n Time taken for training: {self.end_time - self.start_time}s")


if __name__ == "__main__":
    setup_seed(777)
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError("You need to provide a @ckpt_path for validation!")

    system = VoxelSystem(hparams)
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
