import argparse
from os.path import join
from os.path import exists
import yaml


def get_opts():
    parser = argparse.ArgumentParser(description="3D Gaussian for Microscopy Reconstruction")

    parser.add_argument(
        "--config", type=str, default="params.yaml", help="File name of the config file"
    )
    # dataset parameters
    parser.add_argument(
        "--root_dir", type=str, default="./data/flfm/reticular", help="root directory of dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default="LFimage.tif", help="name of the source data file"
    )
    parser.add_argument(
        "--ref_name", type=str, default="data.tif", help="name of the reference data file, if any"
    )
    parser.add_argument("--psf_name", type=str, default="psf.mat", help="name of the psf file")
    parser.add_argument(
        "--lenslet_idxs",
        type=str,
        default="MLcenter_idxes.mat",
        help="name of the lenslet idx file",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="flfm",
        choices=["flfm"],
        help="which dataset to train/test",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "trainval", "trainvaltest"],
        help="use which split to train",
    )
    # physical options
    parser.add_argument(
        "--flfm_forward_mode",
        type=str,
        default="patch",
        choices=["tile", "patch", "total"],
        help="forward model of flfm",
    )
    parser.add_argument(
        "--sensor_field_size",
        nargs="+",
        type=int,
        default=[2048, 2048],
        help="sensor field size, pixel number on each axis, [W, H]",
    )
    parser.add_argument(
        "--object_voxel_size",
        nargs="+",
        type=int,
        default=[50, 616, 616],
        help="object voxel size, voxel number on each axis, [Z, Y, X]",
    )
    parser.add_argument(
        "--object_physical_size",
        nargs="+",
        type=float,
        default=0,
        help="object physical size, um on each axis, [Z, Y, X]",
    )

    # model options
    parser.add_argument(
        "--pretrain", action="store_true", default=False, help="whether to use pretraining"
    )
    parser.add_argument("--init_name", type=str, default="init.tif")
    parser.add_argument(
        "--pretrain_steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--init_ply_path",
        type=str,
        default=None,
    )
    parser.add_argument("--output_file", type=str, default="volume_16bit")
    parser.add_argument("--initial_file", type=str, default="wiener_recon")
    parser.add_argument("--device", type=int, default=0, help="which gpu to use")
    # training options
    parser.add_argument("--num_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--lr", type=float, default=2e-3, help="learning rate")  # 1e-2 dy:1e-4
    parser.add_argument(
        "--main_loss", type=str, default="L2", choices=["L1", "L2"], help="Main loss function"
    )
    parser.add_argument(
        "--fourier_loss",
        type=str,
        default="fft",
        choices=["fft", "fft_window"],
        help="Fourier Domain Loss",
    )
    parser.add_argument(
        "--fourier_weight", type=float, default=0, help="weight for fourier domain loss"
    )
    parser.add_argument("--ssim_weight", type=float, default=0, help="weight for SSIM loss")
    parser.add_argument(
        "--erank_weight",
        type=float,
        default=0,
        help="weight for erank to constraint Gaussians shape",
    )

    # validation options
    parser.add_argument(
        "--val_only",
        action="store_true",
        default=False,
        help="run only validation (need to provide ckpt_path)",
    )
    parser.add_argument(
        "--no_save_test",
        action="store_true",
        default=False,
        help="whether to save test image and video",
    )
    # initialization options
    parser.add_argument(
        "--init_from",
        type=str,
        default="measur_100000",
        help="Gaussian Model Initialization Method",
    )
    parser.add_argument(
        "--init_threshold", type=float, default=0.1, help="Intensity threshold for initialization"
    )

    # misc
    parser.add_argument("--exp_name", type=str, default="./GS/3d012/", help="experiment name")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="pretrained checkpoint to load (including optimizers, etc)",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default=None,
        help="pretrained checkpoint to load (excluding optimizers, etc)",
    )
    parser.add_argument("--scale_min_bound", type=float, default=0.8, help="min_bound")
    parser.add_argument("--scale_max_bound", type=int, default=80, help="max_bound")
    parser.add_argument("--sVoxel", type=int, default=616, help="size of voxel")
    parser.add_argument("--nVoxel", type=int, default=616, help="number of voxel")
    parser.add_argument("--model_path", type=str, default="", help="model_path")
    parser.add_argument(
        "--densify_scale_threshold", type=float, default=0.05, help="densify_scale_threshold"
    )
    parser.add_argument("--densify_until_iter", type=int, default=15000, help="densify_until_iter")
    parser.add_argument("--densify_from_iter", type=int, default=90, help="densify_from_iter")
    parser.add_argument(
        "--densify_grad_threshold",
        type=float,
        default=1.0e-8,  ##5.0e-5
        help="densify_grad_threshold",
    )
    parser.add_argument(
        "--density_min_threshold",
        type=float,
        default=0.00001,  ##0.00001
        help="density_min_threshold",
    )
    parser.add_argument("--max_num_gaussians", type=int, default=500000, help="max_num_gaussians")
    parser.add_argument("--max_screen_size", type=str, required=None, help="max_screen_size")
    parser.add_argument("--max_scale", type=str, required=None, help="max_scale")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument(
        "--bbox",
        type=eval,
        default=[[-25, -308, -308], [25, 308, 308]],
        help="A list of lists of coordinates",
    )
    parser.add_argument("--tv_vol_size", type=int, default=160, help="tv_vol_size")
    parser.add_argument(
        "--compute_cov3D_python",
        action="store_true",
        default=False,
        help="compute 3D coveriance matrix in python",
    )
    parser.add_argument("--debug", action="store_true", default=False, help="debug")

    # Optimization options
    parser.add_argument(
        "--position_lr_init", type=float, default=2e-1, help="position_lr_init"  ###0.0002
    )
    parser.add_argument(
        "--position_lr_final", type=float, default=1e-3, help="position_lr_final"  ###0.00002
    )
    parser.add_argument(
        "--position_lr_max_steps", type=int, default=1000, help="position_lr_max_steps"
    )

    parser.add_argument(
        "--density_lr_init", type=float, default=2e-1, help="position_lr_init"  ###0.01
    )
    parser.add_argument(
        "--density_lr_final", type=float, default=1e-3, help="position_lr_final"  ###0.001
    )
    parser.add_argument(
        "--density_lr_max_steps", type=int, default=1000, help="position_lr_max_steps"
    )

    parser.add_argument(
        "--scaling_lr_init", type=float, default=2e-1, help="position_lr_init"  ###0.005
    )
    parser.add_argument(
        "--scaling_lr_final", type=float, default=1e-3, help="position_lr_final"  ###0.0005
    )
    parser.add_argument(
        "--scaling_lr_max_steps", type=int, default=1000, help="position_lr_max_steps"
    )

    parser.add_argument(
        "--rotation_lr_init", type=float, default=1e-1, help="position_lr_init"  ###0.001
    )
    parser.add_argument(
        "--rotation_lr_final", type=float, default=1e-3, help="position_lr_final"  ###0.0001
    )
    parser.add_argument(
        "--rotation_lr_max_steps", type=int, default=1000, help="position_lr_max_steps"
    )

    parser.add_argument(
        "--densification_interval", type=int, default=100, help="densification_interval"
    )

    parser.add_argument("--figure_interval", type=int, default=200, help="interval to save figures")

    # parser.add_argument('--N', type=int, default=120,
    #                     help='N')

    def post_parser(parser):
        args = parser.parse_args()
        config_file = join(args.root_dir, args.config)

        assert exists(config_file), f"Please specific correct Config File, {config_file} not exist"
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)

        d, h, w = args.object_physical_size if args.object_physical_size else args.object_voxel_size
        args.bbox = [[-d // 2, -h // 2, -w // 2], [d - d // 2, h - h // 2, w - w // 2]]

        return args

    return post_parser(parser)


if __name__ == "__main__":
    args = get_opts()
    # 打印所有参数
    for key, value in vars(args).items():
        print(f"{key}: {value}")
