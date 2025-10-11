import numpy as np
from matplotlib import pyplot as plt
import torch
import os
import torch.nn as nn
import tifffile
import sys
from typing import Union
from skimage.transform import resize

sys.path.append("./")
# from utils import torch_to_np

proj_funcs = {"max": np.max, "mean": np.mean, "min": np.min}


def display_stack_MIP(
    obj: Union[np.ndarray, torch.Tensor],
    title: str,
    mode="max",
    cmap="gray",
    save_dir=None,
    axial_sclae_ratio=None,
    figsize=(15, 5),
):
    """
    Display the MIP of a 3D stack along three axes.

    Parameters:
    obj (numpy.ndarray or torch.Tensor): The 3D stack to be displayed.
    title (str): The title of the plot.
    mode (str, optional): The mode of projection. Options are 'max', 'mean', and 'min'. Default is 'max'.
    cmap (str, optional): The colormap to be used. Default is 'gray'.
    save_dir (str, optional): The directory where the plot will be saved. If None, the plot will be displayed. Default is None.

    Raises:
    ValueError: If the input object is not 3D or the save directory does not exist.
    """
    if type(obj) is torch.Tensor:
        # obj = torch_to_np(obj)
        obj = obj.detach().cpu().numpy()

    if not len(obj.shape) == 3:  # [Z, Y, X]
        raise ValueError("Input object is not 3D!")

    if save_dir is not None and not os.path.exists(save_dir):
        raise ValueError(f"Save directory ({save_dir}) does not exist!")

    fun = proj_funcs[mode]
    obj = (obj.astype(np.float32) - obj.min()) / (obj.max() - obj.min())  # normalize to [0, 1]

    proj_xy = fun(obj, 0)
    proj_xz = fun(obj, 1)
    proj_yz = fun(obj, 2)
    if axial_sclae_ratio is not None:
        proj_xz = resize(
            proj_xz,
            (proj_xz.shape[0] * axial_sclae_ratio, proj_xz.shape[1]),
            anti_aliasing=True,
        )
        proj_yz = resize(
            proj_yz,
            (proj_yz.shape[0] * axial_sclae_ratio, proj_yz.shape[1]),
            anti_aliasing=True,
        )

    plt.figure(figsize=figsize)
    plt.subplot(1, 3, 1)
    plt.imshow(proj_xy, cmap=cmap)
    plt.title("XY MIP")
    # plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(proj_xz, cmap=cmap)
    plt.title("XZ MIP")
    # plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(proj_yz, cmap=cmap)
    plt.title("YZ MIP")
    # plt.colorbar()
    plt.suptitle(title)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, title + ",MIP.png"), dpi=300)
        plt.close()
    else:
        plt.show()
    return None


def display_multistack_MIP(
    objs: Union[list, tuple],
    title: str,
    mode="max",
    cmap="gray",
    save_dir=None,
    axial_sclae_ratio=None,
    figsize=None,
):
    """
    Display the MIP of a list of 3D stacks along three axes,
    and compare them together.

    Parameters:
    obj (list or tuple): A group of 3D stacks to be displayed.
    title (str): The title of the plot.
    mode (str, optional): The mode of projection. Options are 'max', 'mean', and 'min'. Default is 'max'.
    cmap (str, optional): The colormap to be used. Default is 'gray'.
    save_dir (str, optional): The directory where the plot will be saved. If None, the plot will be displayed. Default is None.

    Raises:
    ValueError: If the input object is not 3D or the save directory does not exist.
    """
    if type(objs) is not list and type(objs) is not tuple:
        raise ValueError("Input object should be a list or tuple!")
    len_objs = len(objs)

    if len_objs == 0:
        raise ValueError("No input object!")

    for i in range(len_objs):
        if type(objs[i]) is torch.Tensor or nn.Parameter:
            objs[i] = objs[i].detach().cpu().numpy()
        if not len(objs[i].shape) == 3:
            raise ValueError(f"Input object {i} is not 3D!")
        objs[i] = objs[i] / objs[i].max()

    if save_dir is not None and not os.path.exists(save_dir):
        raise ValueError(f"Save directory ({save_dir}) does not exist!")
    column_titles = title.split("_")[0].split("-")
    assert (
        len(column_titles) == len_objs
    ), "Title must be in the format of 'title1-...-titlen_...', and match with objects"

    fun = proj_funcs[mode]
    fig = plt.figure(figsize=(5 * len_objs, 15) if figsize is None else figsize)
    plt.subplots_adjust(top=0.85, bottom=0.05, hspace=0.3)
    for i in range(1, len_objs + 1):
        proj_xy = fun(objs[i - 1], 0)
        proj_xz = fun(objs[i - 1], 1)
        proj_yz = fun(objs[i - 1], 2)
        if axial_sclae_ratio is not None:
            proj_xz = resize(
                proj_xz,
                (proj_xz.shape[0] * axial_sclae_ratio, proj_xz.shape[1]),
                anti_aliasing=True,
            )
            proj_yz = resize(
                proj_yz,
                (proj_yz.shape[0] * axial_sclae_ratio, proj_yz.shape[1]),
                anti_aliasing=True,
            )

        ax = plt.subplot(3, len_objs, i)
        if column_titles and i <= len(column_titles):
            ax.set_title(f"{column_titles[i-1]}", fontsize=14, pad=20)
        plt.imshow(proj_xy, cmap=cmap)
        plt.subplot(3, len_objs, i + len_objs)
        plt.imshow(proj_xz, cmap=cmap)
        plt.subplot(3, len_objs, i + 2 * len_objs)
        plt.imshow(proj_yz, cmap=cmap)
        if i == 1:
            # XY行标题
            plt.text(
                0.09,
                0.75,
                "XY MIP",
                rotation=90,
                transform=fig.transFigure,
                fontsize=14,
                horizontalalignment="center",
                verticalalignment="center",
            )
            # XZ行标题
            plt.text(
                0.09,
                0.45,
                "XZ MIP",
                rotation=90,
                transform=fig.transFigure,
                fontsize=14,
                horizontalalignment="center",
                verticalalignment="center",
            )
            # YZ行标题
            plt.text(
                0.09,
                0.16,
                "YZ MIP",
                rotation=90,
                transform=fig.transFigure,
                fontsize=14,
                horizontalalignment="center",
                verticalalignment="center",
            )

    plt.suptitle(title)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, title + ",multi-MIP.png"), dpi=300)
        plt.close()
    else:
        plt.show()
    return None


def display_xy_slice(
    obj: Union[np.ndarray, torch.Tensor],
    ref: Union[np.ndarray, torch.Tensor],
    slice_num,
    title="",
    cmap="gray",
    save_dir=None,
    figsize=None,
):
    """
    Display an xy slice of the given 3D objects.

    Parameters:
    obj (numpy.ndarray or torch.Tensor): The 3D object to be displayed.
    ref (numpy.ndarray or torch.Tensor): The 3D reference to be displayed.
    slice_num (int): The number of the slice to be displayed.
    title (str, optional): The title of the plot. Default is ''.
    save_dir (str, optional): The directory where the plot will be saved. If None, the plot will be displayed. Default is None.

    Raises:
    ValueError: If the input objects are not 3D or the save directory does not exist.
    """
    if type(obj) is torch.Tensor:
        obj = obj.detach().cpu().numpy()
    if type(ref) is torch.Tensor:
        ref = ref.detach().cpu().numpy()

    if not len(obj.shape) == 3 or not len(ref.shape) == 3:
        raise ValueError(f"Input object is not 3D, but {obj.shape} and {ref.shape}!")
    if save_dir is not None and not os.path.exists(save_dir):
        raise ValueError(f"Save directory ({save_dir}) does not exist!")

    obj_slice = obj[slice_num, :, :]
    obj_slice = obj_slice.astype(np.float32) / obj_slice.max()
    ref_slice = ref[slice_num, :, :]
    ref_slice = ref_slice.astype(np.float32) / ref_slice.max()
    plt.figure(figsize=(10, 5) if figsize is None else figsize)
    plt.subplot(1, 2, 1)
    plt.imshow(obj_slice, cmap=cmap)
    plt.title("Reconstruction")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(ref_slice, cmap=cmap)
    plt.title("Reference")
    plt.colorbar()
    plt.suptitle(title + f" (xy slice {slice_num})")
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, title + f",xy_slice{slice_num}.png"), dpi=300)
        plt.close()
    else:
        plt.show()
    return None


def display_yz_slice(
    obj: Union[np.ndarray, torch.Tensor],
    ref: Union[np.ndarray, torch.Tensor],
    slice_num,
    title="",
    cmap="gray",
    save_dir=None,
    figsize=None,
):
    """
    Display a yz slice of the given 3D objects.

    Parameters:
    obj (numpy.ndarray or torch.Tensor): The 3D object to be displayed.
    ref (numpy.ndarray or torch.Tensor): The 3D reference to be displayed.
    slice_num (int): The number of the slice to be displayed.
    title (str, optional): The title of the plot. Default is ''.
    save_dir (str, optional): The directory where the plot will be saved. If None, the plot will be displayed. Default is None.

    Raises:
    ValueError: If the input objects are not 3D or the save directory does not exist.
    """
    if type(obj) is torch.Tensor:
        obj = obj.detach().cpu().numpy()
    if type(ref) is torch.Tensor:
        ref = ref.detach().cpu().numpy()

    if not len(obj.shape) == 3 or not len(ref.shape) == 3:
        raise ValueError(f"Input object is not 3D, but {obj.shape} and {ref.shape}!")
    if save_dir is not None and not os.path.exists(save_dir):
        raise ValueError(f"Save directory ({save_dir}) does not exist!")

    obj_slice = obj[:, :, slice_num]
    obj_slice = obj_slice.astype(np.float32) / obj_slice.max()
    ref_slice = ref[:, :, slice_num]
    ref_slice = ref_slice.astype(np.float32) / ref_slice.max()
    plt.figure(figsize=(10, 5) if figsize is None else figsize)
    plt.subplot(1, 2, 1)
    plt.imshow(obj_slice, cmap=cmap)
    plt.title("Reconstruction")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(ref_slice, cmap=cmap)
    plt.title("Reference")
    plt.colorbar()
    plt.suptitle(title + f" (yz slice {slice_num})")
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, title + f",yz_slice{slice_num}.png"), dpi=300)
        plt.close()
    else:
        plt.show()
    return None


def display_zx_slice(
    obj: Union[np.ndarray, torch.Tensor],
    ref: Union[np.ndarray, torch.Tensor],
    slice_num,
    title="",
    cmap="gray",
    save_dir=None,
    figsize=None,
):
    """
    Display a zx slice of the given 3D objects.

    Parameters:
    obj (numpy.ndarray or torch.Tensor): The 3D object to be displayed.
    ref (numpy.ndarray or torch.Tensor): The 3D reference to be displayed.
    slice_num (int): The number of the slice to be displayed.
    title (str, optional): The title of the plot. Default is ''.
    save_dir (str, optional): The directory where the plot will be saved. If None, the plot will be displayed. Default is None.

    Raises:
    ValueError: If the input objects are not 3D or the save directory does not exist.
    """
    if type(obj) is torch.Tensor:
        obj = obj.detach().cpu().numpy()
    if type(ref) is torch.Tensor:
        ref = ref.detach().cpu().numpy()

    if not len(obj.shape) == 3 or not len(ref.shape) == 3:
        raise ValueError(f"Input object is not 3D, but {obj.shape} and {ref.shape}!")
    if save_dir is not None and not os.path.exists(save_dir):
        raise ValueError(f"Save directory ({save_dir}) does not exist!")

    obj_slice = obj[:, slice_num, :]
    obj_slice = obj_slice.astype(np.float32) / obj_slice.max()
    ref_slice = ref[:, slice_num, :]
    ref_slice = ref_slice.astype(np.float32) / ref_slice.max()
    plt.figure(figsize=(10, 5) if figsize is None else figsize)
    plt.subplot(1, 2, 1)
    plt.imshow(obj_slice, cmap=cmap)
    plt.title("Reconstruction")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(ref_slice, cmap=cmap)
    plt.title("Reference")
    plt.colorbar()
    plt.suptitle(title + f" (zx slice {slice_num})")
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, title + f",zx_slice{slice_num}.png"), dpi=300)
        plt.close()
    else:
        plt.show()
    return None


def display_curve(loss: np.ndarray, title: str = "", save_dir=None, metric="loss", figsize=(15, 5)):
    """
    Display a curve of the PSNR loss.

    Parameters:
    loss (list of float): The PSNR loss values.
    save_dir (str): The directory where the plot will be saved.
    title (str, optional): The title of the plot. Default is ''.

    Raises:
    ValueError: If the save directory does not exist.
    """
    if save_dir is not None and not os.path.exists(save_dir):
        raise ValueError(f"Save directory ({save_dir}) does not exist!")

    plt.figure(figsize=figsize)
    plt.plot(loss, label=metric)
    plt.xlabel("Iteration")
    plt.ylabel(metric)
    plt.grid()
    plt.title(title)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, title + ",curve.png"), dpi=300)
        plt.close()
    else:
        plt.show()
    return None


def display_hist(
    array: np.ndarray,
    title: str = "",
    save_dir=None,
    figsize=(10, 5),
    log=False,
):
    """
    Display a histogram of the given array.

    Parameters:
    array (numpy.ndarray): The array to be displayed.
    title (str, optional): The title of the plot. Default is ''.
    save_dir (str, optional): The directory where the plot will be saved. If None, the plot will be displayed. Default is None.

    Raises:
    ValueError: If the save directory does not exist.
    """
    if save_dir is not None and not os.path.exists(save_dir):
        raise ValueError(f"Save directory ({save_dir}) does not exist!")

    plt.figure(figsize=figsize)
    plt.hist(array.flatten(), bins=100, alpha=0.7, color="b", edgecolor="black", log=log)
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, title + ",hist.png"), dpi=300)
        plt.close()
    else:
        plt.show()
    return None


def display_images(
    objs: Union[list, tuple],
    title: str,
    cmap="gray",
    save_dir=None,
    axial_sclae_ratio=None,
    figsize=None,
):
    """
    Display a list of images together.

    Parameters:
    objs (list or tuple): A group of images to be displayed.
    title (str): The title of the plot.
    cmap (str, optional): The colormap to be used. Default is 'gray'.
    save_dir (str, optional): The directory where the plot will be saved. If None, the plot will be displayed. Default is None.

    Raises:
    ValueError: If the save directory does not exist.
    """
    if type(objs) is not list and type(objs) is not tuple:
        raise ValueError("Input object should be a list or tuple!")
    len_objs = len(objs)

    if len_objs == 0:
        raise ValueError("No input object!")

    for i in range(len_objs):
        if type(objs[i]) is torch.Tensor:
            objs[i] = objs[i].detach().cpu().numpy()
        if not len(objs[i].shape) == 2:
            raise ValueError(f"Input object {i} is not 2D!")
        objs[i] = objs[i] / objs[i].max()

    if save_dir is not None and not os.path.exists(save_dir):
        raise ValueError(f"Save directory ({save_dir}) does not exist!")

    plt.figure(figsize=(5 * len_objs, 5) if figsize is None else figsize)
    for i in range(1, len_objs + 1):
        plt.subplot(1, len_objs, i)
        plt.imshow(objs[i - 1], cmap=cmap)
        plt.title(f"Image {i}")
    plt.suptitle(title)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, title + ",multi-image.png"), dpi=300)
        plt.close()
    else:
        plt.show()
    return None
