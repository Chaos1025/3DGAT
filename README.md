<h1 align="center">3D Gaussian Adaptive Reconstruction for Fourier Light-Field Microscopy</h1>

<div align="center">

Chenyu Xu<sup>†</sup>, Zhouyu Jin<sup>†</sup>, Chengkang Shen, Hao Zhu, Zhan Ma, Bo Xiong<sup>\*</sup>, You Zhou<sup>\*</sup>, Xun Cao, Ning Gu

<sup>†</sup>Indicates Equal Contribution
<br>
<sup>*</sup>Corresponding Author

<p>
  <a href="https://www.researching.cn/ArticlePdf/m00132/2025/2/5/055001.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge&logo=Adobe%20Acrobat%20Reader&logoColor=white" alt="Paper PDF">
  </a>
  <a href="https://github.com/Chaos1025/3DGAT">
    <img src="https://img.shields.io/badge/Code-GitHub-balck?style=for-the-badge&logo=github&logoColor=white" alt="Code Repository">
  </a>
</p>

</div>

Code repository of [3D Gaussian Adaptive Reconstruction for Fourier Light-Field Microscopy](https://arxiv.org/abs/2505.12875).

# System Requirements
- Ubuntu 22.4
- cuda 12.7
## Python Packages Requirements

```bash
conda env create -f environment.yml
conda activate 3dgat
pip install -e submodules/us_gaussian_voxelization
pip install -e submodules/simple-knn
```

# Data Preparation
We provide the reticular data displayed in our main text (Fig. 2d) and supplementary material (Fig. S5) as examples. They are located in `data/flfm/reticular1` (Fig. 2d) and `data/flfm/reticular2` (Fig. S5) so you can test them easily.

For your own data, it should be located in `data/<your_data_name>` and the directory should contain:
- `data.tif`: ground truth volume, only exists in simulation experiments, with shape of [D, H, W]
- `H.mat`: the entire system PSF, with shape of [D, M, N], storded with sparse format
- `psf.mat`: 4D PSF cropped from the entire system PSF, according to the MLA center indexs in image coordinates, with shape of [V, D, Y, X]
- `LFimage.tif`: raw FLFM image, with shape of [M, N]
- `MLcenter_idxes.mat`: MLA center indexs in image coordinates
- `params.yaml`: data-related arguments for 3DGAT, typically the volume size to be reconstructed, and the image size of raw FLFM image
- `deconvolve.yaml`: arguments for wiener filtering and RL deconvolution


# Quick Start
There are 2 steps for a quick start to perform our 3DGAT pipeline on the example data.

## 1. Wiener Filtering
Wiener filtering is firstly performed to get an initial estimation.
```bash
python onestep_wiener.py --root_dir data/flfm/reticular1
```
The Wiener filtering results will be stored in the specified `root_dir`

## 2. 3D Gaussian Adaptive Reconstruction
Then the Wiener filtering result is used as the initialization for 3D Gaussian Adaptive Reconstruction. As an example, we provide a reconstruction script for the example data (Fig. 2d in our main text, `scripts/reticular-3dgat.sh`). Some necessary arguments are also given in this scripts. More detailed description of these args can be found in  [`opt.py`](opt.py). 
```bash
bash scripts/reticular-3dgat.sh
```
Reconstruction results will be stored in `results/GS/reticular`

## Comparison
We also provide the other two methods of RL deconvolution and gradient-based optimization baseline for comparison, as we discussed in our paper.

For RL deconvolution,
```bash
python flfm_RLdeconv.py --root_dir data/flfm/reticular1
```

For gradient-based optimization baseline, the reconstruction script for the example data (Fig. 2d in our main text) is also provided in `scripts/reticular-baseline.sh`.
```bash
bash scripts/reticular-baseline.sh
```

# Citation
If you find our paper helpful, please cite us
```bibtex
@article{xu20253d,
  title={3D Gaussian adaptive reconstruction for Fourier light-field microscopy},
  author={Xu, Chenyu and Jin, Zhouyu and Shen, Chengkang and Zhu, Hao and Ma, Zhan and Xiong, Bo and Zhou, You and Cao, Xun and Gua, Ning},
  journal={Advanced Imaging},
  volume={55001},
  pages={1},
  year={2025}
}
```