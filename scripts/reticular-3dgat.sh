CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=3 \
python 3DGAT.py \
--root_dir './data/flfm/reticular1' \
--exp_name './GS/reticular1' \
--init_from 'measur_30000' \
--fourier_weight 2e-4 \
--densify_grad_threshold 1e-8 \
--density_min_threshold 1e-2 \
--scale_max_bound 20 \
--init_threshold 0.1
