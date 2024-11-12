#! /usr/bin/env bash

export GIT_PYTHON_REFRESH=quiet

CUDA_VISIBLE_DEVICES=1 python image2video_ddim_inv_bidirection.py \
  --model_id_or_path /cfs/zhlin/projects/aigc_engine/pretrained_models/stable-video-diffusion-img2vid-xt \
  --image_start /cfs/zhlin/projects/aigc_engine/samples/seal0.jpg \
  --image_end /cfs/zhlin/projects/aigc_engine/samples/seal1.jpg \
  --frames /cfs/zhlin/projects/aigc_engine/samples/seal \
  --height 384 \
  --width 640 \
  --nframes 40 \
  --fps 10 \
  --steps 20 \
  --strength 1 \
  --save_path /cfs/zhlin/projects/aigc_engine/demos/results/seal-m2-w-last-sa-edit
