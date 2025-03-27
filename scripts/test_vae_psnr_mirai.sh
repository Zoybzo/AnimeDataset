#!/bin/bash

python vae_psnr.py \
  --model_path "$MODEL_HOME/stable-diffusion-xl-base-1.0" \
  --dataset "mirai" \
  --device "cpu" \
  --batch_size 1024

