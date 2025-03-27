#!/bin/bash

python vae_psnr.py \
    --model_path "{$MODEL_HOME}/stable-diffusion-xl-base-1.0" \
    --dataset "ImageNet" \
    --device "cuda:2"

