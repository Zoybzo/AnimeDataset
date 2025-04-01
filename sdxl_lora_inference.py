import os
import argparse
import toml
import re

from diffusers import DiffusionPipeline
import torch

from loguru import logger as loguru_logger

from utils.get_path import MODEL_HOME


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="stable-diffusion-xl-base-1.0")
    parser.add_argument("--lora_path", default=f"{MODEL_HOME}/kohya_ss")
    parser.add_argument("--lora_prefix")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", default="float16")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    device = args.device
    dtype = args.dtype
    dtype = torch.float16 if dtype == "float16" else torch.float32
    lora_path = args.lora_path
    lora_prefix = f"test-{args.lora_prefix}"
    step_range = range(10, 51, 10)
    # 正则表达式模式
    pattern = re.compile(rf"{lora_prefix}-.*-(\d+)\.safetensors$")

    # Loras
    # 筛选符合条件的文件路径
    selected_files = []
    for filename in os.listdir(lora_path):
        match = pattern.match(filename)
        if match:
            step_str = match.group(1)
            try:
                step = int(step_str)
                if step in step_range:
                    selected_files.append(os.path.join(lora_path, filename))
            except ValueError:
                continue
    # 打印结果
    print("Selected files:")
    for file_path in selected_files:
        print(file_path)

    # sdxl
    pipe_id = os.path.join(MODEL_HOME, model_path)
    # pipe.scheduler =
    pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=dtype euler_at_final=True,
                                             ).to(device)

    # 读取 TOML 文件
    with open("configs/prompts.toml", "r", encoding="utf-8") as file:
        data = toml.load(file)
    # 获取 prompt 列表
    prompts = data.get("prompts", [])
    # 打印 prompt 列表
    print("读取的 prompt 列表:")
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt}")

    # 生成图片
    for ckpt in selected_files:
        pipe.lora_lora_weights(
            ckpt,
        )
        for idx, prompt in enumerate(prompts):
            # 生成图片
            image = pipe(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

            # 保存图片
            image_path = os.path.join(output_dir, f"{ckpt}_prompt{idx}.png")
            image.save(image_path)
            print(f"Saved image: {image_path}")
