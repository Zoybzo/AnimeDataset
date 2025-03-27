import os
import argparse

from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchvision.datasets import ImageNet

from loguru import logger

from utils.get_path import DATASET_HOME
from utils.resize_pad import ResizeAndPad
from dataset.ImageDataset import ImageDataset
from trainers.VaeTester import VaeTester


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--subfolder", default="vae")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dataset", default="mirai")
    parser.add_argument("--batch_size", default=1024, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    subfolder = args.subfolder
    device = args.device
    dataset = args.dataset
    batch_size = args.batch_size

    custom_transform = transforms.Compose(
        [
            ResizeAndPad((224, 224)),  # 调整大小并填充
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    custom_anti_transform = transforms.ToPILImage()
    if dataset == "ImageNet":
        root_dir = f"{DATASET_HOME}/ImageNet"
        dataset = ImageNet(root=root_dir, transform=custom_transform)
    elif dataset == "mirai":
        root_dir = f"{DATASET_HOME}/miraimind_12702"
        file_name = "title.txt"
        image_folder = "images"
        color = "RGB"
        dataset = ImageDataset(
            root_dir=root_dir,
            file_name=file_name,
            image_folder=image_folder,
            transform=custom_transform,
            color=color,
        )
    dtype = torch.float32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vae_tester = VaeTester(
        model_path=model_path, subfolder=subfolder, device=device, dtype=dtype
    )
    avg_psnr = vae_tester.validate(dataloader)
    logger.info(f"Avg psnr: {avg_psnr}")
