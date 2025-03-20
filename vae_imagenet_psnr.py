from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchvision.datasets import ImageNet

from loguru import logger

from utils.get_path import DATASET_HOME
from utils.resize_pad import ResizeAndPad
from trainers.VaeTester import VaeTester

if __name__ == "__main__":
    # dataset
    root_dir = f"{DATASET_HOME}/ImageNet"
    # file_name = "title.txt"
    # image_folder = "images"
    # color = "RGB"
    # model
    model_path = "THUDM/CogView4-6B"
    subfolder = "vae"
    device = "cuda:1"

    custom_transform = transforms.Compose(
        [
            ResizeAndPad((224, 224)),  # 调整大小并填充
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    custom_anti_transform = transforms.ToPILImage()
    dataset = ImageNet(root=root_dir, transform=custom_transform)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
    vae_tester = VaeTester(model_path=model_path, subfolder=subfolder, device=device)
    avg_psnr = vae_tester.validate(dataloader)
    logger.info(f"Avg psnr: {avg_psnr}")
