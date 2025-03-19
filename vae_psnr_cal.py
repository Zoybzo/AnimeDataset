from torchvision import transforms
from torch.utils.data import DataLoader
import torch

from loguru import logger

from dataset.MiraiDataset import MiraiDataset
from utils.get_path import DATASET_HOME
from trainers.VaeTester import VaeTester

if __name__ == "__main__":
    # dataset
    root_dir = f"{DATASET_HOME}/miraimind_12702"
    file_name = "title.txt"
    image_folder = "images"
    color = "RGB"
    # model
    # model_path = "THUDM/CogView4-6B"
    model_path = "stabilityai/sdxl-turbo"
    subfolder = "vae"
    device = "cuda:0"

    custom_transform = transforms.Compose(
        [
            transforms.RandomCrop(size=(224, 224)),
            transforms.ToTensor(),
        ]
    )
    custom_anti_transform = transforms.ToPILImage()
    dataset = MiraiDataset(
        root_dir=root_dir,
        file_name=file_name,
        image_folder=image_folder,
        transform=custom_transform,
        color=color,
    )
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    vae_tester = VaeTester(model_path=model_path, subfolder=subfolder, device=device)
    avg_psnr = vae_tester.validate(dataloader)
    logger.info(f"Avg psnr: {avg_psnr}")
