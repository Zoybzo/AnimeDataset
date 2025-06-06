from torchvision import transforms
from torch.utils.data import DataLoader
import torch

from dataset.MiraiDataset import MiraiDataset
from utils.get_path import DATASET_HOME

root_dir = f"{DATASET_HOME}/miraimind_12702"
file_name = "title.txt"
image_folder = "images"
color = "L"
custom_transform = transforms.Compose(
    [
        transforms.RandomCrop(size=(224, 224)),
        transforms.ToTensor(),
    ]
)
dataset = MiraiDataset(
    root_dir=root_dir,
    file_name=file_name,
    image_folder=image_folder,
    transform=custom_transform,
    color=color,
)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
for idx, features in enumerate(dataloader):
    print(features)
    print(features.size())
