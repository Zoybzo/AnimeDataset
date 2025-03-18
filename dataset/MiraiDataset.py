import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class MiraiDataset(Dataset):
    def __init__(self, root_dir, file_name, image_folder, transform=None, color="RGB"):
        super().__init__()
        self.root_dir = root_dir
        self.image_folder = os.path.join(self.root_dir, image_folder)
        self.file_path = os.path.join(self.root_dir, file_name)
        self.color = color
        self.transform = transform
        # read the file path
        self.image_path_list = list()
        with open(self.file_path, "r", encoding="utf-8") as file:
            self.image_path_list = file.readlines()
        self.image_path_list = [
            os.path.join(self.image_folder, line.strip())
            for line in self.image_path_list
        ]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # read from images
        sample = Image.open(self.image_path_list[idx]).convert(self.color)
        if self.transform:
            sample = self.transform(sample)
        return sample
