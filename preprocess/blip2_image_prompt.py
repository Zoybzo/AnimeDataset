from logging import root
import os
from re import L

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torch.utils.data import DataLoader
from loguru import logger as loguru_logger

from utils.get_path import DATASET_HOME, MODEL_HOME
from dataset.ImageDataset import ImageDataset


class Blip2ImagePrompt:
    def __init__(self, model_path, device="cpu", torch_type=torch.float16):
        self.model_path = model_path
        self.device = device
        self.torch_type = torch_type
        self.processor = Blip2Processor.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_path, local_files_only=True
        ).to("cuda:0")
        self.question = "Could you describe this image in detail?"

    def generate_prompt(self, text=None, image_path=None):
        if text is None:
            text = self.question
        if image_path is None or not os.path.exists(image_path):
            print(
                "You did not enter image path, the following will be a plain text conversation."
            )
            return
        else:
            image = Image.open(image_path).convert("RGB")
        history = []
        query = text
        question = "Could you describe this image in detail?"
        inputs = self.processor(image, question, return_tensors="pt").to(
            self.device, self.torch_type
        )
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


if __name__ == "__main__":
    # model
    # MODEL_PATH = os.path.join(MODEL_HOME, "cogvlm2-llama3-chat-19B")
    MODEL_PATH = os.path.join(MODEL_HOME, "")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_TYPE = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    # dataset
    root_dir = os.path.join(DATASET_HOME, "kafka_dataset")
    file_name = "title.txt"
    image_folder = "images"
    color = "RGB"

    # build
    image_generation = Cogvlm2ImagePrompt(
        model_path=MODEL_PATH, device=DEVICE, torch_type=TORCH_TYPE
    )
    image_dataset = ImageDataset(
        root_dir=root_dir,
        file_name=file_name,
        image_folder=image_folder,
        color=color,
    )
    dataloader = DataLoader(image_dataset, batch_size=1, shuffle=False)
    for idx, sample in enumerate(dataloader):
        loguru_logger.debug(sample)

        # data = data.to(device)
