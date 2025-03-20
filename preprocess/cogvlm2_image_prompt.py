from logging import root
import os
from re import L

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# from transformers_modules.cogvlm2_llama3_chat_19B.tokenization_cogvlm import (
# CogVLMTokenizer,
# )
from torch.utils.data import DataLoader
from loguru import logger as loguru_logger

from utils.get_path import DATASET_HOME, MODEL_HOME
from dataset.ImageDataset import ImageDataset


class Cogvlm2ImagePrompt:
    def __init__(self, model_path, device="cpu", torch_type=torch.float16):
        self.model_path = model_path
        self.device = device
        self.torch_type = torch_type
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_type,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )
        self.text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

    def generate_prompt(self, text, image_path=None, text_only_template=None):
        if text_only_template is None:
            text_only_template = self.text_only_template
        text_only_first_query = False
        if image_path is None or not os.path.exists(image_path):
            print(
                "You did not enter image path, the following will be a plain text conversation."
            )
            image = None
            text_only_first_query = True
        else:
            image = Image.open(image_path).convert("RGB")
        history = []
        query = text
        if image is None:
            if text_only_first_query:
                query = text_only_template.format(query)
                text_only_first_query = False
            else:
                old_prompt = ""
                for _, (old_query, response) in enumerate(history):
                    old_prompt += old_query + " " + response + "\n"
                query = old_prompt + "USER: {} ASSISTANT:".format(query)
        if image is None:
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer, query=query, history=history, template_version="chat"
            )
        else:
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer,
                query=query,
                history=history,
                images=[image],
                template_version="chat",
            )
        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self.device),
            "token_type_ids": input_by_model["token_type_ids"]
            .unsqueeze(0)
            .to(self.device),
            "attention_mask": input_by_model["attention_mask"]
            .unsqueeze(0)
            .to(self.device),
            "images": (
                [[input_by_model["images"][0].to(DEVICE).to(TORCH_TYPE)]]
                if image is not None
                else None
            ),
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0]
            print("\nCogVLM2:", response)
        history.append((query, response))


if __name__ == "__main__":
    # model
    MODEL_PATH = os.path.join(MODEL_HOME, "cogvlm2-llama3-chat-19B")
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
