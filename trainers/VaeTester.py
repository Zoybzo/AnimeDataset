from diffusers import AutoencoderKL
import torch
import numpy as np

from loguru import logger

from trainers.Trainer import Trainer
from utils import metrics


class VaeTester(Trainer):
    def __init__(self, model_path, subfolder="vae", device="cpu"):
        super().__init__()
        self.model_path = model_path
        self.subfolder = subfolder
        self.device = device
        self.prepare_models(self.model_path, self.subfolder, self.device)
        assert self.vae is not None

    def prepare_models(self, model_path=None, subfolder=None, device="cpu"):
        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            subfolder=self.subfolder,
        ).to(self.device)
        self.vae.enable_slicing()
        self.vae.enable_tiling()

    @torch.no_grad()
    def validate(self, dataloader):
        self.vae.eval()
        tot_psnr = 0.0
        data_lens = len(dataloader)
        for idx, features in enumerate(dataloader):
            features = features.to(device=self.device, dtype=torch.bfloat16)
            sample = self.vae.forward(features)["sample"]
            psnr = metrics.calculate_psnr(features, sample).item()
            logger.info(f"{idx} psnr: {psnr}")
            tot_psnr += psnr
        avg_psnr = tot_psnr / data_lens
        return avg_psnr
