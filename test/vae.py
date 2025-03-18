from diffusers import CogView4Pipeline, AutoencoderKL
from torchvision import transforms
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import numpy as np


from loguru import logger

vae = AutoencoderKL.from_pretrained(
    "THUDM/CogView4-6B",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    subfolder="vae",
).to("cpu")
vae.enable_slicing()
vae.enable_tiling()

# encoder input: batch_size, num_channels, height, width
# encoder output: latent_dist

# load image
# img_path = "./assets/test/naruto.jpeg"
# result_img_path = "./assets/test/vae_naruto.png"
img_path = "./assets/test/744_1624.jpg"
result_img_path = "./assets/test/vae_744_1624.png"
raw_image = Image.open(img_path).convert("RGB")  # [0, 1]
logger.debug(f"raw_image: {raw_image.size}")
# 创建转换器对象
transform = transforms.ToTensor()
to_pil_image = transforms.ToPILImage()
# 转换图片为 tensor
image_tensor = transform(raw_image).unsqueeze(0).to(device="cpu", dtype=torch.bfloat16)
logger.debug(f"image_tensor.shape: {image_tensor.shape}")
sample = vae.forward(image_tensor)
logger.debug(f"sample: {sample}")
sample = sample["sample"].to(dtype=torch.float32).squeeze(0)
logger.debug(f"sample.shape: {sample.shape}")
rec_image = to_pil_image(sample)
rec_image.save(result_img_path)


img1 = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
img2 = np.array(Image.open(result_img_path).convert("L"), dtype=np.float32)
psnr_value = peak_signal_noise_ratio(img1, img2, data_range=255)
logger.info(f"PSNR: {psnr_value:.2f} dB")
