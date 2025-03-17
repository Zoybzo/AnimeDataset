from diffusers import CogView4Pipeline, AutoencoderKL
from torchvision import transforms
import torch
from PIL import Image

vae = AutoencoderKL.from_pretrained(
    "THUDM/CogView4-6B",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    subfolder="vae",
)

# encoder input: batch_size, num_channels, height, width
# encoder output: latent_dist

# load image
img_path = "./assets/test/naruto.jpeg"
raw_image = Image.open(img_path).convert("RGB")
# 创建转换器对象
transform = transforms.ToTensor()
to_pil_image = transforms.ToPILImage()
# 转换图片为 tensor
image_tensor = (
    transform(raw_image).unsqueeze(0).to(device="cuda:0", type=torch.bfloat16)
)
latent_dist = vae.encode(image_tensor)
sample = vae.decode(latent_dist)
rec_image = to_pil_image(sample)
rec_image.save("./assets/test/vae_naruto.png")
