import numpy as np
from PIL import Image
import torch


def calculate_psnr(input_tensor, recon_tensor, max_val=255.0):
    """
    计算批量PSNR
    Args:
        input_tensor: 原始图像Tensor，形状[B, C, H, W]，范围[0, 1]
        recon_tensor: 重建图像Tensor，形状同input_tensor
        max_val: 像素最大值，默认255.0
    Returns:
        批量的平均PSNR
    """
    # 缩放至0-255范围
    input_scaled = input_tensor * 255.0
    recon_scaled = recon_tensor * 255.0

    # 计算MSE，dim指定为通道、高、宽维度，保持批次维度
    mse = torch.mean((input_scaled - recon_scaled) ** 2, dim=(1, 2, 3))

    # 避免除以零
    eps = 1e-10
    psnr = 10 * torch.log10(max_val**2 / (mse + eps))

    return torch.mean(psnr)  # 返回批次平均PSNR
