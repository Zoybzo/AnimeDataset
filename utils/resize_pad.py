from torchvision.transforms import functional as F


class ResizeAndPad:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        # 调整大小，保持宽高比
        img = F.resize(
            img, self.target_size, interpolation=F.InterpolationMode.BILINEAR
        )
        # 计算需要填充的宽度和高度
        w, h = img.size
        new_w, new_h = self.target_size
        padding_left = (new_w - w) // 2
        padding_top = (new_h - h) // 2
        padding_right = new_w - w - padding_left
        padding_bottom = new_h - h - padding_top
        # 填充图像
        img = F.pad(
            img, (padding_left, padding_top, padding_right, padding_bottom), fill=0
        )
        return img
