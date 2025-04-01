import os
from PIL import Image


def image_flip(image, horizontal=True):
    if horizontal:
        flip = image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        flip = image.transpose(Image.FLIP_TOP_BOTTOM)
    return flip


def rotete(image, angle=15):
    rotated_image = image.rotate(angle, expand=True)
    return rotated_image


if __name__ == "__main__":
    # 提示用户输入图片路径
    image_path = input("Enter the path to the image: ").strip()
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print("Error: The specified file does not exist.")
        exit(0)

    # 打开图片
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error: Unable to open the image. {e}")
        exit(0)

    angle = float(input("Enter rotation angle (in degrees): ").strip())
    # image = image.rotate(angle, expand=True)
    image = image_flip(image)

    # 获取图片的文件名和扩展名
    base_path, file_extension = os.path.splitext(image_path)

    # 保存处理后的图片
    output_path = f"{base_path}_processed{file_extension}"
    image.save(output_path)
    print(f"Processed image saved as: {output_path}")
