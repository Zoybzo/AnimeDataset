import os
import re


def replace_special_characters(path):
    """
    递归替换文件夹和文件名称中的特殊字符（保留点号 .）
    """
    # 定义需要替换的特殊字符模式（排除点号 .）
    special_chars_pattern = r"[\"'()（）\[\]{}<>~`!@#$%^&*+=|\\/:;,\s?\"']"

    # 遍历指定路径下的所有文件和文件夹
    for root, dirs, files in os.walk(path, topdown=False):
        # 替换文件夹名称
        for dir_name in dirs:
            new_dir_name = re.sub(special_chars_pattern, "_", dir_name)
            if dir_name != new_dir_name:
                old_dir_path = os.path.join(root, dir_name)
                new_dir_path = os.path.join(root, new_dir_name)
                os.rename(old_dir_path, new_dir_path)
                print(f"Renamed folder: {old_dir_path} -> {new_dir_path}")

        # 替换文件名称
        for file_name in files:
            # 分离文件名和扩展名
            name_part, extension_part = os.path.splitext(file_name)
            # 只替换文件名部分的特殊字符
            new_name_part = re.sub(special_chars_pattern, "_", name_part)
            new_file_name = new_name_part + extension_part
            new_file_name = new_file_name.replace("图像", "img")

            if file_name != new_file_name:
                old_file_path = os.path.join(root, file_name)
                new_file_path = os.path.join(root, new_file_name)
                os.rename(old_file_path, new_file_path)
                print(f"Renamed file: {old_file_path} -> {new_file_path}")


if __name__ == "__main__":
    folder_path = input("请输入目标文件夹路径: ")
    if os.path.exists(folder_path):
        replace_special_characters(folder_path)
        print("文件夹和文件名称替换完成！")
    else:
        print("输入的路径不存在，请检查！")
