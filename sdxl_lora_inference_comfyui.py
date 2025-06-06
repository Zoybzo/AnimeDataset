import os
import argparse
from datetime import datetime
import toml
import re
import random
import sys
import torch
import numpy as np
from PIL import Image
from typing import Sequence, Mapping, Any, Union
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from loguru import logger as loguru_logger

COMFYUI_PATH = os.environ.get("COMFYUI_PATH")

MODEL_HOME = os.environ.get("MODEL_HOME")
DATASET_HOME = os.environ.get("DATASET_HOME")

HOME = os.environ.get("HOME")


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    # comfyui_path = find_path("ComfyUI")
    comfyui_path = COMFYUI_PATH
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
# add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS
from folder_paths import set_output_directory


def main(lora_name_list, text_list, save_path, batch_size, st, ed, step, lora_prefix):
    import_custom_nodes()
    loguru_logger.info("Inference...")
    lora_mp = {
        lora_name: lora_name.split("-")[2]
        + "-"
        + lora_name.split("-")[-1].split(".")[0]
        for lora_name in lora_name_list
    }
    keys1 = list(lora_mp.values())
    # [
    #     lora_name.split("-")[2] + "-" + lora_name.split("-")[-1].split(".")[0]
    #     for lora_name in lora_name_list
    # ]
    keys2 = [idx for idx in range(0, len(text_list))]
    image_dict = {key1: {key2: None for key2 in keys2} for key1 in keys1}
    with torch.inference_mode():
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_1 = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_base_1.0.safetensors"
        )
        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        emptylatentimage_7 = emptylatentimage.generate(
            width=1024, height=1024, batch_size=batch_size
        )

        ksampleradvanced = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()
        # need lora
        for lora_name in lora_name_list:
            loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
            loraloader_9 = loraloader.load_lora(
                # lora_name="test-0329--sdxl_base_1.0-p9-000050.safetensors",
                lora_name=lora_name,
                strength_model=1,
                strength_clip=1,
                model=get_value_at_index(checkpointloadersimple_1, 0),
                clip=get_value_at_index(checkpointloadersimple_1, 1),
            )

            for text_idx, text in enumerate(text_list):
                cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
                cliptextencode_2 = cliptextencode.encode(
                    text=text,
                    clip=get_value_at_index(loraloader_9, 1),
                )

                cliptextencode_3 = cliptextencode.encode(
                    text="out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature,",
                    clip=get_value_at_index(loraloader_9, 1),
                )

                for q in range(1):
                    ksampleradvanced_4 = ksampleradvanced.sample(
                        add_noise="enable",
                        noise_seed=random.randint(1, 2**64),
                        steps=30,
                        cfg=8,
                        sampler_name="euler_ancestral",
                        scheduler="normal",
                        start_at_step=0,
                        end_at_step=10000,
                        return_with_leftover_noise="disable",
                        model=get_value_at_index(loraloader_9, 0),
                        positive=get_value_at_index(cliptextencode_2, 0),
                        negative=get_value_at_index(cliptextencode_3, 0),
                        latent_image=get_value_at_index(emptylatentimage_7, 0),
                    )

                    vaedecode_5 = vaedecode.decode(
                        samples=get_value_at_index(ksampleradvanced_4, 0),
                        vae=get_value_at_index(checkpointloadersimple_1, 2),
                    )

                    # saveimage_6 = saveimage.save_images(
                    #     filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_5, 0)
                    # )
                    images = get_value_at_index(vaedecode_5, 0)
                    batch_images = save_images(
                        save_path,
                        images,
                        f"{''.join(lora_name)}_P{text_idx}",
                    )
                    image_dict[lora_mp[lora_name]][text_idx] = batch_images[0][0]
    # prefix = "".join(lora_name_list[0].split("-")[:2])
    figure_name = os.path.join(
        save_path,
        f"{lora_prefix}_{get_datetime()}_{st}_{ed}_{step}.png",
    )
    generate_plt(image_dict, keys1, keys2, figure_name)


def generate_plt(image_dict, keys1, keys2, figure_name):
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(len(keys1), len(keys2), wspace=0.05, hspace=0.05)
    # 遍历嵌套字典，将图片添加到大图中
    for i, key1 in enumerate(keys1):
        for j, key2 in enumerate(keys2):
            ax = plt.subplot(gs[i, j])
            image = image_dict[key1][key2]
            ax.imshow(image)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis="both", length=0)

            # 添加标题（仅在第一行显示 key2）
            if i == 0:
                ax.set_title(f"{key2}", fontsize=10)

            # 添加纵轴标签（仅在第一列显示 key1）
            if j == 0:
                ax.set_ylabel(f"{key1}", fontsize=10)

    # 保存大图
    plt.tight_layout()
    plt.savefig(figure_name, dpi=300, bbox_inches="tight")
    loguru_logger.info(f"Saved large image: {figure_name}")
    pass


def save_images(output_dir, images, filename):
    loguru_logger.info("Saving images...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    batch_images = list()
    for batch_number, image in enumerate(images):
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        file = f"{filename}_{batch_number}_{get_datetime()}.png"
        img_path = os.path.join(output_dir, file)
        img.save(img_path)
        batch_images.append([img, img_path])
    return batch_images


def get_datetime():
    return datetime.now().strftime("%m%d%I%M%S")


def filter_files(
    file_list, step_range=None, date=None, id=None, model=None, dataset=None, epoch=None
):
    """
    从文件列表中筛选出符合条件的文件名。

    参数:
        file_list (list): 文件名列表。
        date (str, optional): 日期条件，格式为 "MMDD"。
        id (str, optional): ID 条件。
        model (str, optional): 模型名称条件。
        dataset (str, optional): 数据集名称条件。
        epoch (str, optional): epoch 条件。

    返回:
        list: 符合条件的文件名列表。
    """
    filtered_files = []
    pattern = r"test-(\d{4})-(\d+)-([\w_.]+)-([\w_.]+)-(\d+).safetensors"

    for file_name in file_list:
        match = re.match(pattern, file_name)
        if match:
            file_date, file_id, file_model, file_dataset, file_epoch = match.groups()
            if (
                (date is None or file_date == date)
                and (id is None or file_id == id)
                and (model is None or file_model == model)
                and (dataset is None or file_dataset == dataset)
                and (epoch is None or file_epoch == epoch)
            ):
                filtered_files.append(file_name)
    if id != None and epoch is None and step_range != None:
        res = list()
        for file_name in filtered_files:
            step = int(file_name.split("-")[-1].split(".")[0])
            if step in step_range:
                res.append(file_name)
        filtered_files = res
    return filtered_files


def get_lora_list(
    lora_path, step_range=None, date=None, id=None, model=None, dataset=None, epoch=None
):
    loguru_logger.info(f"Lora path: {lora_path}")
    loguru_logger.info(f"Lora prefix: {lora_prefix}")

    # pattern = re.compile(rf"{lora_prefix}-.*-(\d+)\.safetensors$")

    # Loras
    # 筛选符合条件的文件路径
    selected_files = []
    filename_list = os.listdir(lora_path)
    selected_files = filter_files(
        filename_list,
        step_range=step_range,
        date=date,
        id=id,
        model=model,
        dataset=dataset,
        epoch=epoch,
    )
    # for filename in os.listdir(lora_path):
    #     match = pattern.match(filename)
    #     if match:
    #         step_str = match.group(1)
    #         try:
    #             step = int(step_str)
    #             if step in step_range:
    #                 # selected_files.append(os.path.join(lora_path, filename))
    #                 selected_files.append(filename)
    #         except ValueError:
    #             continue
    # 打印结果
    loguru_logger.info("Selected files:")
    for file_path in selected_files:
        print(file_path)

    return selected_files


def get_prompt(prompt_file):
    # 读取 TOML 文件
    with open(f"./configs/{prompt_file}", "r", encoding="utf-8") as file:
        data = toml.load(file)
    # 获取 prompt 列表
    prompts = data.get("prompts", [])
    # 打印 prompt 列表
    loguru_logger.info("读取的 prompt 列表:")
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt}")
    return prompts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="stable-diffusion-xl-base-1.0")
    parser.add_argument("--lora_path", default=f"{MODEL_HOME}/kohya_ss")
    parser.add_argument("--lora_prefix", default="test-0331-38")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--save_path", default=f"{MODEL_HOME}/kohya_samples")
    parser.add_argument("--batch_size", default=1)

    parser.add_argument("--char", default="kafka")

    # lora_path, step_range=None, date=None, id=None, model=None, dataset=None, epoch=None
    parser.add_argument("--step", default=10)
    parser.add_argument("--start", default=0)
    parser.add_argument("--end", default=51)
    parser.add_argument("--date", default=None, type=str)
    parser.add_argument("--model", default="sdxl_base_1.0")
    parser.add_argument("--id", default=None, type=str)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--epoch", default=None, type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    device = args.device
    dtype = args.dtype
    dtype = torch.float16 if dtype == "float16" else torch.float32
    lora_path = args.lora_path
    step = args.step
    prompt_file = args.prompt_file
    save_path = args.save_path
    batch_size = args.batch_size
    st, ed = args.start, args.end
    date = args.date
    model = args.model
    id = args.id
    dataset = args.dataset
    epoch = args.epoch
    char = args.char

    # lora_prefix = f"{args.lora_prefix}"
    lora_prefix = ""
    if date is not None:
        lora_prefix += "-" + str(date)
    if id is not None:
        lora_prefix += "-" + str(id)
    if dataset is not None:
        lora_prefix += "-" + str(dataset)
    if epoch is not None:
        lora_prefix += "-" + str(epoch)
    lora_prefix += "-"
    step_range = range(st, ed, step)

    lora_name_list = get_lora_list(
        lora_path,
        step_range,
        date=date,
        model=model,
        id=id,
        dataset=dataset,
        epoch=epoch,
    )
    lora_name_list.sort()

    if prompt_file is None:
        prompt_file = f"{char}_prompts.toml"
    text_list = get_prompt(prompt_file)

    main(lora_name_list, text_list, save_path, batch_size, st, ed, step, lora_prefix)
