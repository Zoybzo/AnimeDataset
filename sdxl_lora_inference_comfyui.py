import os
import argparse
import toml
import re
import random
import sys
import torch
from typing import Sequence, Mapping, Any, Union
from loguru import logger as loguru_logger

COMFYUI_PATH = os.environ.get("COMFYUI_PATH")

MODEL_HOME = os.environ.get("MODEL_HOME")
DATASET_HOME = os.environ.get("DATASET_HOME")

HOME = os.environ.get("HOME")


def save_images(output_dir, image, filename):
    loguru_logger.info("Saving images...")
    image_path = os.path.join(output_dir, filename)
    image.save(image_path)


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
        from ComfyUI.main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from ComfyUI.utils.extra_config import load_extra_path_config

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


def main(lora_name_list, text_list, save_path):
    import_custom_nodes()
    loguru_logger.info("Inference...")
    with torch.inference_mode():
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_1 = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_base_1.0.safetensors"
        )
        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        emptylatentimage_7 = emptylatentimage.generate(
            width=1024, height=1024, batch_size=1
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
                    save_images(
                        save_path,
                        images,
                        f"{lora_name.split('.')[-2].split('-')[-1]}_P{text_idx}_{q}.png",
                    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="stable-diffusion-xl-base-1.0")
    parser.add_argument("--lora_path", default=f"{MODEL_HOME}/kohya_ss")
    parser.add_argument("--lora_prefix", default="test-0331-38")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--step", default=10)
    parser.add_argument("--prompt_file", default="prompts.toml")
    parser.add_argument("--save_path", default=f"{MODEL_HOME}/kohya_samples")
    return parser.parse_args()


def get_lora_list(step_range):
    pattern = re.compile(rf"{lora_prefix}-.*-(\d+)\.safetensors$")

    # Loras
    # 筛选符合条件的文件路径
    selected_files = []
    for filename in os.listdir(lora_path):
        match = pattern.match(filename)
        if match:
            step_str = match.group(1)
            try:
                step = int(step_str)
                if step in step_range:
                    selected_files.append(os.path.join(lora_path, filename))
            except ValueError:
                continue
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

    lora_prefix = f"test-{args.lora_prefix}"
    step_range = range(10, 51, step)

    lora_name_list = get_lora_list(step_range)
    text_list = get_prompt(prompt_file)

    main(lora_name_list, text_list, save_path)
