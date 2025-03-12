from lmdeploy import pipeline, ChatTemplateConfig
from lmdeploy.vl import load_image

from utils.get_path import MHOME

model_path = f"{MHOME}/Models/llava-llama-3-8b"
pipe = pipeline(model_path,
                chat_template_config=ChatTemplateConfig(model_name='llama3'))
image_path = "./assets/test/naruto.jpeg"
image = load_image(image_path)
response = pipe(('describe this image', image))
print(response)

