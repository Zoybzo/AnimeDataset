from lmdeploy import pipeline, ChatTemplateConfig
from lmdeploy.vl import load_image

from utils.get_path import MODEL_HOME

model_path = f"{MODEL_HOME}/Models/llava-llama-3-8b"
pipe = pipeline(
    model_path, chat_template_config=ChatTemplateConfig(model_name="llama3")
)
image_path = "./assets/test/naruto.jpeg"
image = load_image(image_path)
response = pipe(
    (
        "The characters in this image are both Uzumaki Naruto from anime Naruto, now describe this image",
        image,
    )
)
print(response)
