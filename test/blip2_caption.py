import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from utils.get_path import MHOME

# model_path = f"{MHOME}/Models/blip2-opt-2.7b"
model_path = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(model_path, local_files_only=True)
model = Blip2ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

image_path = "./assets/test/naruto.jpeg"
# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(img_path).convert('RGB')

question = "describe this image in detail"
inputs = processor(raw_image, question, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True).strip())

