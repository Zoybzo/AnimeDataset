import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

model_path = "/Users/user/Downloads/ckpt/CartoonCaptionGen"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
img_url = "/Users/user/Downloads/img/漩涡鸣人.jpeg"
raw_image = Image.open(img_url).convert('RGB')

# conditional image captioning
# text = "a photography of"
# inputs = processor(raw_image, text, return_tensors="pt")
#
# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))
# >>> a photography of a woman and her dog

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt")
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
