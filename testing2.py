import requests
from PIL import Image

url = "E:\\karthik_drugs_recomendations\\mongo\\mongo\\new0.jpg"
image = Image.open(url).convert("RGB")
image
from transformers import TrOCRProcessor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten",use_fast=True)
# calling the processor is equivalent to calling the feature extractor
device = "cpu"
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
print(pixel_values.shape)
from transformers import VisionEncoderDecoderModel
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)