import os 
from io import BytesIO
from typing import List, Optional

import torch
from PIL import Image
from transformers import (

    AutoImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel

)

from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------
# Load Model
# -------------

model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(device)

try:

    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

except:

    base_id = "cnmoro/tiny-image-captioning"
    processor = AutoImageProcessor.from_pretrained(base_id)
    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)

if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token

def caption_image(
        
    image_path: str,
    max_length: int = 32,
    num_beams: int = 3,
    no_repeat_ngram_size: int = 2,

) -> str:
    
    """
    
    This is just to see if the model inference is working or not.

    """

    image = Image.open(image_path).convert("RGB")
    image.show()
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():

        output_ids = model.generate(

            pixel_values=pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram_size,

        )
    
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    
    print("Caption for", image_path, ":")

    return caption

if __name__ == "__main__":

    print("Device:", device)


    print(caption_image("example.jpg"))

    print(caption_image("example1.jpg"))