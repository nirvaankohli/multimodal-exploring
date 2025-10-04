import os
import sys
from typing import List

from PIL import Image
import torch
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "./")

def get_image_file(path, file_type: str = "single") -> List[str]:

    supported_types = (".png", ".jpg", ".jpeg", ".webp")

    if file_type == "single":

        if (not os.path.isfile(path)) or (not path.lower().endswith(supported_types)):
            
            print(f"Error: The file {path} does not exist or is not a supported image type.")
            
            sys.exit(1)
        
        return [path]

    if isinstance(path, list):
        
        image_files: List[str] = []
        
        for p in path:
            
            if os.path.isfile(p) and p.lower().endswith(supported_types):
                
                image_files.append(p)
            
            else:
                print(f"Warning: The file {p} does not exist or is not a supported image type. Skipping.")

        if not image_files:
            print("Error: No valid image files found.")
            sys.exit(1)
        return image_files

    if file_type == "dir":
        if not os.path.isdir(path):
            print(f"Error: The directory {path} does not exist or is not a directory.")
            sys.exit(1)

        image_files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(supported_types)
        ]

        if not image_files:
            print(f"Error: No supported image files found in directory {path}.")
            sys.exit(1)

        image_files.sort()
        return image_files

    print("Error: Invalid file_type. Use 'single', 'dir', or provide a list of file paths.")
    sys.exit(1)


def load_Image(image_path: str, max_side: int = 512) -> Image.Image:


    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error: Failed to open image '{image_path}': {e}")
        sys.exit(1)

    w, h = image.size
    if max_side and max(w, h) > max_side:
        if w >= h:
            new_w = max_side
            ratio = new_w / w
            new_h = int(h * ratio)
        else:
            new_h = max_side
            ratio = new_h / h
            new_w = int(w * ratio)
        image = image.resize((new_w, new_h), resample=Image.LANCZOS)

    return image


def main():
    img_paths = "exampleimgs"

    if isinstance(img_paths, str) and not img_paths.endswith((".png", ".jpg", ".jpeg", ".webp")):
        
        img_type = "dir"
        img_paths = os.path.join(BASE_URL, img_paths)

    if isinstance(img_paths, str):

        img_type = "dir" if os.path.isdir(img_paths) else "single"

        if img_type == "dir":

            print(f"Loading images from directory: {img_paths}")

            img_list = [
                os.path.join(img_paths, f)
                for f in os.listdir(img_paths)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            ]

            print("Images found:", img_list)

    else:
        img_type = "list"

    args = dict(
        img_paths=img_paths,
        img_type=img_type,
        model = "Salesforce/blip-image-captioning-base",
        device="cpu" if not torch.cuda.is_available() else "cuda",
        max_new_tokens=24,
        num_beams=4,
        max_side=512,
    )

    captioner = pipeline(
        task="image-to-text",
        model=args["model"],
        tokenizer=args["model"],
        device=-1 if args["device"] == "cpu" else 0,
    )

    img_files = get_image_file(args["img_paths"], args["img_type"])

    total = len(img_files)
    for i, img_path in enumerate(img_files, start=1):

        image = load_Image(img_path, args["max_side"])
        
        out = captioner(
            image,
            generate_kwargs={
                "max_new_tokens": args["max_new_tokens"],
                "num_beams": args["num_beams"],
            },
        )

        caption = out[0].get("generated_text", "").strip() if out else ""
        print(f"[{i}/{total}] {img_path} --> {caption}")



if __name__ == "__main__":
    main()
