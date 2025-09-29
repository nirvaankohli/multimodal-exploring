import os, json, random, time, math
from pathlib import Path
from collections import defaultdict

import torch

intensity = 5  # Scale of 1-10 for CPU usage
torch.set_num_threads(max(1, int(os.cpu_count() // (1 / (intensity / 10))))) # Sets the number of threads to the cpu_count divided by the reciprocal of the intensity(scaled 1 - 10) divided by 10

from PIL import Image
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd

from transformers import (

    AutoTokenizer,
    AutoProcessor,
    BlipForConditionalGeneration,
    TrainingArguments,
    Trainer,
    default_data_collator,

)
import csv
from datetime import datetime


# Paths -> Data, Images, Captions.txt; Replace your paths here

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "Images"
CAPTIONS_FILE = DATA_DIR / "captions.txt"

# Make sure nothing is missing, everything exists, etc.

assert IMAGES_DIR.exists(), f"Images directory {IMAGES_DIR} does not exist. Run the cells above."
assert CAPTIONS_FILE.exists(), f"Captions file {CAPTIONS_FILE} does not exist. Run the cells above."

# Function: 

def parse_captions(captions_file, images_dir):

    """
    
    Takes in two arguments:

    - captions_file: The path to the captions file
    - images_dir: The path to the directory containing images

    Returns:

    - df: A pandas DataFrame with two columns: "image" and "caption"
    - caps_by_image: A dictionary mapping each image filename to a list of its captions



    """

    try:

        df = pd.read_csv(captions_file)

        if set(df.columns.map(str.lower)) >= {"image", "caption"}:

            df = df.rename(columns={i: i.lower() for i in df.columns})

        else:

            df = pd.read_csv(captions_file, header=None, names=["image", "caption"])

    except Exception as e:

        print(f"Error reading {captions_file}: {e}")

        df = pd.read_csv(captions_file, header=None, names=["image", "caption"], engine="python")

    # Cleaning: Remove whitespaces. Check rows for missing files and remove both that image and caption

    df["image"] = df["image"].astype(str).str.strip()
    df["caption"] = df["caption"].astype(str).str.strip()
    df = df["image"].map(lambda yo: (images_dir / yo)).to_frame().join(df["caption"])

    # Group captions that belong to the same image
    # We will, however, only use one caption per image for training

    caps_by_img = defaultdict(list)

    for r in df.itertuples(index=False):

        caps_by_img[r.image].append(r.caption)

    return df, caps_by_img

df, caps_by_img = parse_captions(CAPTIONS_FILE, IMAGES_DIR)

df.head()  # First 5 rows of the dataframe


# Training Split

pairs = [
    
    {"image_path": str(IMAGES_DIR / f), "caption": caps[0]}

         for f, caps in caps_by_img.items()
         
         ]

random.seed(42)
random.shuffle(pairs)

split = [.75, .15, .1]  # Train, Val, Test
amount = 8000

n = len(pairs)
n_train = min(amount * split[0], int(split[0] * n))
n_val = min(amount * split[1], int(split[1] * n))

train_pairs = pairs[: int(n_train)]
val_pairs = pairs[int(n_train) : int(n_train + n_val)]
test_pairs = pairs[int(n_train + n_val) : amount]

len(train_pairs), len(val_pairs), len(test_pairs)

ds = DatasetDict(
    
    {
        
        "train": Dataset.from_pandas(pd.DataFrame(train_pairs)),
        "val": Dataset.from_pandas(pd.DataFrame(val_pairs)),
        "test": Dataset.from_pandas(pd.DataFrame(test_pairs)),

    }
)

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
cap_id = "nlpconnect/vit-gpt2-image-captioning"

image_processor = ViTImageProcessor.from_pretrained(cap_id)
tokenizer       = AutoTokenizer.from_pretrained(cap_id)
model           = VisionEncoderDecoderModel.from_pretrained(cap_id).to("cpu")


def preprocess(batch):

    images = []

    for image_path in batch["image_path"]:

        try:

            image = Image.open(image_path).convert("RGB")

        except Exception as e:

            img = Image.new("RGB", (2,2), color = (0, 0, 0))

        images.append(img)
    
    pix = image_processor(
        
        images=images, 
        
        return_tensors="pt"
        
        )

    tok = tokenizer(
        
        batch["caption"], 
        
        padding="max_length", 
        
        max_length=32, 
        
        return_tensors="pt"
        
        )

    labels = tok.input_ids.clone()

    labels[labels == tokenizer.pad_token_id] = -100

    return {"pixel_values": pix["pixel_values"], "labels": labels}


processed_ds = ds.with_transform(preprocess)

class AccuracyTrackingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.csv_file = "training_metrics.csv"
        self.training_start_time = datetime.now()
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'train_loss', 'eval_loss', 'learning_rate', 'timestamp'])
        
        print(f"Training metrics will be logged to: {self.csv_file}")
    
    def log(self, logs):
        super().log(logs)
        
        epoch = logs.get('epoch', 0)
        step = logs.get('step', 0)
        train_loss = logs.get('train_loss', None)
        eval_loss = logs.get('eval_loss', None)
        learning_rate = logs.get('learning_rate', None)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if train_loss is not None or eval_loss is not None:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, step, train_loss, eval_loss, learning_rate, timestamp])
            
            print(f"Step {step} - Epoch {epoch:.2f}")
            if train_loss is not None:
                print(f"  Train Loss: {train_loss:.4f}")
            if eval_loss is not None:
                print(f"  Eval Loss: {eval_loss:.4f}")
            if learning_rate is not None:
                print(f"  Learning Rate: {learning_rate:.2e}")
            print(f"  Time: {timestamp}")
            print("-" * 40)
    
    

OUT_DIR = "finetuned-model-blip-flicker8k"

args = TrainingArguments(

    output_dir=OUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    eval_steps=200,
    logging_steps=50,
    save_steps=400,
    save_total_limit=1,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    report_to=[],
    evaluation_strategy="steps",  # Enable evaluation during training
    
)


trainer = AccuracyTrackingTrainer(

    model=model,
    args=args,
    train_dataset=processed_ds["train"],
    eval_dataset=processed_ds["val"],
    data_collator=default_data_collator,
)

train_res = trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)  
def display_training_metrics():
    print("\n" + "="*60)
    print("TRAINING METRICS SUMMARY")
    print("="*60)
    
    try:
        df_metrics = pd.read_csv("training_metrics.csv")
        
        print(f"Total training steps: {len(df_metrics)}")
        print(f"Final train loss: {df_metrics['train_loss'].dropna().iloc[-1]:.4f}")
        print(f"Final eval loss: {df_metrics['eval_loss'].dropna().iloc[-1]:.4f}")
        print(f"Best eval loss: {df_metrics['eval_loss'].dropna().min():.4f}")
        
        print("\nLast 10 training steps:")
        print(df_metrics.tail(10).to_string(index=False))
        
        print(f"\nFull metrics saved to: training_metrics.csv")
        
    except Exception as e:
        print(f"Error reading metrics: {e}")

display_training_metrics()

