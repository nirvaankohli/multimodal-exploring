import os, random, math, re, argparse, json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import torch
from PIL import Image
import pandas as pd

from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    VisionEncoderDecoderModel,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback,
    set_seed,
)

MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except Exception:
    SACREBLEU_AVAILABLE = False

# --------------
# Config
# --------------

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "Images"
CAPTIONS_FILE = DATA_DIR / "captions.txt"

MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"

OUT_DIR = "captioner-finetuned-flicker8k"

INTENSITY = 4
torch.set_num_threads(max(1, int(os.cpu_count() // (1 / (INTENSITY / 10)))))
torch.set_num_interop_threads(1)

SEED = 42
set_seed(SEED)

SPLIT          = (0.75, 0.15, 0.10)
AMOUNT_LIMIT   = 8000

MAX_TOKENS = 32

GEN_KW = dict(
    
    max_length=MAX_TOKENS,
    min_length=6,
    num_beams=4,
    no_repeat_ngram_size=3,
    length_penalty=1.15,
    early_stopping=True

)

TRAINING_ARGS = dict(
    output_dir=OUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=20,
    learning_rate=1e-4,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=250,
    save_steps=500,
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  
    greater_is_better=False,
    remove_unused_columns=False,
    report_to=[],
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    fp16=False, bf16=False,
)

# --------------
# Data
# --------------

def parse_captions(captions_file: Path) -> List[Dict]:

    """

    Reads captions file (CSV with header image,caption or two columns w/o header)
    
    Returns:
      df: rows with absolute image paths and captions
      caps_by_img: dict[Path] -> List[str] (all captions per image)
    
    """

    try: 

        df = pd.read_csv(captions_file)

        if set(df.columns.map(str.lower)) >= {"image", "caption"}:

            df = df.rename(columns={c: c.lower() for c in df.columns})

        else:
            
            df = pd.read_csv(captions_file, header=None, names=["image", "caption"])

    except Exception:

        df = pd.read_csv(captions_file, header=None, names=["image", "caption"])

    
    df["image"] = df["image"].astype(str).str.strip()
    df["caption"] = df["caption"].astype(str).str.strip()
    df["image_path"] = df["image"].apply(lambda x: IMAGES_DIR / x)

    # filter out non-existing images

    df = df[df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)

    caps_by_img = defaultdict(list)

    for r in df.itertuples(index=False):

        caps_by_img[r.image_path].append(r.caption)

    print(caps_by_img)

    return df, caps_by_img

def build_da_splits(caps_by_img: Dict[Path, List[str]], amount_limit=AMOUNT_LIMIT, split=SPLIT):

    """
    
    Builds train/val/test splits from the captions dict
    
    """

    pairs = []

    for p, caps in caps_by_img.items():

        for c in caps:

            pairs.append({"image_path": str(p), "caption": c})

    random.seed(SEED)
    random.shuffle(pairs)

    n = min(len(pairs), amount_limit)

    n_train = int(n * split[0])
    n_val   = int(n * split[1])
    n_test  = n - n_train - n_val

    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:n_train+n_val]
    test_pairs  = pairs[n_train+n_val:n_train+n_val+n_test]

    print(f"Total pairs: {len(pairs)}. Using {n} pairs.")

    ds = DatasetDict({
        "train": Dataset.from_pandas(pd.DataFrame(train_pairs)),
        "val":   Dataset.from_pandas(pd.DataFrame(val_pairs)),
        "test":  Dataset.from_pandas(pd.DataFrame(test_pairs)),
    })

    return ds

# --------------
# Cleanup the Text
# --------------

def cleanup_text(text: str) -> str:

    s = re.sub(r"\s*([.,!?;:])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(?:\b\w+\b\s+)\1+", r"\1 ", s)
    if s and s[0].islower(): s = s[0].upper() + s[1:]
    if s and s[-1] not in ".!?": s += "."
    return s

# --------------
# Preprocessing
# --------------

def make_preprocess(image_processor, tokenizer):

    pad_id = tokenizer.pad_token_id

    def preprocess(batch):
        images = []
        for p in batch["image_path"]:
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                im = Image.new("RGB", (224, 224), color=(0, 0, 0))
            images.append(im)

        pix = image_processor(images=images, return_tensors="pt")
        tok = tokenizer(
            batch["caption"],
            padding="max_length",
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        labels = tok.input_ids.clone()
        labels[labels == pad_id] = -100
        return {"pixel_values": pix["pixel_values"], "labels": labels}

    return preprocess

# --------------
# Metrics & all att
# --------------

def postprocess_text(preds: List[str]) -> List[str]:

    preds = [p.strip() for p in preds]
    preds = [clean_caption(p) for p in preds]
    return preds

def compute_bleu(preds: List[str], refs: List[str]) -> Dict:

    if not SACREBLEU_AVAILABLE:

        return {"bleu": float}

    bleu = sacrebleu.corpus_bleu(preds, [refs])

    return {"bleu": bleu.score}

def build_compute_metrics(tokenizer):

    def _compute(eval_pred):

        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids


        preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        label_ids = [

            [tok if tok != -100 else tokenizer.pad_token_id for tok in seq]
            for seq in label_ids

        ]

        refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        preds = postprocess_text(preds)
        refs  = postprocess_text(refs)

        return compute_bleu(preds, refs)
    
    return _compute

# --------------
# Build/ Freeze ts/ lora
# --------------

def attach_lora(model):

    if not PEFT_AVAILABLE:

        print("PEFT not installed; training full decoder params only.")
        return model

    target_modules = ["c_attn", "c_proj", "c_fc"]
    task_type = TaskType.CAUSAL_LM
    modules_to_save = None

    lora = LoraConfig(

        r=8,

        lora_alpha=16,

        lora_dropout=0.05,

        bias="none",
        
        task_type=task_type,

        target_modules=target_modules,

        modules_to_save=modules_to_save,
    )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    return model

def freeze_encoder_unfreeze_decoder(model):

    for p in model.parameters():

        p.requires_grad = False

    for name, p in model.named_parameters():

        if name.startswith("decoder"):

            p.requires_grad = True
        
        if "crossattention" in name or "encoder_to_decoder" in name or "proj" in name:
            
            p.requires_grad = True
    
    return model


def generate_captions(model, processor, tokenizer, paths: List[str]) -> List[str]:
    
    images = [Image.open(p).convert("RGB") for p in paths]
    batch = processor(images=images, return_tensors="pt")
    
    pixel_values = batch["pixel_values"]
    
    with torch.no_grad():
    
        out = model.generate(pixel_values=pixel_values, **GEN_KW)
    
    caps = tokenizer.batch_decode(out, skip_special_tokens=True)
    caps = [cleanup_text(c) for c in caps]
    
    return caps

# --------------
# Main
# --------------

def main():

    assert IMAGES_DIR.exists(), f"Images dir {IMAGES_DIR} does not exist."
    assert CAPTIONS_FILE.exists(), f"Captions file {CAPTIONS_FILE} does not exist."

    print("Parsing captions...")

    df, caps_by_img = parse_captions(CAPTIONS_FILE)

    print(f"Rows(raw) {len(df)}. | Unique images {len(caps_by_img)}.")

    ds = build_da_splits(caps_by_img, amount_limit=AMOUNT_LIMIT, split=SPLIT)

    print({k: len(v) for k, v in ds.items()})

    print("Loading model and tokenizer...")

    print("Model ID:", MODEL_ID )

    image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    tokenizer       = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model           = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    if hasattr(model.config, "max_length") and model.config.max_length is None:

        model.config.max_length = MAX_TOKENS

    model = freeze_encoder_unfreeze_decoder(model)

    model = attach_lora(model)

    print("Preparing the data...")

    preprocess = make_preprocess(image_processor, tokenizer)
    processed_ds = ds.with_transform(preprocess)

    print("Computing ts metrics...")

    compute_metrics = build_compute_metrics(tokenizer)

    print("Preparing trainer...")

    args = TrainingArguments(**TRAINING_ARGS)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=processed_ds["train"],
        eval_dataset=processed_ds["val"],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting training...")

    trainer.train()

    print("Saving model + processors...")
    trainer.save_model(OUT_DIR)
    image_processor.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print("Running quick inference sample...")

    sample = processed_ds["val"].shuffle(seed=SEED).select(range(min(4, len(processed_ds["val"]))))
    paths = [sample[i]["image_path"] for i in range(len(sample))]
    caps  = generate_captions(model, image_processor, tokenizer, paths)
    
    for p, c in zip(paths, caps):

        print(f"[CAPTION] {Path(p).name}: {c}")

    try:

        import torch.quantization as tq

        q_model = torch.quantization.quantize_dynamic(

            model, 
            
            {torch.nn.Linear}, 
            
            dtype=torch.qint8

        )

        q_dir = OUT_DIR + "-int8"
        q_model.save_pretrained(q_dir)

        print(f"Saved int8 quantized model to: {q_dir}")

    except Exception as e:

        print("INT8 quantization skipped (torch build may not support it).")


if __name__ == "__main__":
    main()