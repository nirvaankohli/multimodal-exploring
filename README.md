# Multimodal Image Captioning

A simple image captioning project split into two parts: training experiments and a working API.


## What's This About?

This project has two main parts:

### 1. Training (`example/` folder)
I tried fine-tuning a BLIP model on the Flickr8k dataset. Spoiler alert: it didn't work out great. The results were pretty mediocre and the model didn't learn much beyond what it already knew. You can check out the training code in `exploring.py` and `exploring.ipynb` if you're curious about ts.

### 2. API (`backend/` folder) 
This is the part that actually works! A Flask API that uses the pre-trained Salesforce BLIP model to generate captions for images. It's got a simple web interface and REST endpoints.

## Quick Start

```bash
# Install stuff
cd backend
pip install -r requirements.txt

# Run the server
python app.py
```

Go to `http://localhost:5000` and upload some images!

## API Usage

Generate an API key:
```bash
curl -X POST http://localhost:5000/generate-api-key
```

Caption an image:
```bash
curl -X POST http://localhost:5000/generate-caption \
  -H "X-API-KEY: your_key_here" \
  -F "image=@your_image.jpg"
```

That's it! The API uses the pre-trained BLIP model which actually works pretty well for general image captioning.

## Why Two Parts?

Originally wanted to train a custom model, but turns out fine-tuning image captioning models is harder than it looks. The pre-trained model does a good job anyway, so I just built an API around that instead.

The training code is still there if you want to see what went wrong or try to improve it!