# --------------
# Imports
# --------------
import io, os, time, threading
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify, abort
import torch
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
from dotenv import load_dotenv

# --------------
# Config
# --------------

load_dotenv()

app = Flask(__name__)

API_KEYS = os.getenv("API_KEYS", "").split(", ")

print("API_KEYS:", API_KEYS)

# --------------
# Auth
# --------------

@app.route('/')

def home():

    return "Hello, Flask is running!"

if __name__ == '__main__':

    app.run(debug=True)
