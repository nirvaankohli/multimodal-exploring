# --------------
# Imports
# --------------
import io, os, time, threading
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
import torch
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
from dotenv import load_dotenv
from hmac import compare_digest
from flask import Flask, request, jsonify, g
from functools import wraps
import hashlib

# --------------
# Config
# --------------

load_dotenv()

app = Flask(__name__)

API_KEYS = [[i[0], i[1], i[2].split('*')] for i in (k.split("|") for k in os.getenv("API_KEYS", "").split(","))]


KEYS  = {}

for key in range(len(API_KEYS)):

    key = API_KEYS[key]
    
    KEYS["key_" + str(key)] = {

        "hash": hashlib.sha256(key[0].encode()).hexdigest(),
        "owner": key[1],
        "scopes": key[2]

    }


def verify_api_key(raw_header: str):
    filler = (1)


# --------------
# Auth
# --------------

@app.route('/')

def home():

    return "Hello, this API is up and running!"

@app.route('/generate-caption', methods=['POST'])
def generate_caption():

    data = request.get_json()
    username = data['username']
    password = data['password']







if __name__ == '__main__':

    app.run(debug=True)
