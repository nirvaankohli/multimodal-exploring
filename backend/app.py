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
ALLOWED_TYPE = {"image/jpeg", "image/png", "image/webp"}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

KEYS  = {}

for i in range(len(API_KEYS)):

    key = API_KEYS[i]
    
    KEYS["key_" + str(i)] = {

        "hash": hashlib.sha256(key[0].encode()).hexdigest(),
        "owner": key[1],
        "scopes": key[2],
        "plaintext": key[0]


    }




def verify_api_key(raw_header: str):
    
    """
    
    Except a raw API header in the following format:

    `
    X-API-KEY: <key_id>.<plain_key>
    `

    The `key_id` HAS to be one of the keys in the KEYS dictionary.

    """

    if not raw_header or "." not in raw_header or raw_header.split(".")[0] not in KEYS:

        

        return None, None


    key_id, plaintext = raw_header.split(".")

    rec = KEYS.get(key_id)

    h = hashlib.sha256(plaintext.encode()).hexdigest()

    if compare_digest(h, rec["hash"]):

        return rec, key_id

    print("It happened because of no match:", h, rec["hash"])
    return None, None


def require_key(*, scopes=None):

    scopes = set(scopes or [])

    def decorator(fn):

        @wraps(fn)

        def wrapper(*args, **kwargs):
            
            key_header = request.headers.get("X-API-KEY")

            rec, key_id = verify_api_key(key_header)

            if not rec:

                print(request.headers.get("X-API-KEY"))

                return jsonify({"error": "Unauthorized"}), 401
            if any(item in scopes for item in rec["scopes"]):

                return jsonify({"error": "Forbidden"}), 403
            
            g.api_key = key_id

            g.api_owner = rec["owner"]

            return fn(*args, **kwargs)

        return wrapper
    
    return decorator

# --------------
# Auth
# --------------

@app.route('/')

def home():

    return "Hello, this API is up and running!"

@app.route('/generate-caption', methods=['POST'])

@require_key(scopes=["all"])

def generate_caption():

    if "image" not in request.files:

        return jsonify({"message": "pop"})

    file = request.files['image']

    if file.mimetype not in ALLOWED_TYPE:
        
        return jsonify({"error": "Unsupported Media Type"}), 415
    
    raw = file.read()

    try:

        img = Image.open(io.BytesIO(raw))
        img.verify()                     
        img = Image.open(io.BytesIO(raw))
        img.load()

    except UnidentifiedImageError:

        return jsonify({"error": "invalid image"}), 400
    
    return jsonify({"message": "Image received successfully", "owner": g.api_owner, "key": g.api_key})


    

if __name__ == '__main__':

    app.run(debug=True)
