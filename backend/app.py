# --------------
# Imports
# --------------
import io, os, time, threading, json, uuid, secrets
from typing import List, Tuple
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
import torch
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
from dotenv import load_dotenv
from hmac import compare_digest
from flask import Flask, request, jsonify, g
from functools import wraps
import hashlib
from transformers import pipeline


# --------------
# Config
# --------------

load_dotenv()

app = Flask(__name__)





API_KEYS = [[i[0], i[1], i[2].split('*')] for i in (k.split("|") for k in os.getenv("API_KEYS", "").split(","))]
ALLOWED_TYPE = {"image/jpeg", "image/png", "image/webp"}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

MODEL_NAME = os.getenv("MODEL_NAME", "Salesforce/blip-image-captioning-base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_INDEX = 0 if DEVICE == "cuda" else -1
MAX_SIDE = int(os.getenv("MAX_SIDE", "512"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "24"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

if DEVICE == "cuda":

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    AMP_DTYPE = torch.float16 if torch.cuda.is_bf16_supported() else torch.float32

else: 

    torch.set_num_threads(max(1, os.cpu_count() - 1))
    AMP_DTYPE = torch.float32

# Global variables for model and threading
captioner = None
_caption_lock = threading.Lock()

KEYS  = {}

# Load user tokens from JSON file
USER_TOKENS_FILE = "user_tokens.json"

def load_user_tokens():
    """Load user tokens from JSON file"""
    if os.path.exists(USER_TOKENS_FILE):
        try:
            with open(USER_TOKENS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def save_user_tokens(tokens_data):
    """Save user tokens to JSON file"""
    with open(USER_TOKENS_FILE, 'w') as f:
        json.dump(tokens_data, f, indent=2)

user_tokens_db = load_user_tokens()

for i in range(len(API_KEYS)):

    key = API_KEYS[i]
    
    KEYS["key_" + str(i)] = {

        "hash": hashlib.sha256(key[0].encode()).hexdigest(),
        "owner": key[1],
        "scopes": key[2],
        "plaintext": key[0]


    }


# --------------
# Auth
# --------------


def verify_api_key(raw_header: str):
    

    
    if raw_header.startswith("user_"):

        user_tokens_db = load_user_tokens()

        api_key = raw_header



        if api_key in user_tokens_db:
        
            user_data = user_tokens_db[api_key]

        
            rec = {
                "hash": api_key,
                "owner": user_data.get("owner", "User"),
                "scopes": ["All", "User"],
                "plaintext": api_key,
                "tokens": user_data.get("tokens", 0)
            }
        
            return rec, api_key
        
        return None, None
    
    if not raw_header or "." not in raw_header:

        return None, None
    # Original API key verification for admin keys
    if raw_header.split(".")[0] not in KEYS:
        return None, None

    key_id, plaintext = raw_header.split(".")

    rec = KEYS.get(key_id)

    h = hashlib.sha256(plaintext.encode()).hexdigest()

    if compare_digest(h, rec["hash"]):

        return rec, key_id

    return None, None


def require_key(*, scopes=None):

    scopes = set(scopes or [])

    def decorator(fn):

        @wraps(fn)

        def wrapper(*args, **kwargs):

            
            key_header = request.headers.get("X-API-KEY")

            rec, key_id = verify_api_key(key_header)

            if not rec:

                
                return jsonify({"error": "Unauthorized"}), 401
            if not any(item in scopes for item in rec["scopes"]):

                return jsonify({"error": "Forbidden"}), 403
            
            g.api_key = key_id

            g.api_owner = rec["owner"]

            return fn(*args, **kwargs)

        return wrapper
    
    return decorator

# --------------
# Utility Functions
# --------------

def _resize_image(img: Image.Image, max_side: int) -> Image.Image:
    """Resize image while maintaining aspect ratio"""
    width, height = img.size
    if max(width, height) <= max_side:
        return img
    
    if width > height:
        new_width = max_side
        new_height = int(height * max_side / width)
    else:
        new_height = max_side
        new_width = int(width * max_side / height)
    
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def _preprocess_image(img: Image.Image) -> Image.Image:
    """Preprocess image for model input"""
    return img

def _chunk(lst: List, batch_size: int):
    """Yield successive n-sized chunks from lst"""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# --------------
# On Start
# --------------

def load_model():

    global captioner

    captioner = pipeline(

        task="image-to-text",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=DEVICE_INDEX,

    )

    dummy = Image.new("RGB", (MAX_SIDE, MAX_SIDE), color = (0, 0, 0))

    with torch.inference_mode():

        if DEVICE == "cuda":
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                captioner(dummy, generate_kwargs={"max_new_tokens": 4, "num_beams": 1})
        else:
            captioner(dummy, generate_kwargs={"max_new_tokens": 4, "num_beams": 1})

_model_loaded = False

def init_app():

    global _model_loaded
    if not _model_loaded:
        load_model()
        _model_loaded = True


init_app()


from flask import render_template

# --------------
# Routes
# --------------

@app.route('/')

def home():

    return render_template("index.html")

@app.route('/healthz', methods=['GET'])

def healthz():

    return jsonify({"status": "healthy", "device": DEVICE, "default_model": MODEL_NAME}), 200

@app.route('/generate-api-key', methods=['POST'])
def generate_api_key():
    """Generate a new API key for a user with 200 starting tokens"""
    try:

        api_key = f"user_{uuid.uuid4().hex}"
        

        user_data = {
            "api_key": api_key,
            "tokens": 200,
            "owner": f"User_{secrets.token_hex(4)}",
            "created_at": time.time(),
            "usage": 0
        }
        

        user_tokens_db[api_key] = user_data
        save_user_tokens(user_tokens_db)
        
        return jsonify({
            "api_key": api_key,
            "tokens": 200,
            "message": "API key generated successfully"
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to generate API key: {str(e)}"}), 500

@app.route('/check-tokens', methods=['POST'])
@require_key(scopes=["All"])
def check_tokens():

    try:

        key_header = request.headers.get("X-API-KEY")
        
        if key_header and key_header.startswith("user_"):

            user_data = user_tokens_db.get(key_header, {})
            tokens = user_data.get("tokens", 0)
            
            return jsonify({
                "tokens": tokens,
                "api_key": key_header,
                "owner": user_data.get("owner", "Unknown")
            }), 200
        
        else:

            return jsonify({
                "tokens": "unlimited",
                "api_key": key_header,
                "owner": getattr(g, 'api_owner', 'Admin')
            }), 200
            
    except Exception as e:
        return jsonify({"error": f"Failed to check tokens: {str(e)}"}), 500

def deduct_tokens(api_key: str, amount: int) -> bool:

    if not api_key or not api_key.startswith("user_"):
        return True
    
    if api_key in user_tokens_db:
        user_data = user_tokens_db[api_key]
        current_tokens = user_data.get("tokens", 0)
        
        if current_tokens >= amount:
            user_data["tokens"] = current_tokens - amount
            user_data["usage"] = user_data.get("usage", 0) + amount
            save_user_tokens(user_tokens_db)
            return True
    
    return False

@app.route('/generate-caption', methods=['POST'])
@require_key(scopes=["All"])
def generate_caption():

    if 'image' not in request.files:

        return jsonify({"error": "No image part in the request"}), 400
    
    files = request.files.getlist('image')

    if not files or files[0].filename == '':

        return jsonify({"error": "No selected file"}), 400 
    
    key_header = request.headers.get("X-API-KEY")
    tokens_cost = 10

    if key_header and key_header.startswith("user_"):

        user_data = user_tokens_db.get(key_header, {})

        available_tokens = user_data.get("tokens", 0)
        
        if available_tokens < tokens_cost:
            return jsonify({
                "error": f"Insufficient tokens. Required: {tokens_cost}, Available: {available_tokens}"
            }), 402  # Payment Required
    
    try:
        override_max_new_tokens = int(request.args.get("max_new_tokens", MAX_NEW_TOKENS))
    except Exception:
        override_max_new_tokens = MAX_NEW_TOKENS
    try:
        override_num_beams = int(request.args.get("num_beams", NUM_BEAMS))
    except Exception:
        override_num_beams = NUM_BEAMS

    to_process: List[Tuple[str, Image.Image]] = []
    results = []

    for file in files:

        if file.mimetype not in ALLOWED_TYPE:

            results.append({"filename": file.filename, "error": "Unsupported Media Type"})
            
            continue

        raw = file.read()

        try:

            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img = _resize_image(img, MAX_SIDE)
            img = _preprocess_image(img)

            to_process.append((file.filename or "image", img))
            
        except UnidentifiedImageError:

            results.append({"filename": file.filename, "error": "Invalid image"})

        except Exception as e:

            results.append({"filename": file.filename, "error": f"Invalid image: {e}"})

    if to_process:
        
        if not deduct_tokens(key_header, tokens_cost):
            return jsonify({
                "error": "Insufficient tokens for processing",
                "required_tokens": tokens_cost
            }), 402

        gen_kwargs = {

            "max_new_tokens": override_max_new_tokens,

            "num_beams": override_num_beams,

        }


        with torch.inference_mode():

            if DEVICE == "cuda":

                with _caption_lock:  
                    
                    for batch in _chunk([img for _, img in to_process], BATCH_SIZE):

                        with torch.autocast("cuda", dtype=AMP_DTYPE):

                            outs = captioner(batch, generate_kwargs=gen_kwargs)

                        for (fname, _), out in zip(to_process[: len(batch)], outs):

                            text = (out[0]["generated_text"] if isinstance(out, list) else out.get("generated_text", "")).strip()
                            
                            results.append({"filename": fname, "caption": text})
                        
                        to_process = to_process[len(batch):]
            else:
                
                with _caption_lock:
                    
                    for batch in _chunk([img for _, img in to_process], BATCH_SIZE):
                        
                        outs = captioner(batch, generate_kwargs=gen_kwargs)
                        
                        for (fname, _), out in zip(to_process[: len(batch)], outs):
                        
                            text = (out[0]["generated_text"] if isinstance(out, list) else out.get("generated_text", "")).strip()
                        
                            results.append({"filename": fname, "caption": text})
                        
                        to_process = to_process[len(batch):]


    response_data = {"results": results}

    if key_header and key_header.startswith("user_"):

        user_data = user_tokens_db.get(key_header, {})
        response_data["tokens_remaining"] = user_data.get("tokens", 0)
        response_data["tokens_used"] = tokens_cost

    return jsonify(response_data), 200

@app.route('/whoami', methods=['POST'])

@require_key(scopes=["All"])

def whoami():

    key_id = getattr(g, 'api_key', None)
    owner = getattr(g, 'api_owner', None)

    rec = KEYS.get(key_id) if key_id else None
    scopes = rec.get('scopes', []) if rec else []

    return jsonify({

        'key_id': key_id,

        'owner': owner,

        'scopes': scopes,

    }), 200


@app.route('/exampleimgs/<path:filename>')
def serve_exampleimg(filename):

    from flask import send_from_directory

    base = os.path.join(os.path.dirname(__file__), 'exampleimgs')

    return send_from_directory(base, filename)

    

if __name__ == '__main__':
    
    app.run(debug=True)
