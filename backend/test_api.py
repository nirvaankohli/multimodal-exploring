import requests
import os
from dotenv import load_dotenv

BASE_URL = "http://127.0.0.1:5000"
URL = f"{BASE_URL}/generate-caption"

load_dotenv()

env = [[i[0], i[1], i[2].split('*')] for i in (k.split("|") for k in os.getenv("API_KEYS", "").split(","))][0][0]

key_id = "key_0"

# just checking if the api auth works

print(requests.post(URL, headers={"X-API-KEY": f"{key_id}.{env}"}).json())

# testing file transfer

data = {
    "use-case" : "test",
    "model": "tbd",
    "language": "en"
}

with open("example.png", "rb") as f:

    files = {
        "image": ("example.png", f, "image/png")
    }
    
    resp = requests.post(

        URL,
        headers={
            "X-API-KEY": f"{key_id}.{env}"
            },
        data=data,
        files=files,
        timeout=30

    )

resp.raise_for_status()
print(resp.json())

