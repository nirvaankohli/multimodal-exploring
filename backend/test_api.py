import requests
import os
from dotenv import load_dotenv

BASE_URL = "http://127.0.0.1:5000"
URL = f"{BASE_URL}/generate-caption"

load_dotenv()

env = [[i[0], i[1], i[2].split('*')] for i in (k.split("|") for k in os.getenv("API_KEYS", "").split(","))][0][0]

key_id = "key_0"

# testing file transfer

data = {
    "use-case" : "test",
    "model": "tbd",
    "language": "en"
}

img_name = "example2.jpg"

with open(f"exampleimgs/{img_name}", "rb") as f:

    files = {
        "image": (img_name, f, "image/jpeg")
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

print("Response from captioning model(raw):")

print(resp.json()['results'][0]['caption'])

resp = requests.post("https://ai.hackclub.com/chat/completions",
    headers={
      "Content-Type": "application/json"
    },
    json={
      "messages": [
        {
          "content": f"""
You are an expert image captioner. Given a vague caption, improve it to be more detailed. Do not change the meaning of the caption, only elaborate and fix grammar. It should be max a sentence or two.
Here is the vague caption: {resp.json()['results'][0]['caption']}

""",
          "role": "user"
        }
      ], 
      
      "model": "openai/gpt-oss-120b"
    
    }
)

print("Response from GPT-OSS:")

print(resp.json()['choices'][0]['message']['content'])
