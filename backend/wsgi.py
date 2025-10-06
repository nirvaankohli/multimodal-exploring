import os
from dotenv import load_dotenv

load_dotenv()

from app import app

application = app

if __name__ == "__main__":

    application.run(host='0.0.0.0', port=8000)
