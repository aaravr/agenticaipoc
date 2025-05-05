# src/config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = "http://localhost:8080/api"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")