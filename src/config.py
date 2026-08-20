"""config.py — Load env vars and initialise API clients."""
import os
from dotenv import load_dotenv
from google import genai
from hydra_db import HydraDB

load_dotenv()

HYDRA_DB_API_KEY: str = os.environ["HYDRA_DB_API_KEY"]
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Singleton clients — import these everywhere
hydra = HydraDB(token=HYDRA_DB_API_KEY)
gemini = genai.Client(api_key=GEMINI_API_KEY)