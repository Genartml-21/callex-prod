"""
Configuration Module for Lakhu Teleservices Voice Bot
Central configuration - all credentials and settings in one place
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory (Project Root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

# ─── Database Configuration ───
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/data/call_recordings.db")

# ─── Recording Storage ───
RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR", BASE_DIR / "recordings"))
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

RECORDING_FORMAT = "wav"
RECORDING_SAMPLE_RATE = 16000  # Hz
RECORDING_CHANNELS = 1  # Mono

# ─── Retention Policy ───
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", 90))

# ─── API Settings ───
API_CORS_ORIGINS = os.getenv("API_CORS_ORIGINS", "*").split(",")
API_PREFIX = "/api"

# ─── Pagination ───
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200

# ─── Firebase Storage Configuration ───
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_credentials.json")
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", "lakhuteleservices-1f9e0.appspot.com")

# ─── AWS Settings ───
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "callex-recordings")
