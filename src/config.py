"""Configuration settings for the Food Recommendation RAG System."""

import os
from pathlib import Path
from typing import Dict, Any

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_EMBEDDING_MODEL = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
BEDROCK_LLM_MODEL = os.getenv("BEDROCK_LLM_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")

# Model Parameters
LLM_MODEL_KWARGS: Dict[str, Any] = {
    "max_tokens": int(os.getenv("MAX_TOKENS", "10000")),
    "temperature": float(os.getenv("TEMPERATURE", "0.0")),
    "stop_sequences": ["\n\nHuman"],
}

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_PATH = BASE_DIR / "output" / "faiss_index"
CSV_DATA_PATH = DATA_DIR / "restaurants_menu_data.csv"

# Search Configuration
SEARCH_K = int(os.getenv("SEARCH_K", "5"))  # Number of results to retrieve
MAX_RECOMMENDATIONS = int(os.getenv("MAX_RECOMMENDATIONS", "3"))

# Image Configuration
ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg"]
MAX_IMAGE_SIZE_MB = 10

# Retry Configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

