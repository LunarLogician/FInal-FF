import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")

# Ports
MAIN_SERVICE_PORT = int(os.environ.get("MAIN_SERVICE_PORT", 5001))
ESG_SERVICE_PORT = int(os.environ.get("ESG_SERVICE_PORT", 5002))
ADDITIONAL_SERVICE_PORT = int(os.environ.get("ADDITIONAL_SERVICE_PORT", 5004))

# Model Settings
MAX_TOKENS = 600
MAX_LENGTH = 512
MODEL_NAME = "gpt-3.5-turbo"

# CORS Settings
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5001",
    "http://localhost:5002",
    "http://localhost:5004",
] 