import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

# Models
LLM_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
