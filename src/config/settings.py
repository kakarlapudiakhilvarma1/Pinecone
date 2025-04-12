import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "pdf-rag-index"
PDF_DIRECTORY = "pdf_files"

# Pinecone settings
PINECONE_DIMENSION = 768
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Text splitting settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n\n", "\n", " ", ""]

# Retrieval settings
TOP_K = 3