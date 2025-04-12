from pinecone import Pinecone, ServerlessSpec
from src.config.settings import (
    PINECONE_API_KEY, 
    INDEX_NAME, 
    PINECONE_DIMENSION, 
    PINECONE_METRIC,
    PINECONE_CLOUD,
    PINECONE_REGION
)

def init_pinecone():
    """Initialize Pinecone client and index."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, if not create it
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )
    
    # Get the index
    index = pc.Index(INDEX_NAME)
    return index, pc