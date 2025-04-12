from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.config.settings import GOOGLE_API_KEY, INDEX_NAME

def create_vector_store(splits):
    """Create and return a vector store using Google Embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=GOOGLE_API_KEY
    )
    
    vector_store = PineconeVectorStore.from_documents(
        splits, 
        embeddings, 
        index_name=INDEX_NAME
    )
    
    return vector_store