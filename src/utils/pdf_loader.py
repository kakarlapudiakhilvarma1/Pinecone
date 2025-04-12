from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config.settings import PDF_DIRECTORY, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS

def load_and_process_pdfs():
    """Load and process PDF files from the specified directory."""
    # Load PDFs from directory
    loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS
    )
    splits = text_splitter.split_documents(documents)
    
    return splits