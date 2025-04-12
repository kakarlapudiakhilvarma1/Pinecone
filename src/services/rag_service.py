from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.config.settings import GOOGLE_API_KEY, TOP_K
from src.utils.prompt_templates import get_rag_prompt_template

def create_rag_chain(vector_store):
    """Create and return a RAG chain."""
    # Initialize Google Generative AI model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    
    # Get prompt template
    prompt = get_rag_prompt_template()
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain