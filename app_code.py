import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone, ServerlessSpec

# Set page configuration
st.set_page_config(page_title="PDF RAG Application", layout="wide")
st.title("RAG Application with Google Embeddings and Pinecone")


from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "pdf-rag-index"      #Add your required index name to create/connect with pinecone

# Initialize Pinecone
@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, if not create it
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,  # Dimension for Google's text embeddings
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    
    # Get the index
    index = pc.Index(INDEX_NAME)
    return index, pc

# Function to load and process PDFs
@st.cache_data
def load_and_process_pdfs():
    # Load PDFs from directory
    pdf_directory = "pdf_files"  # Directory containing PDFs
    loader = PyPDFDirectoryLoader(pdf_directory)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    
    return splits

# Function to create vector store
@st.cache_resource
def create_vector_store(_splits):
    # Initialize Google Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    # Create and persist vector store in Pinecone
    vector_store = PineconeVectorStore.from_documents(
        _splits, 
        embeddings, 
        index_name=INDEX_NAME
    )
    
    return vector_store

# Create RAG Chain
@st.cache_resource
def create_rag_chain(_vector_store):
    # Initialize Google Generative AI model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)
    
    # Create retriever
    retriever = _vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Create prompt template - Note the change from {question} to {input}
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant that responds to queries based on the provided documents.
    Everytime don't give the same response, only follow the below format when questions asked about alarms from context.\n
        Act as a Telecom NOC Engineer with expertise in Radio Access Networks (RAN).
        Response should be in short format.
        Your responses should follow this structured format:
            1. Response: Provide an answer based on the given situation, with slight improvements for better clarity but from the context.
            2. Explanation of the issue: Include a brief explanation on why the issue might have occurred.
            3. Recommended steps/actions: Suggest further steps to resolve the issue.
            4. Quality steps to follow:
                - Check for relevant INC/CRQ tickets.
                - Follow the TSDANC format while creating INC.
                - Mention previous closed INC/CRQ information if applicable.
                - If there are >= 4 INCs on the same issue within 90 days, highlight the ticket to the SAM-SICC team and provide all relevant details.
        
        Context: {context}
        Question: {input}
    
    Answer:
    """)
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain

# Sidebar for processing
with st.sidebar:
    st.header("Process Documents")
    process_button = st.button("Process PDF Files")
    
    if process_button:
        with st.spinner("Initializing Pinecone..."):
            pinecone_index, pinecone_client = init_pinecone()
            st.session_state.pinecone_client = pinecone_client
        
        with st.spinner("Loading and processing PDFs..."):
            splits = load_and_process_pdfs()
            st.session_state.document_chunks = len(splits)
        
        with st.spinner("Creating vector store with Google Embeddings..."):
            vector_store = create_vector_store(splits)
            st.session_state.vector_store = vector_store
        
        with st.spinner("Creating RAG chain..."):
            rag_chain = create_rag_chain(vector_store)
            st.session_state.rag_chain = rag_chain
        
        st.success(f"Processed {st.session_state.document_chunks} document chunks successfully!")

# Main content
st.header("Ask Questions About Your Documents")

# Input for questions
question = st.text_input("Enter your question:", key="question_input")

if question:
    if 'rag_chain' not in st.session_state:
        st.warning("Please process the PDF files first by clicking the button in the sidebar.")
    else:
        with st.spinner("Retrieving relevant information and generating answer..."):
            # Get response from RAG chain - Note the change from "question" to "input"
            response = st.session_state.rag_chain.invoke({"input": question})
            
            # Display answer
            st.subheader("Answer:")
            st.write(response["answer"])
            
            # Display source documents
            st.subheader("Source Documents:")
            for idx, doc in enumerate(response["context"]):
                with st.expander(f"Document {idx + 1}"):
                    st.write(doc.page_content)
                    st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")

# Information about the application
with st.expander("About this RAG Application"):
    st.write("""
    This application uses:
    - PyPDFDirectoryLoader to load PDF files from the "pdf_files" directory
    - LangChain for the document processing pipeline
    - Google Generative AI Embeddings to create vector representations of text
    - Pinecone as the vector database for storing and retrieving document chunks
    - Google's Generative AI (Gemini) for answering questions
    - Streamlit for the user interface
    
    To use:
    1. Make sure your PDF files are in the "pdf_files" directory
    2. Click "Process PDF Files" in the sidebar
    3. Ask questions related to the content of your PDFs
    """)
