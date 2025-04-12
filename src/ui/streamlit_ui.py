import streamlit as st

def setup_page():
    """Setup the Streamlit page configuration."""
    st.set_page_config(page_title="PDF RAG Application", layout="wide")
    st.title("RAG Application with Google Embeddings and Pinecone")

def render_sidebar():
    """Render the sidebar with processing button."""
    with st.sidebar:
        st.header("Process Documents")
        return st.button("Process PDF Files")

def render_question_input():
    """Render the question input field."""
    st.header("Ask Questions About Your Documents")
    return st.text_input("Enter your question:", key="question_input")

def render_response(response):
    """Render the response and source documents."""
    st.subheader("Answer:")
    st.write(response["answer"])
    
    st.subheader("Source Documents:")
    for idx, doc in enumerate(response["context"]):
        with st.expander(f"Document {idx + 1}"):
            st.write(doc.page_content)
            st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")

def render_about():
    """Render the about section."""
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