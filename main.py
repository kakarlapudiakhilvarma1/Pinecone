import streamlit as st
from src.ui.streamlit_ui import (
    setup_page,
    render_sidebar,
    render_question_input,
    render_response,
    render_about
)
from src.utils.pdf_loader import load_and_process_pdfs
from src.services.pinecone_service import init_pinecone
from src.services.embedding_service import create_vector_store
from src.services.rag_service import create_rag_chain

def main():
    setup_page()
    process_button = render_sidebar()
    
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

    question = render_question_input()

    if question:
        if 'rag_chain' not in st.session_state:
            st.warning("Please process the PDF files first by clicking the button in the sidebar.")
        else:
            with st.spinner("Retrieving relevant information and generating answer..."):
                response = st.session_state.rag_chain.invoke({"input": question})
                render_response(response)

    render_about()

if __name__ == "__main__":
    main()