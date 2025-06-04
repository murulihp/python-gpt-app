import streamlit as st
import os
import sys

# Ensure the project root is in sys.path
# This allows imports from 'src' when running with streamlit run app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir # In this case, app.py is in the root
sys.path.append(project_root)

# Import your existing modules
from src.pdf_processor import PDFProcessor
from src.ai_responder import AIResponder

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📚",
    layout="centered" # "wide" for wider layout
)

st.title("📚 PDF Q&A Assistant")
st.markdown("Ask questions about your PDF documents.")

# --- Caching expensive resources ---
# Use st.cache_resource to avoid reloading the PDFProcessor and AIResponder
# every time the app reruns (e.g., when a user types something)
@st.cache_resource
def load_pdf_processor():
    """Initializes and processes PDFs. Runs only once."""
    pdf_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    if not os.path.exists(pdf_dir):
        st.error(f"Data directory '{pdf_dir}' not found. Please create it and place your PDFs inside.")
        st.stop() # Stop the app execution

    processor = PDFProcessor(pdf_dir)
    with st.spinner("Processing PDFs... This may take a moment."):
        processor.process_pdfs()
    
    if not processor.knowledge_base:
        st.warning("No PDF documents found or successfully processed in the 'data' directory.")
        st.warning("Please ensure your PDFs are valid and placed in the correct folder.")
        st.stop() # Stop if no PDFs are processed
        
    return processor

@st.cache_resource
def load_ai_responder():
    """Initializes the AIResponder. Runs only once."""
    try:
        responder = AIResponder()
        st.success(f"AI Responder initialized with model: {responder.model.model_name}")
        return responder
    except ValueError as e:
        st.error(f"Error initializing AI Responder: {e}")
        st.error("Please ensure your GEMINI_API_KEY is correctly set in your .env file.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during AI Responder initialization: {e}")
        st.stop()


# --- Load Resources ---
pdf_processor = load_pdf_processor()
ai_responder = load_ai_responder()


# --- User Interface ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your PDFs..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            # 1. Get relevant chunks
            relevant_chunks = pdf_processor.get_relevant_chunks(prompt)

            if not relevant_chunks:
                response = "I could not find highly relevant information in the documents for your question using keyword search."
            else:
                # 2. Generate answer using AI
                response = ai_responder.generate_answer(prompt, relevant_chunks)
            
            st.markdown(response)
    # Add assistant response to chat history
    st.session_state["messages"].append({"role": "assistant", "content": response})

st.markdown("---")
