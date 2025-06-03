import os
import sys

# Add the project root to the sys.path to allow imports from 'src'
# This handles cases where the script might be run directly without -m
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)

from src.pdf_processor import PDFProcessor
from src.ai_responder import AIResponder

def main():
    # Define the directory where PDF documents are stored
    # It's set to 'data' folder one level up from 'src'
    pdf_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    # Ensure the 'data' directory exists
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"Created data directory: {pdf_dir}")
        print("Please place your PDF documents inside the 'data' folder.")
        print("Exiting. Please rerun the application after adding PDFs.")
        return

    # Initialize the PDF processor and load content from PDFs
    pdf_processor = PDFProcessor(pdf_dir)
    pdf_processor.process_pdfs()

    # Check if any PDFs were successfully processed and added to the knowledge base
    if not pdf_processor.knowledge_base:
        print("No PDF documents found or successfully processed in the 'data' directory.")
        print("Please ensure your PDFs are valid and placed in the correct folder.")
        return

    # Initialize the AI responder (will configure Gemini API)
    try:
        ai_responder = AIResponder()
    except ValueError as e:
        print(f"Error initializing AI Responder: {e}")
        print("Please ensure your GEMINI_API_KEY is correctly set in your .env file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during AI Responder initialization: {e}")
        return

    print("\n--- PDF Q&A Assistant is Ready! ---")
    print("Type your questions below. Type 'exit' to quit.")

    # Main interaction loop
    while True:
        user_question = input("\nYour Question: ").strip()

        if user_question.lower() == 'exit':
            print("Exiting PDF Q&A Assistant. Goodbye!")
            break

        if not user_question:
            print("Please enter a question.")
            continue

        print("Searching for relevant information and generating answer...")

        # Retrieve relevant chunks from the PDF knowledge base
        relevant_chunks = pdf_processor.get_relevant_chunks(user_question)

        if not relevant_chunks:
            print("No highly relevant information found in the documents for your question using keyword search.")
            print("Assistant: I cannot answer that question based on the provided documents.")
            continue

        # Generate an answer using the AI responder with the question and relevant context
        answer = ai_responder.generate_answer(user_question, relevant_chunks)
        print(f"\nAssistant: {answer}")

if __name__ == "__main__":
    main()