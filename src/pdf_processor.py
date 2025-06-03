import os
from pypdf import PdfReader
from typing import List, Dict
from src.utils import chunk_text # Import the chunking utility

class PDFProcessor:
    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir
        # knowledge_base will store processed chunks:
        # Each item: {'filename': 'file.pdf', 'text': 'chunk of text content'}
        self.knowledge_base: List[Dict[str, str]] = []

    def process_pdfs(self) -> None:
        """
        Reads all PDF files in the specified directory, extracts text, and chunks it.
        The extracted and chunked text is then stored in the knowledge_base.
        """
        print(f"Processing PDFs in: {self.pdf_dir}")
        if not os.path.exists(self.pdf_dir):
            print(f"Error: PDF directory '{self.pdf_dir}' not found. Please create it and place your PDFs inside.")
            return

        pdf_files_found = False
        for filename in os.listdir(self.pdf_dir):
            if filename.lower().endswith(".pdf"):
                pdf_files_found = True
                filepath = os.path.join(self.pdf_dir, filename)
                try:
                    reader = PdfReader(filepath)
                    full_text = ""
                    # Extract text page by page
                    for page in reader.pages:
                        # Use .extract_text() which returns string or None
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + "\n" # Add newline between pages for better readability

                    # Chunk the extracted text using the utility function
                    # Adjust max_tokens here if you want smaller/larger initial chunks for retrieval
                    chunks = chunk_text(full_text, max_tokens=1500)
                    for i, chunk in enumerate(chunks):
                        self.knowledge_base.append({
                            'filename': filename,
                            'text': chunk
                        })
                    print(f"  - Successfully processed '{filename}'. Chunks created: {len(chunks)}")
                except Exception as e:
                    print(f"  - Error processing '{filename}': {e}")

        if not pdf_files_found:
            print(f"No PDF files found in '{self.pdf_dir}'. Please ensure PDFs are present.")
        print(f"Total chunks in knowledge base: {len(self.knowledge_base)}")


    def get_relevant_chunks(self, query: str, top_n: int = 3) -> List[str]:
        """
        Retrieves text chunks from the knowledge base that are most relevant to the query.
        This implementation uses a simple keyword-based scoring.
        For significantly better results, especially with complex queries,
        consider integrating vector embeddings and semantic search (e.g., using a vector database).
        """
        query_words = query.lower().split()
        scored_chunks = []

        for chunk_data in self.knowledge_base:
            chunk_text_lower = chunk_data['text'].lower()
            # Calculate a simple score based on how many query words are in the chunk
            score = sum(1 for word in query_words if word in chunk_text_lower)
            if score > 0: # Only consider chunks that contain at least one query word
                scored_chunks.append((score, chunk_data['text']))

        # Sort chunks by score in descending order and return the top_n
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_n]]