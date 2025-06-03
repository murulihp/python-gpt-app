# No longer importing tiktoken as it's not needed for Gemini or simple chunking
# import tiktoken

def count_tokens(text: str, model_name: str = "placeholder") -> int:
    """
    Counts the approximate number of tokens in a text string.
    Since tiktoken is removed for Gemini, this provides a character-based approximation.
    Rough estimate: 1 token is approximately 4 characters for English text.
    """
    # This is a very rough approximation. For more precision,
    # you'd need a specific tokenizer or a more advanced general method.
    return len(text) // 4 # Divide by 4 to approximate tokens

def chunk_text(text: str, max_tokens: int = 1000) -> list[str]:
    """
    Splits a large text into smaller chunks based on approximate token count.
    Aims to split at natural breakpoints like sentences to maintain coherence.
    """
    # Replace newlines with spaces to treat paragraphs as continuous text for sentence splitting
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    for sentence in sentences:
        # Add back the period and a space for more accurate sentence token counting
        sentence_to_add = sentence.strip()
        if not sentence_to_add: # Skip empty sentences
            continue

        # Use the simplified count_tokens based on character length
        sentence_tokens = count_tokens(sentence_to_add + ". ")

        # If adding the current sentence keeps the chunk within max_tokens, add it
        if current_chunk_tokens + sentence_tokens < max_tokens:
            current_chunk.append(sentence_to_add)
            current_chunk_tokens += sentence_tokens
        else:
            # If the current chunk is not empty, save it before starting a new one
            if current_chunk:
                chunks.append(". ".join(current_chunk) + ".")
            # Start a new chunk with the current sentence
            current_chunk = [sentence_to_add]
            current_chunk_tokens = sentence_tokens

    # Add any remaining content as the last chunk
    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks