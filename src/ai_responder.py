import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import List
from google.api_core import exceptions # For more specific error handling

# Note: The count_tokens function in src/utils.py uses tiktoken and is primarily for
# initial PDF chunking estimation. For actual Gemini prompt token counting,
# we'll use genai.GenerativeModel.count_tokens directly.
# So, no change needed in utils.py's count_tokens definition for this switch.

class AIResponder:
    def __init__(self):
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file. Please get one from Google AI Studio.")

        # Configure the Google Generative AI client
        genai.configure(api_key=gemini_api_key)

        # Choose your Gemini model. 'gemini-pro' is a good general-purpose model.
        # For more advanced capabilities (e.g., larger context), consider 'gemini-1.5-pro' or 'gemini-1.5-flash'
        # if your API key has access and pricing is acceptable.
        self.model = genai.GenerativeModel('gemini-1.5-flash-8b')
        # You might need to check if the model is available for your region/account.
        # print(f"Using Gemini model: {self.model.model_name}")

    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """
        Generates an answer using the Google Gemini API based on the question
        and provided context chunks.
        """
        if not context_chunks:
            return "I don't have enough information in the provided documents to answer that question."

        context = "\n\n".join(context_chunks)

        # Gemini typically handles context window internally, but we can check token count
        # to prevent sending excessively large prompts if desired, or for warning the user.
        # Note: Gemini's token counting works differently than OpenAI's.
        # It's better to structure the prompt as messages for more accurate counting.

        # Construct messages for Gemini. Gemini models typically prefer a direct
        # user-system turn structure. We embed the context in the user message.
        messages = [
            {"role": "user", "parts": [f"Based on the following document excerpts, answer the question:\n\nDocument Excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer:"]}
        ]

        # Use the model's count_tokens method for Gemini-specific token estimation
        try:
            # Check estimated token count for the prompt.
            # This is an estimation and might differ slightly from actual usage depending on model specifics.
            estimated_token_count = self.model.count_tokens(messages).total_tokens
            # Gemini models usually have large context windows (e.g., gemini-pro is 30,720 tokens,
            # gemini-1.5-flash/pro are much larger). You might not need explicit truncation for most PDFs.
            # However, if your PDFs are extremely large, you might still hit limits.
            # For simplicity, we'll rely on the API to handle truncation or errors for very large inputs.
            # The max_context_tokens from the previous version might become less relevant.
            if estimated_token_count > 25000: # A high threshold for warning, given Gemini's context
                print(f"Warning: Estimated prompt tokens ({estimated_token_count}) are very high.")

        except Exception as e:
            print(f"Warning: Could not estimate token count for Gemini. Error: {e}")


        try:
            # Generate content using the Gemini model
            response = self.model.generate_content(
                messages,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500 # Max tokens for the assistant's response
                )
            )
            # Access the text from the response
            return response.text.strip()
        except exceptions.ResourceExhausted as e:
            print(f"Gemini API Error: Resource Exhausted (Quota issue). Please check your Google Cloud Console billing and API quotas: {e}")
            return "An error occurred due to quota limits. Please check your Gemini API usage."
        except exceptions.FailedPrecondition as e:
            print(f"Gemini API Error: Failed Precondition. This often means content safety violation or invalid input. Error: {e}")
            return "The request was blocked due to content safety or invalid input. Please try rephrasing your question."
        except exceptions.InvalidArgument as e:
            print(f"Gemini API Error: Invalid Argument. Likely an issue with the model name or request format. Error: {e}")
            return "An error occurred with the request format. Please check the model name or input."
        except Exception as e:
            print(f"An unexpected error occurred with Gemini API: {e}")
            return "An unexpected error occurred."

    # The _truncate_context method is less critical with Gemini's larger context windows,
    # but could still be useful for very large inputs if you hit limits or want to optimize.
    # For now, we'll keep it but primarily rely on Gemini's internal handling and a warning.
    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """Trims the context to fit within the token limit (approximation for Gemini)."""
        # This is a very basic character-based truncation.
        # For Gemini, it's better to rely on its API's token counting and context window management.
        # If you still need to pre-truncate before sending to Gemini, you'd use a more sophisticated method.
        # For now, it serves as a conceptual placeholder if the main count_tokens logic is removed.
        if len(context) > max_tokens * 4: # Rough char-to-token ratio
            return context[:max_tokens * 4] + "..."
        return context