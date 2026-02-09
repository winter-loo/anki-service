import os
from google import genai
import json
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_word_explanation(word: str, model: str = "gemini-2.5-flash-lite"):
    prompt = f"""
    You are an advanced English dictionary and linguistic expert.
    Explain the word: "{word}".

    Return the response ONLY in raw JSON format (no markdown code blocks) with the following structure:
    {{
        "word": "{word}",
        "ipa_pronunciation": "International Phonetic Alphabet representation",
        "part_of_speech": "noun/verb/adjective etc.",
        "simple_definition": "A clear, concise definition for a general audience.",
        "detailed_explanation": "A deeper look into the nuance and usage.",
        "in_chinese": "中文释义"
        "etymology": "Brief origin of the word.",
        "examples": ["Sentence 1", "Sentence 2", "Sentence 3"],
        "synonyms": ["synonym1", "synonym2"],
        "antonyms": ["antonym1", "antonym2"],
        "context_usage": "When to use this word (formal, casual, slang, archaic)."
    }}
    """

    try:
        with genai.Client(api_key=GENAI_API_KEY) as client:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            # Clean up potential markdown formatting if the model adds ```json
            cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_text)
    except Exception as e:
        return {"error": str(e)}
