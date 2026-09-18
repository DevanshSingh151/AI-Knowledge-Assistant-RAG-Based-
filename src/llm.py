import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

def get_llm():
    """Returns the Gemini LLM instance."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
    # Primary model
    primary_llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.0,
        google_api_key=api_key,
        max_output_tokens=1024,
        max_retries=1,
        timeout=15
    )
    
    # Fallback models in case of 429/503 errors (quota/heavy load)
    fallback_llms = [
        ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0, google_api_key=api_key, max_retries=1, timeout=15),
        ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key=api_key, max_retries=1, timeout=15),
        ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0.0, google_api_key=api_key, max_retries=1, timeout=15)
    ]
    
    return primary_llm.with_fallbacks(fallback_llms)
