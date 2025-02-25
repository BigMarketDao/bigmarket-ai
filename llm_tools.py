import os
from langchain_community.llms import OpenAI
import requests

# Load environment variables
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # Default to OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def get_llm_response(prompt):
    """Selects LLM provider and returns the generated response."""
    if LLM_PROVIDER.lower() == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise ValueError("DeepSeek API key is missing.")

        response = requests.post(
            "https://api.deepseek.com/v1/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat", "prompt": prompt, "temperature": 0}
        )
        return response.json()["choices"][0]["text"].strip()

    elif LLM_PROVIDER.lower() == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is missing.")

        llm = OpenAI(api_key=OPENAI_API_KEY)
        return llm.invoke(prompt)

    else:
        raise ValueError("Invalid LLM_PROVIDER. Use 'openai' or 'deepseek'.")

def get_ai_resolution(market_title, description, resolution_criteria, outcome_categories, evidence):
    """Uses an LLM (OpenAI or DeepSeek) to determine the correct market outcome."""
    prompt = f"""
    You are an AI resolving a prediction market.

    Market: {market_title}
    Description: {description}
    Resolution Criteria: {resolution_criteria}
    Outcomes: {', '.join(outcome_categories)}
    Evidence: {evidence}

    Which outcome best matches the resolution criteria? Respond with the **index number** of the most suitable outcome.
    """

    return get_llm_response(prompt)
