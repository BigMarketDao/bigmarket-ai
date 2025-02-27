import os
import requests
from langchain_community.llms import OpenAI

# Load environment variables
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # Default to OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def get_llm_response(prompt):
    """Selects LLM provider and returns the generated response."""
    if LLM_PROVIDER.lower() == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise ValueError("DeepSeek API key is missing.")

        try:
            response = requests.post(
                "https://api.deepseek.com/beta/completions",
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "deepseek-chat",
                    "prompt": prompt,
                    "temperature": 0,
                    "max_tokens": 5,  # Restrict output to a short number
                    "stop": ["\n"]  # Ensure it stops after a single response
                }
            )
            response_json = response.json()  # Get JSON response
            
            # 🛠 Debugging: Print the API response
            print("🔍 DeepSeek API Response:", response_json)

            if "choices" not in response_json or not response_json["choices"]:
                raise ValueError(f"Unexpected API response: {response_json}")

            # ✅ Extract response and clean it
            ai_response = response_json["choices"][0]["text"].strip()

            if not ai_response.isdigit():
                raise ValueError(f"Invalid response format: {result_text}")

            return ai_response

        except Exception as e:
            print(f"🚨 Error calling DeepSeek API: {e}")
            return "ERROR: DeepSeek API failed"

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

    print("🔍 Sent Prompt to DeepSeek:\n", prompt)  # 🛠 Debugging the prompt

    ai_response = get_llm_response(prompt)
    try:
        outcome_index = int(ai_response)
    except ValueError:
        outcome_index = -1  # If the AI response is invalid, return -1

    return prompt, ai_response, outcome_index, "deepseek-chat"
