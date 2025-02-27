import os
import json
import requests
import re
from bs4 import BeautifulSoup

# Load OpenAI or DeepSeek API keys
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

MARKET_DISCOVERY_PROMPT = """
You are an AI specializing in identifying prediction market opportunities from real-world news.
Given the following news story, suggest one **highly actionable prediction market**.

### **News Story:**
{news_story}

### **Instructions:**
1. Extract speculative or uncertain aspects of the story.
2. Convert them into a **prediction market question**.
3. Ensure the market can be **objectively resolved within 30-90 days**.
4. **The resolution date MUST be in the future.** If the news event has already happened, adjust the resolution date to be a future speculation on the topic.
4. Suggest **reliable sources** where the outcome can be verified.

### **Response format (JSON):**
{{
    "markets": [
        {{
            "title": "...",
            "description": "...",
            "outcome_categories": ["...", "...", "..."],
            "market_sector": "...",
            "resolution_criteria": "...",
            "earliest_resolution_date": "...",
            "sources": ["...", "..."]
        }},
        ...
    ]
}}

DO NOT include explanations—just return valid JSON.
"""

def scrape_news_summary(news_url):
    """Scrapes the article summary from a news URL."""
    try:
        response = requests.get(news_url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title & first paragraph (or meta description)
        title = soup.find("title").get_text()
        paragraphs = soup.find_all("p", limit=3)
        summary = " ".join([p.get_text() for p in paragraphs])

        return f"**{title}**\n{summary}" if summary else "No summary available."
    except Exception as e:
        print(f"❌ Error scraping news: {e}")
        return "No summary available."

def get_llm_response(prompt):
    """Selects LLM provider and returns structured JSON response."""
    if LLM_PROVIDER.lower() == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise ValueError("DeepSeek API key is missing.")

        response = requests.post(
            "https://api.deepseek.com/beta/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat", "prompt": prompt, "temperature": 0}
        )
        response_json = response.json()
        return response_json["choices"][0]["text"].strip()

    elif LLM_PROVIDER.lower() == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is missing.")

        response = requests.post(
            "https://api.openai.com/v1/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4",
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 500
            }
        )
        response_json = response.json()
        return response_json["choices"][0]["text"].strip()

    else:
        raise ValueError("Invalid LLM_PROVIDER. Use 'openai' or 'deepseek'.")

def discover_markets_from_news(news_url):
    """Generates market ideas based on a news story."""
    news_story = scrape_news_summary(news_url)
    print(f"🔍 Scraped News Summary:\n{news_story}")

    prompt = MARKET_DISCOVERY_PROMPT.format(news_story=news_story)
    ai_response = get_llm_response(prompt)

    print(f"🔍 Raw AI Response:\n{ai_response}")

    # ✅ Step 1: Remove triple backticks if present
    ai_response_cleaned = re.sub(r"```json\n(.*?)\n```", r"\1", ai_response, flags=re.DOTALL)

    # 🛠 Debugging: Print cleaned response
    print(f"🔍 Cleaned AI Response:\n{ai_response_cleaned}")

    # ✅ Step 2: Try parsing the JSON
    try:
        market_data = json.loads(ai_response_cleaned)  # Convert string to JSON
        return market_data
    except json.JSONDecodeError:
        print(f"❌ Error parsing LLM response: {ai_response_cleaned}")
        return {"error": "Invalid response format"}
