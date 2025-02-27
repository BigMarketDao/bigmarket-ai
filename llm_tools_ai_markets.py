import os
import json
import requests
import re
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta

# Load OpenAI or DeepSeek API keys
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def get_current_utc_date():
    """Returns the current date in UTC format YYYY-MM-DD."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

MARKET_DISCOVERY_PROMPT = """
You are an AI specializing in identifying prediction market opportunities from real-world news.
Given the following news story, suggest 1 **highly actionable prediction market**.

### **Current UTC Date:**
{current_date}

### **News Story:**
{news_story}

### **Instructions:**
1. Extract speculative or uncertain aspects of the story.
2. Convert them into a **prediction market question**.
3. **The market MUST be resolvable between 2 and 30 days from today ({current_date}).**
4. If no market fits this requirement, **respond with an error message instead of guessing**.
5. Ensure the market has:
   - A title
   - A clear description
   - 2-10 possible outcomes
   - An objective resolution criteria
   - A valid resolution date **between {min_resolution_date} and {max_resolution_date}**
   - Reliable news sources for verification.

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

def ensure_valid_resolution_date(date_str):
    """Ensures the resolution date is between 2 and 30 days in the future."""
    today = datetime.now(timezone.utc)
    min_future_date = today + timedelta(days=2)  # Min: 2 days ahead
    max_future_date = today + timedelta(days=30) # Max: 30 days ahead

    try:
        resolution_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if resolution_date < min_future_date or resolution_date > max_future_date:
            print(f"⚠️ Invalid date {date_str}. Adjusting to within allowed range.")
            resolution_date = min_future_date  # Default to minimum allowed
    except ValueError:
        resolution_date = min_future_date  # Default if parsing fails

    return resolution_date.strftime("%Y-%m-%d")

def discover_markets_from_news(news_url):
    """Generates market ideas based on a news story and ensures resolution dates are in the future."""
    news_story = scrape_news_summary(news_url)
    print(f"🔍 Scraped News Summary:\n{news_story}")

    current_date = get_current_utc_date()
    min_resolution_date = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
    max_resolution_date = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")

    prompt = MARKET_DISCOVERY_PROMPT.format(
        news_story=news_story,
        current_date=current_date,
        min_resolution_date=min_resolution_date,
        max_resolution_date=max_resolution_date
    )

    ai_response = get_llm_response(prompt)

    print(f"🔍 Raw AI Response:\n{ai_response}")

    # ✅ Step 1: Remove triple backticks if present
    ai_response_cleaned = re.sub(r"```json\n(.*?)\n```", r"\1", ai_response, flags=re.DOTALL)

    print(f"🔍 Cleaned AI Response:\n{ai_response_cleaned}")

    # ✅ Step 2: Try parsing the JSON
    try:
        market_data = json.loads(ai_response_cleaned)  # Convert string to JSON

        # ✅ Step 3: Fix resolution dates
        for market in market_data.get("markets", []):
            if "earliest_resolution_date" in market:
                market["earliest_resolution_date"] = ensure_valid_resolution_date(market["earliest_resolution_date"])

        return market_data
    except json.JSONDecodeError:
        print(f"❌ Error parsing LLM response: {ai_response_cleaned}")
        return {"error": "Invalid response format"}
