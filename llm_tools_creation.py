import os
import json
import re
import requests
from datetime import datetime, timedelta
from langchain_community.llms import OpenAI
from decimal import Decimal
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# Load environment variables
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # Default to OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

MARKET_CREATION_PROMPT_NEWS = """
You are an AI that generates structured event-based prediction markets based on a user's speculative idea.
Your response must be in **JSON format**.

### **User's Input:**
{user_idea}

### **Live News Context (Scraped from Reputable Sources):**
{news_summary}

### **Generate the following structured event market details:**
1. **title** → A short and engaging title for the market.
2. **description** → A 2-3 sentence explanation of what the market is about.
3. **outcome_categories** → Generate **between 2 and 6 possible outcomes**, directly based on recent news developments.
4. **market_sector** → Select one from: `["politics", "sports", "economy", "climate", "technology"]`.
5. **resolution_criteria** → Define **how the market is resolved** using reputable news sources.
6. **earliest_resolution_date** → **Set a reasonable future date** (e.g., 3-7 days for short-term events).
7. **sources** → List of **trusted news sources** where users can verify the outcome.

### **Response format (JSON):**
{{
    "title": "...",
    "description": "...",
    "outcome_categories": ["...", "...", "..."],
    "market_sector": "...",
    "resolution_criteria": "...",
    "earliest_resolution_date": "...",
    "sources": ["...", "..."]
}}

DO NOT include explanations—just return a valid JSON object.
"""

MARKET_CREATION_PROMPT_FINANCIAL = """
You are an AI that generates structured financial prediction markets based on a user's speculative idea.
Your response must be in **JSON format**.

### **User's Input:**
{user_idea}

### **Live Asset Pricing:**
- **Asset Name**: {asset}
- **Current Price**: ${current_price}
- **Price Volatility Context**: Expected short-term movement of {volatility_range}%.

### **Generate the following structured financial market details:**
1. **title** → A short and engaging title for the market.
2. **description** → A 2-3 sentence explanation of what the market is about.
3. **outcome_categories** → Generate **realistic price brackets** based on asset history and short-term volatility.
4. **market_sector** → Set to `"crypto"`, `"stocks"`, or `"commodities"` based on the asset type.
5. **resolution_criteria** → Clearly define **how the price is measured** (e.g., CoinGecko closing price at 23:59 UTC).
6. **earliest_resolution_date** → **Must be at least 3 days in the future**.
7. **sources** → A list of **trusted financial data sources**.

### **Response format (JSON):**
{{
    "title": "...",
    "description": "...",
    "outcome_categories": ["...", "...", "..."],
    "market_sector": "...",
    "resolution_criteria": "...",
    "earliest_resolution_date": "...",
    "sources": ["...", "..."]
}}

DO NOT include explanations—just return a valid JSON object.
"""

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
        print("🔍 DeepSeek API Response:", response_json)
        return response_json["choices"][0]["text"].strip()

    elif LLM_PROVIDER.lower() == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is missing.")

        llm = OpenAI(api_key=OPENAI_API_KEY)
        return llm.invoke(prompt)

    else:
        raise ValueError("Invalid LLM_PROVIDER. Use 'openai' or 'deepseek'.")

def get_asset_price(asset):
    """Fetch the current price of an asset from CoinGecko and return a properly formatted price."""
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={asset}&vs_currencies=usd"
    try:
        response = requests.get(url)
        data = response.json()
        price = data.get(asset, {}).get("usd", None)

        if price is not None:
            return Decimal(price)  # Ensure a properly formatted decimal value
        return None
    except Exception as e:
        print(f"❌ Error fetching asset price: {e}")
        return None  # Return None if price lookup fails

def get_realistic_price_brackets(asset, current_price, resolution_timeframe):
    """Generate more realistic price brackets based on the asset's price and resolution timeframe."""
    if current_price is None:
        return ["Below $0.50", "$0.50 - $1", "$1 - $2", "$2 - $3", "Above $3"]  # Default fallback

    current_price = Decimal(current_price)

    # Set expected short-term volatility based on timeframe
    if resolution_timeframe in ["1 day", "2 days"]:
        volatility = Decimal("0.05")  # 5% expected short-term movement
    elif resolution_timeframe in ["1 week"]:
        volatility = Decimal("0.10")  # 10% movement
    elif resolution_timeframe in ["1 month"]:
        volatility = Decimal("0.20")  # 20% movement
    else:
        volatility = Decimal("0.30")  # 30% movement for long-term predictions

    lower_bound = current_price * (1 - volatility)
    upper_bound = current_price * (1 + volatility)

    # Generate reasonable price brackets
    brackets = [
        f"Below ${lower_bound.quantize(Decimal('0.01'))}",
        f"${lower_bound.quantize(Decimal('0.01'))} - ${current_price.quantize(Decimal('0.01'))}",
        f"${current_price.quantize(Decimal('0.01'))} - ${upper_bound.quantize(Decimal('0.01'))}",
        f"Above ${upper_bound.quantize(Decimal('0.01'))}"
    ]

    return brackets

def extract_asset(user_idea):
    """Extract asset name from a user's idea about financial markets."""
    user_idea = user_idea.lower()
    
    # Look for known cryptocurrency keywords
    known_assets = ["bitcoin", "ethereum", "stacks", "solana", "cardano", "ripple"]
    for asset in known_assets:
        if asset in user_idea:
            return asset  # Return the matched asset
    
    # Extract words after "price of" or "value of"
    if "price of" in user_idea or "value of" in user_idea:
        words = user_idea.split()
        index = words.index("price") if "price" in words else words.index("value")
        if index + 1 < len(words):
            return words[index + 1]  # Return the next word as asset name

    return None  # Return None if no asset is detected

def scrape_latest_news(user_idea):
    """Scrapes recent news articles related to the market topic."""
    search_query = user_idea.replace(" ", "+")  # Convert user input to a search-friendly format
    news_url = f"https://news.google.com/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        response = requests.get(news_url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        headlines = soup.find_all("h3", limit=5)  # Get the top 5 news headlines
        
        news_summary = []
        for headline in headlines:
            title = headline.get_text()
            link = "https://news.google.com" + headline.find("a")["href"][1:]  # Build full URL
            news_summary.append(f"{title} ({link})")

        return "\n".join(news_summary) if news_summary else "No relevant news found."

    except Exception as e:
        print(f"❌ Error scraping news: {e}")
        return "No relevant news available."

def ensure_future_date():
    """Ensures the market resolution date is in the future (at least 3 days ahead)."""
    today = datetime.today()
    min_future_date = today + timedelta(days=3)  # Ensure at least 3 days ahead
    return min_future_date.strftime("%Y-%m-%d")

def generate_market(user_idea, market_type):
    """Calls LLM to generate a market structure based on whether it's financial or news-based."""

    if market_type == "financial":
        asset = extract_asset(user_idea)
        current_price = get_asset_price(asset) if asset else None
        volatility_range = "10-20"  # Default expected short-term volatility in %

        print(f"🔍 Detected Asset: {asset}, Current Price: ${current_price}")

        prompt = MARKET_CREATION_PROMPT_FINANCIAL.format(
            user_idea=user_idea,
            asset=asset if asset else "unknown",
            current_price=current_price if current_price else "unknown",
            volatility_range=volatility_range
        )

    elif market_type == "news":
        news_summary = scrape_latest_news(user_idea)
        print(f"🔍 Scraped News Summary:\n{news_summary}")

        prompt = MARKET_CREATION_PROMPT_NEWS.format(
            user_idea=user_idea,
            news_summary=news_summary
        )

    else:
        return {"error": "Invalid market type"}

    ai_response = get_llm_response(prompt)

    print(f"🔍 Raw AI Response:\n{ai_response}")

    ai_response_cleaned = re.sub(r"```json\n(.*?)\n```", r"\1", ai_response, flags=re.DOTALL)

    print(f"🔍 Cleaned AI Response:\n{ai_response_cleaned}")

    try:
        market_data = json.loads(ai_response_cleaned)
        return market_data
    except json.JSONDecodeError:
        print(f"❌ Error parsing LLM response: {ai_response_cleaned}")
        return {"error": "Invalid response format"}
