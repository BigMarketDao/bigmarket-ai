import os
import requests
from flask import Flask, request, jsonify
from scraping_tools import fetch_resolution_data
from llm_tools import get_ai_resolution

app = Flask(__name__)

@app.route('/resolve-market', methods=['POST'])
def resolve_market():
    data = request.json
    print(f"🔹 Received Market Resolution Request: {data}")  # 🛠 Debugging incoming request

    market_id = data["market_id"]
    market_type = data["market_type"]
    sources = data["sources"]

    # Scrape market resolution sources
    scraped_data = fetch_resolution_data(sources)
    print(f"🔹 Scraped Data: {scraped_data}")  # 🛠 Debugging scraped data

    # Use LLM to determine correct outcome
    prompt, ai_response, outcome_index, model = get_ai_resolution(
        market_title=data["title"],
        description=data["description"],
        resolution_criteria=data["resolution_criteria"],
        outcome_categories=data["outcome_categories"],
        evidence=scraped_data
    )

    print(f"✅ AI Resolved Outcome: {outcome_index}")  # 🛠 Debugging AI response

    return jsonify({
        "market_id": market_id,
        "market_type": market_type, 
        "resolution": outcome_index,
        "prompt": prompt,
        "ai_response": ai_response,
        "model": model
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
