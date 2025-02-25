import os
import requests
from flask import Flask, request, jsonify
from scraping_tools import fetch_resolution_data
from llm_tools import get_ai_resolution

RESOLUTION_API_URL = os.getenv("RESOLUTION_API_URL", "http://127.0.0.1:4000")

app = Flask(__name__)

@app.route('/resolve-market', methods=['POST'])
def resolve_market():
    data = request.json
    market_id = data["market_id"]
    market_type = data["market_type"]
    sources = data["sources"]

    # Scrape market resolution sources
    scraped_data = fetch_resolution_data(sources)

    # Use LLM to determine correct outcome
    outcome_index = get_ai_resolution(
        market_title=data["title"],
        description=data["description"],
        resolution_criteria=data["resolution_criteria"],
        outcome_categories=data["outcome_categories"],
        evidence=scraped_data
    )

    # Send resolved outcome back to Node.js
    # response = requests.post(f"{RESOLUTION_API_URL}/market-resolution", json={
    #     "market_id": market_id,
    #     "resolution": outcome_index
    # })
    return jsonify({"market_id": market_id, "resolution": outcome_index})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

