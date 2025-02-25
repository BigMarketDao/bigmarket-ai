import requests
from bs4 import BeautifulSoup

def fetch_resolution_data(urls):
    """
    Scrapes resolution sources for market data.
    """
    scraped_data = []
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
            soup = BeautifulSoup(response.text, 'html.parser')
            text_content = soup.get_text()[:2000]  # Limit text to 2000 characters
            scraped_data.append(text_content)
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    return " ".join(scraped_data)  # Return all scraped data as a single string

