import os
import requests
from dotenv import load_dotenv
from config.settings import NEWS_API_KEY, HF_TOKEN


def get_financial_news(query="Apple", limit=5):
    """
    Fetch relevant financial or business-related news for a company.
    """
    url = f"https://newsapi.org/v2/everything?q={query}+stock+finance&language=en&sortBy=publishedAt&pageSize={limit}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return [{"title": "Error fetching news", "description": response.text}]
    
    articles = response.json().get("articles", [])
    
    results = []
    for article in articles:
        desc = article.get("description", "") or ""
        if not desc.strip():  # skip empty ones
            continue
        
        results.append({
            "title": article["title"],
            "source": article["source"]["name"],
            "description": desc,
            "url": article["url"]
        })
    
    return results
