import requests
from dotenv import load_dotenv
import yfinance as yf
import os
from transformers import pipeline

load_dotenv()


def get_ape_wisdom_data():
    r = requests.get("https://apewisdom.io/api/v1.0/filter/stocks")

    if r.status_code == 200:
        return r.json()
    else:
        print(f"Error fetching Ape Wisdom data: {r.status_code}")
        return None


def get_news_company(ticker, start_date, end_date):
    """Fetches news articles from Finnhub for a specific company within a given date range. Returns a list of news articles that has headline, summary, source, and url.

    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in the format YYYY-MM-DD
        end_date (str): End date in the format YYYY-MM-DD

    Returns:
        list: A list of news articles for the specified company and date range in this format:
        [
            {
                'category': 'company',
                'datetime': 1622505600,
                'headline': 'Apple announces new iPhone',
                'id': 123456,
                'image': 'https://example.com/image.jpg',
                'related': 'AAPL',
                'source': 'Reuters',
                'summary': 'Apple has announced the release of its new iPhone model, which features a larger display and improved battery life.',
                'url': 'https://www.reuters.com/article/us-apple-iphone-idUSKCN2D90XG'
            }
        ]
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start_date}&to={end_date}&token={api_key}"
    r = requests.get(url)

    if r.status_code == 200:
        return r.json()
    else:
        print(f"Error fetching news data: {r.status_code}")
        return None


def get_volume_data(ticker, start_date, end_date):
    """Fetches the volume of trades conducted on the specificed company and date.

    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in the format YYYY-MM-DD
        end_date (str): End date in the format YYYY-MM-DD

    Returns:
        pandas.Series: A series containing the volume of trades for the specified company and date range
    """
    ticker = yf.Ticker(ticker)
    hist = ticker.history(start=start_date, end=end_date)
    return hist["Volume"]


if __name__ == "__main__":

    finbert = pipeline(
        "text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert"
    )

    start_date = "2026-05-01"
    end_date = "2026-05-02"
    # get_ape_wisdom_data()
    print(get_news_company("AAPL", start_date, end_date))
    print(get_volume_data("AAPL", start_date, end_date))
    
    # TO DO:
    # 1. Transform the news data into a format that can be fed into the FinBERT model (e.g., extract headlines and summaries).

    texts = [
        "Apple beats earnings estimates and raises guidance.",
        "Tesla faces lawsuit over autopilot safety concerns.",
    ]

    results = finbert(texts)
    print(results)
