import requests
from dotenv import load_dotenv
import yfinance as yf
import os

load_dotenv()

# Ape Wisdom API
def get_ape_wisdom_data():
    r = requests.get("https://apewisdom.io/api/v1.0/filter/stocks")

    if r.status_code == 200:
        print(r.json())
        

def get_news_company(ticker):
    api_key = os.getenv("FINNHUB_API_KEY")
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2025-05-01&to=2026-05-01&token={api_key}"
    r = requests.get(url)

    if r.status_code == 200:
        print(r.json())

def get_volume_data(ticker):
    data = yf.download(ticker, start="2026-05-01", end="2026-05-01")
    print(data['Volume'])
    
    
    
if __name__ == "__main__":
    # get_ape_wisdom_data()
    # get_news_company("AAPL")
    get_volume_data("AAPL")


