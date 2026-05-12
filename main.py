import requests
from dotenv import load_dotenv
import yfinance as yf
import os
from transformers import pipeline
from datetime import datetime
import pandas as pd

load_dotenv()

# Extraction 

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

# Transformation


def transform_news_data(news_data):

    transformed = []

    for article in news_data:

        headline = article.get("headline", "")
        summary = article.get("summary", "")

        # combine text
        text = f"{headline}. {summary}"

        # cleaning
        text = text.replace("\xa0", " ")
        text = text.replace("&#39;", "'")
        text = " ".join(text.split())
        text = text.encode("utf-8", "ignore").decode("utf-8")

        # convert timestamp
        date = datetime.fromtimestamp(
            article["datetime"]
        ).strftime("%Y-%m-%d")

        transformed.append({
            "ticker": article.get("related"),
            "headline": headline,
            "summary": summary,
            "text": text,
            "source": article.get("source"),
            "date": date
        })

    return transformed


   # main pipeline


if __name__ == "__main__":

    
    #  Load model
   
    finbert = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert"
    )

    
     # Parameters
   
    start_date = "2026-05-01"
    end_date = "2026-05-02"


    # EXTRACT
  
    news = get_news_company("AAPL", start_date, end_date)

    
    #  TRANSFORM
    
    transformed_news = transform_news_data(news)

    texts = [article["text"] for article in transformed_news]

    
    # SENTIMENT ANALYSIS
    
    results = finbert(texts)

 
    # BUILD FINAL DATASET
    
    final_dataset = []

    for article, sentiment in zip(transformed_news, results):

        article["sentiment"] = sentiment["label"]
        article["confidence"] = sentiment["score"]

        final_dataset.append(article)

    
    # OUTPUT
 
    df = pd.DataFrame(final_dataset)

    for row in final_dataset:
        print("\n-------------------")
        print(f"Headline: {row['headline']}")
        print(f"Sentiment: {row['sentiment']}")
        print(f"Confidence: {row['confidence']:.2f}")
        print(f"Date: {row['date']}")

       
#  EDA statistics 


df = pd.DataFrame(final_dataset)

print("\n=========================")
print("DESCRIPTIVE EDA")
print("=========================")

# 1. Basic info
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# 2. Sentiment distribution
print("\nSentiment distribution:\n", df["sentiment"].value_counts())

# 3. Source distribution (bias indicator)
print("\nSource distribution:\n", df["source"].value_counts())

# 4. Confidence summary
print("\nConfidence stats:\n", df["confidence"].describe())

# 5. Sentiment over time
print("\nSentiment by date:\n")
print(df.groupby("date")["sentiment"].value_counts())

# 6. Average confidence per sentiment
print("\nAvg confidence by sentiment:\n")
print(df.groupby("sentiment")["confidence"].mean())


#EDA VISUALIZATION


# BIAS EXPLORATION

print("\n=========================")
print("BIAS ANALYSIS")
print("=========================")

# Bias check 1: source imbalance
source_counts = df["source"].value_counts()
print("\nSource imbalance:\n", source_counts)

# Bias check 2: sentiment skew
sentiment_counts = df["sentiment"].value_counts(normalize=True)
print("\nSentiment proportions:\n", sentiment_counts)

# Bias check 3: missing data patterns
print("\nMissing data pattern:\n", df.isna().mean())

# Bias check 4: sentiment by source
print("\nSentiment by source:\n")
print(df.groupby("source")["sentiment"].value_counts())

# Bias check 5: potential overconfidence
high_confidence = df[df["confidence"] > 0.90]
print("\nHigh confidence articles (>0.90):", len(high_confidence))