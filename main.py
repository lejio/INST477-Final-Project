import requests
from dotenv import load_dotenv
import yfinance as yf
import os
from transformers import pipeline
from datetime import datetime, timedelta
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

load_dotenv()

# Extraction 

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

def get_news_range(symbol, start_date, end_date):
    """Get the news based on the range of the dates. Increments a day at a time to bypass return limits. Finnhub could only return a limited amount of articles at a time. Therefore we chunked it up to get more dates.

    Args:
        symbol (str): The symbol you are trying to fetch
        start_date (str): Start date in %Y-%m-%d
        end_date (str): End date in %Y-%m-%d

    Returns:
        [dict]: List of all the article metadata
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    all_news = []
    cur = start
    print("Begin fetching news articles:")
    while cur < end:
        day_str = cur.strftime("%Y-%m-%d")
        chunk = get_news_company(symbol, day_str, day_str) or []
        all_news.extend(chunk)
        cur += timedelta(days=1)
        time.sleep(0.05)
        print("Day:", cur)
        
    return all_news

def data_transformation_engine(news_raw, volume):
    """Combines both yFinance volume data and Finnhub article data.
    Finnhub articles uses unix time, which needs to be converted to string time.
    Also converts the Series integer data structure in pandas to python primitive int.
    Iterates through both news and volume data inputs and combines them into a single dataset.

    Args:
        news_raw (list(dict)): A giant list of dictionaries containing metadata information for articles release
        volume (pandas.Series): A pandas series of dates and the traded volumes

    Returns:
        _type_:
        
        type dataset:
            key: date strings in the format of "%Y-%m-%d"
            value: Day
            
        type Day:
            'articles': [Article] --> list of articles
            'volume': int
            
        type Article:
            "ticker": str,
            "headline": str,
            "summary": str,
            "text": str,
            "source": str,
            "date": datetime
    """
    dataset = {}
    news = transform_news_data(news_raw)
    
    vol_by_date = {}
    for ts, vol in volume.items():
        date_str = ts.strftime("%Y-%m-%d")
        vol_by_date[date_str] = int(vol)

    for article in news:
        date = article['date']
        if date not in vol_by_date:
            continue
        if date not in dataset:
            dataset[date] = {
                'articles': [article],
                'volume': vol_by_date[date],
            }
        else:
            dataset[date]['articles'].append(article)
    
    # print("Verification:")
    # for d in dataset:
    #     print(d, len(dataset[d]['articles']), dataset[d]['volume'])
        
    return dataset
        

def sentimental_analysis_engine(sen_engine, dataset):

    for day in dataset:
        txt_buffer = [txt['text'] for txt in dataset[day]['articles']]
        dataset[day]['sentiment_results'] = sen_engine(txt_buffer)
        

def multiday_financial_news_sentiment_pipeline(symbol, start_date, end_date, sen_engine):
    """Multiday news sentiment engine. Uses FinBERT as the sentimental analysis. Obtains daily volume data from yfinance and processed news data from Finnhub.

    Args:
        symbol (str): The symbol you are trying to fetch
        start_date (str): Start date in %Y-%m-%d
        end_date (str): End date in %Y-%m-%d
        
    Returns:
        variable: transformed_dataset
    
        type dict Transformed_Dataset:
            key: date strings in the format of "%Y-%m-%d"
            value: type Day
            
        variable: final_dataset:
        
        type list Final_Dataset:
            items: type Article
            
        type dict Day:
            'articles': [Article] --> list of articles
            'volume': int
            
        type dict Article:
            "ticker": str,
            "headline": str,
            "summary": str,
            "text": str,
            "source": str,
            "date": datetime,
            "sentiment": int,
            "sen_confidence": int
            
        variable:
            
    """
    # Extraction
    news_raw = get_news_range(symbol, start_date, end_date)
    vol = get_volume_data(symbol, start_date, end_date)
    
    # Transformation
    transformed_dataset = data_transformation_engine(news_raw, vol)
    
    # Load
    sentimental_analysis_engine(sen_engine, transformed_dataset)
    
    # Build final dataset
    # print(clean_data.keys()) --> Days
    # print(clean_data['2026-05-01'].keys()) --> {'articles', 'volume', 'sentiment_results'}

    final_dataset = []
    
    for day in transformed_dataset:
        for article, sentiment in zip(transformed_dataset[day]['articles'], transformed_dataset[day]['sentiment_results']):

            article["sentiment"] = sentiment["label"]
            article["sen_confidence"] = sentiment["score"]

            final_dataset.append(article)
            # print(article.keys()) --> {'ticker', 'headline', 'summary', 'text', 'source', 'date', 'sentiment', 'sen_confidence'}
            
    
    
    return [final_dataset, transformed_dataset]


if __name__ == "__main__":

    
    #  Load finbert model
    finbert = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert"
    )


    
     # Set start and end dates, in our case we will use May 01 - May 02
   
    start_date = "2026-05-01"
    end_date = "2026-05-08"
    symbol = "AAPL"


    # Obtain news regarding a specific symbol (in our case it will be Apple)
  
    [final_dataset, results] = multiday_financial_news_sentiment_pipeline(symbol, start_date, end_date, finbert)
    
    # OUTPUT
 
    df = pd.DataFrame(final_dataset)
    
    df["date"] = pd.to_datetime(df["date"])


    for row in final_dataset:
        print("\n-------------------")
        print(f"Headline: {row['headline']}")
        print(f"Sentiment: {row['sentiment']}")
        print(f"Confidence: {row['sen_confidence']:.2f}")
        print(f"Date: {row['date']}")

       
    #  EDA statistics

    print("\n=========================")
    print("DESCRIPTIVE EDA")
    print("=========================")

    # 1. Basic info
    print("\nDataset shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())

    # 2. Ticker distribution
    print("\nTicker distribution:\n", df["ticker"].value_counts())

    # 3. Sentiment distribution
    print("\nSentiment distribution:\n", df["sentiment"].value_counts())

    # 4. Source distribution (bias indicator)
    print("\nSource distribution:\n", df["source"].value_counts())

    # 5. Confidence summary
    # 50% is the median value
    print("\nConfidence stats:\n", df["sen_confidence"].describe())

    # 6. Sentiment over time
    print("\nSentiment by date:")
    print(df.groupby("date")["sentiment"].value_counts())

    # 7. Average confidence per sentiment
    print("\nAvg confidence by sentiment:")
    print(df.groupby("sentiment")["sen_confidence"].mean())

    # 8. Sentiment distribution
    print("\nSentiment distribution:\n", df["sentiment"].value_counts())

    # 9. Text columns info

    # Headline
    # Usually a short, concise news title
    # Typically 5-20 words 
    # Designed to summarize the main event quickly

    # Summary
    # A longer paragraph-style description of the article
    # Typically 20-100+ words
    # Provides key financial/business events and more context

    # Text
    # Headline + summary concatenated (combined)
    # At least as long as the headline
    # Usually longer than just the summary
    # Used as input for the FinBERT sentiment model
    

    text_columns = ["headline", "summary", "text"]

    summary_stats = []

    for col in text_columns:

        word_counts = df[col].fillna("").apply(
            lambda x: len(str(x).split())
        )

        summary_stats.append({
            "column": col,
            "min": word_counts.min(),
            "max": word_counts.max(),
            "mean": round(word_counts.mean(), 2),
            "median": word_counts.median(),
            "missing": df[col].isnull().sum()
        })

    stats_df = pd.DataFrame(summary_stats)
    print("\nStats for text columns:")
    print(stats_df)

    # 9. Date column info
    df["date"] = pd.to_datetime(df["date"])

    earliest_date = df["date"].min()
    latest_date = df["date"].max()

    print("\nDate stats:")
    print("min ", earliest_date)
    print("max ", latest_date)
    print("timespan ", latest_date - earliest_date)
    print("median ", df["date"].median())
    print("mean ", df["date"].mean())

    print("\nDate distribution:")
    print(df["date"].value_counts())

    # EDA VISUALIZATION
    
    

    # Confidence distribution histogram

    plt.figure(figsize=(8,5))

    sns.histplot(
        df["sen_confidence"],
        bins=20,
        kde=True
    )

    plt.title("Distribution of FinBERT Confidence Scores")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.show()

    # Ticker distribution bar graph

    plt.figure(figsize=(6,4))

    sns.countplot(
        data=df,
        x="ticker",
        order=df["ticker"].value_counts().index
    )

    plt.title("Ticker Distribution")
    plt.xlabel("Ticker")
    plt.ylabel("Count")
    plt.show()

    # Sentiment distribution bar graph

    plt.figure(figsize=(6,4))

    sns.countplot(
        data=df,
        x="sentiment",
        order=df["sentiment"].value_counts().index
    )

    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

    # Source distribution bar graph

    plt.figure(figsize=(10,6))

    source_counts = df["source"].value_counts().head(10) # up to 10 sources

    sns.barplot(
        x=source_counts.values,
        y=source_counts.index
    )

    plt.title("Top News Sources")
    plt.xlabel("Article Count")
    plt.ylabel("Source")

    plt.show()

    # Word count histograms for text columns

    df["headline_word_count"] = df["headline"].fillna("").apply(
        lambda x: len(str(x).split())
    )

    df["summary_word_count"] = df["summary"].fillna("").apply(
        lambda x: len(str(x).split())
    )

    df["text_word_count"] = df["text"].fillna("").apply(
        lambda x: len(str(x).split())
    )

    plt.figure(figsize=(8,5))

    sns.histplot(
        df["headline_word_count"],
        bins=15,
        kde=True
    )

    plt.title("Headline Word Count Distribution")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(8,5))

    sns.histplot(
        df["summary_word_count"],
        bins=20,
        kde=True
    )

    plt.title("Summary Word Count Distribution")
    plt.xlabel("Words")
    plt.ylabel("Frequency")

    plt.show()

    plt.figure(figsize=(8,5))

    sns.histplot(
        df["text_word_count"],
        bins=20,
        kde=True
    )

    plt.title("Combined Text Word Count Distribution")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.show()

    # Date frequencies bar graph

    plt.figure(figsize=(10,5))

    date_counts = df["date"].value_counts().sort_index()

    sns.barplot(
        x=date_counts.index.astype(str),
        y=date_counts.values
    )

    plt.xticks(rotation=45)
    plt.title("Articles Published Per Day")
    plt.xlabel("Date")
    plt.ylabel("Article Count")

    plt.show()

    # Confidence by sentiment box and whisker plots

    plt.figure(figsize=(8,5))

    sns.boxplot(
        data=df,
        x="sentiment",
        y="sen_confidence"
    )

    plt.title("Confidence by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Confidence Score")
    plt.show()

    # Sentiment by date

    df["date"] = pd.to_datetime(df["date"])

    sentiment_time = pd.crosstab(
        df["date"],
        df["sentiment"]
    )

    sentiment_time.index = sentiment_time.index.strftime("%Y-%m-%d")

    sentiment_time.plot(
        kind="bar",
        stacked=False,
        figsize=(10,6)
    )

    plt.title("Sentiment Distribution Over Time")
    plt.xlabel("Date")
    plt.ylabel("Article Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Sentiment by news source bar graph

    top_sources = df["source"].value_counts().head(5).index

    filtered_df = df[df["source"].isin(top_sources)]

    source_sentiment = pd.crosstab(
        filtered_df["source"],
        filtered_df["sentiment"]
    )

    source_sentiment.plot(
        kind="bar",
        stacked=False,
        figsize=(10,6)
    )

    plt.title("Sentiment by News Source")
    plt.xlabel("Source")
    plt.ylabel("Article Count")
    plt.xticks(rotation=45)
    plt.show()

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
    high_confidence = df[df["sen_confidence"] > 0.90]
    print("\nHigh confidence articles (>0.90):", len(high_confidence))

    # DATA PRODUCT AND DESCRIPTION
    df.to_csv("stocks_v2.csv", index=False)

    # =========================
    # VOLUME vs SENTIMENT VISUALIZATION
    # =========================
    # Build a per-day frame using `results` (transformed_dataset), which
    # already has aligned `volume` and per-article `sentiment_results`.
    #
    # Daily net sentiment score:
    #   for each article:  +confidence if positive, -confidence if negative, 0 if neutral
    #   daily score = mean of those signed confidences  (range roughly [-1, 1])

    sentiment_sign = {"positive": 1, "negative": -1, "neutral": 0}

    rows = []
    for day, payload in results.items():
        scores = [
            sentiment_sign.get(s["label"].lower(), 0) * s["score"]
            for s in payload["sentiment_results"]
        ]
        net_sentiment = sum(scores) / len(scores) if scores else 0.0
        rows.append({
            "date": day,
            "volume": payload["volume"],
            "article_count": len(payload["articles"]),
            "net_sentiment": net_sentiment,
        })

    daily_df = pd.DataFrame(rows)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    print("\nDaily volume vs sentiment:")
    print(daily_df)

    # Daily volume vs sentiment
    fig, ax1 = plt.subplots(figsize=(11, 6))

    x_labels = daily_df["date"].dt.strftime("%Y-%m-%d")

    ax1.bar(
        x_labels,
        daily_df["volume"],
        color="steelblue",
        alpha=0.6,
        label="Trade Volume",
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Trade Volume", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(
        x_labels,
        daily_df["net_sentiment"],
        color="darkorange",
        marker="o",
        linewidth=2,
        label="Net Sentiment",
    )
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Net Sentiment (−1 to +1)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(-1, 1)

    plt.title(f"{symbol}: Daily Trade Volume vs. FinBERT Net Sentiment")
    fig.tight_layout()
    plt.show()