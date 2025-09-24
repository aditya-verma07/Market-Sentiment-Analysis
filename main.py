import requests
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
import yfinance as yf

nltk.download('vader_lexicon')

API_KEY = "e2f68c88e825448a8ce431e9dd70d590" 
TICKERS = {"Tesla": "Tesla", "BYD": "BYD"}  

def fetch_headlines(api_key, query, language='en', page_size=20):
    """
    Fetch recent news headlines for a given query
    """
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'language': language,
        'pageSize': page_size,
        'sortBy': 'publishedAt',
        'apiKey': api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    articles = data.get('articles', [])
    headlines = [article['title'] for article in articles if article.get('title')]
    return headlines

headlines_data = {}
for name, query in TICKERS.items():
    headlines = fetch_headlines(API_KEY, query)
    headlines_data[name] = headlines
    print(f"{name}: fetched {len(headlines)} headlines")
sid = SentimentIntensityAnalyzer()

def analyze_sentiment(headlines):
    sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for headline in headlines:
        score = sid.polarity_scores(str(headline))['compound']
        if score >= 0.05:
            sentiment_counts['Positive'] += 1
        elif score <= -0.05:
            sentiment_counts['Negative'] += 1
        else:
            sentiment_counts['Neutral'] += 1
    return sentiment_counts

sentiment_results = {}
for company, headlines in headlines_data.items():
    sentiment_results[company] = analyze_sentiment(headlines)
    print(f"{company} sentiment counts: {sentiment_results[company]}")
def plot_sentiment(sentiments, title):
    companies = list(sentiments.keys())
    categories = ['Positive', 'Neutral', 'Negative']
    
    counts = {cat: [sentiments[company][cat] for company in companies] for cat in categories}
    x = range(len(companies))
    width = 0.25
    plt.figure(figsize=(8,5))
    plt.bar([p - width for p in x], counts['Positive'], width=width, color='green', label='Positive')
    plt.bar(x, counts['Neutral'], width=width, color='gray', label='Neutral')
    plt.bar([p + width for p in x], counts['Negative'], width=width, color='red', label='Negative')
    plt.xticks(x, companies)
    plt.ylabel('Number of Headlines')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_sentiment(sentiment_results, "Sentiment Analysis of Recent News Headlines")
def fetch_stock_data(ticker, period='1mo'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data['Close']

tesla_stock = fetch_stock_data('TSLA')
byd_stock = fetch_stock_data('BYDDY') 
print("\nTesla recent close prices:\n", tesla_stock.tail())
print("\nBYD recent close prices:\n", byd_stock.tail())
