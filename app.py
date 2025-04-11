import os
from groq import Groq
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import requests
import time

# === CONFIGURABLE PARAMETERS === #
NUM_HEADLINES = 2
SUMMARY_SENTENCES = 2
MIN_MARKET_CAP = 25 * 1_000_000 #Change to 0 for no minimum market cap
MAX_MARKET_CAP = 250 * 1_000_000 #Add _000_000 to remove max market cap restriction
MIN_PERCENT_CHANGE = 50
MIN_PREV_PRICE = 3

# Set up Groq client
client = Groq(api_key="REPLACE WITH YOUR KEY") #Replace the API key

def fetch_news(ticker):
    print(f"Fetching news for ticker: {ticker}")
    news_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(news_url)
    if response.status_code != 200:
        print(f"Error fetching news feed: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, "xml")
    items = soup.find_all("item")
    news = []
    for item in items[:NUM_HEADLINES]:
        title = item.title.text
        link = item.link.text
        summary = summarize_article_from_url(link, ticker)
        news.append((title, link, summary))

    if not news:
        news.append(("No news found", "", ""))
    return news

def summarize_article_from_url(url, ticker):
    try:
        prompt = (
            f"Here's the link to a news article about {ticker} stock: {url}\n\n"
            f"Can you summarize it in {SUMMARY_SENTENCES} sentences?"
        )
        response = client.chat.completions.create(
            model="llama3-70b-8192", #May need to change the model if dysfunctional
            messages=[
                {"role": "system", "content": "You are a financial news summarizer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        summary = response.choices[0].message.content.strip()
        summary_sentences = summary.split(". ")

        summary_sentences = summary_sentences[:SUMMARY_SENTENCES]

        cleaned_summary = ".\n".join(summary_sentences)
        if not cleaned_summary.endswith('.'):
            cleaned_summary += '.'

        return cleaned_summary
    except Exception as e:
        print(f"Groq summarization error: {e}")
        return "Error summarizing content."

# Set up headless browser
options = Options()
options.headless = True
driver = webdriver.Chrome(options=options)

url = "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/"
print(f"Opening TradingView")
driver.get(url)
time.sleep(5)
soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

rows = soup.find_all("tr", class_="row-RdUXZpkv listRow")

print("🟢 Filtered Stocks to Short with Headlines, News Links, and Summaries\n")
found = False

def format_market_cap(market_cap):
    if market_cap >= 1_000_000_000:
        return f"{market_cap / 1_000_000_000:.2f}B"
    elif market_cap >= 1_000_000:
        return f"{market_cap / 1_000_000:.2f}M"
    elif market_cap >= 1_000:
        return f"{market_cap / 1_000:.2f}K"
    else:
        return f"{market_cap:.2f}"

def clean_market_cap(market_cap_raw):
    market_cap_raw = market_cap_raw.replace('\u202f', '').replace(',', '').replace(' USD', '')
    if 'B' in market_cap_raw:
        return float(market_cap_raw.replace('B', '')) * 1_000_000_000
    elif 'M' in market_cap_raw:
        return float(market_cap_raw.replace('M', '')) * 1_000_000
    elif 'K' in market_cap_raw:
        return float(market_cap_raw.replace('K', '')) * 1_000
    elif market_cap_raw.strip() == '—':
        return 0  # If no market cap, treat as 0 to avoid error
    return float(market_cap_raw)

for row in rows:
    try:
        cols = row.find_all("td")
        if len(cols) < 8:
            continue

        ticker = row.find("span", class_="tickerCell-GrtoTeat").find("a").text.strip()
        name = row.find("span", class_="tickerCell-GrtoTeat").find("sup").text.strip()

        price_str = cols[2].text.strip().replace(' USD', '').replace(',', '')
        price = float(price_str)

        percent_change_element = cols[1].find("span", class_="positive-p_QIAEOQ")
        change_percent = float(percent_change_element.text.strip().replace('%', '')) if percent_change_element else None
        if change_percent is None:
            continue

        market_cap_raw = cols[5].text.strip()
        market_cap = clean_market_cap(market_cap_raw)

        prev_price = price / (1 + change_percent / 100)

        if change_percent > MIN_PERCENT_CHANGE and MIN_MARKET_CAP <= market_cap <= MAX_MARKET_CAP and prev_price > MIN_PREV_PRICE:
            found = True
            print(f"\n🔹 {ticker} - {name}")
            print(f"   Price: ${price:.2f} | +{change_percent:.2f}% | Market Cap: {format_market_cap(market_cap)}\n")

            news_items = fetch_news(ticker)
            for title, link, summary in news_items:
                print(f"📰 Headline: {title}")
                print(f"🔗 {link}")
                print("📝 Summary:")
                print(f"   • {summary}")
                print()
                print()
            print("-" * 90)

    except Exception as e:
        print(f"Error processing row: {e}")
        continue

if not found:
    print("🚫 No stocks met the filtering criteria today.")
