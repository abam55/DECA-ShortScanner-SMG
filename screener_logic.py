import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from newspaper import Article
from urllib.parse import urlparse, parse_qs, unquote
from groq import Groq
import yfinance as yf

# === CONFIGURATION === #
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/115.0.0.0 Safari/537.36"
)


def init_client(api_key):
    return Groq(api_key=api_key)


def resolve_final_url(url):
    headers = {"User-Agent": USER_AGENT}

    if "news.google.com" in url and "url=" in url:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        actual_url = qs.get("url")
        if actual_url:
            return unquote(actual_url[0])

    try:
        res = requests.get(url, headers=headers, timeout=5, allow_redirects=True)
        soup = BeautifulSoup(res.content, "html.parser")

        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            return canonical["href"]

        og = soup.find("meta", property="og:url")
        if og and og.get("content"):
            return og["content"]

        return res.url
    except:
        return url


def summarize_text(raw, ticker, client, model, num_sentences):
    prompt = (
        f"Summarize the following article about {ticker} stock in {num_sentences} sentences without starting with, here is a x sentence summary of the article:\n\n{raw}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial news summarizer."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.5
        )
        text = resp.choices[0].message.content.strip()
        parts = text.split(". ")[:num_sentences]
        summary = ". ".join(p.rstrip(".") for p in parts).strip()
        return summary + "." if not summary.endswith(".") else summary
    except:
        return "Summary generation failed."


def fetch_news(ticker, client, model, num_headlines, num_sentences):
    rss = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    r = requests.get(rss, timeout=5, headers={"User-Agent": USER_AGENT})
    if r.status_code != 200:
        return []

    items = BeautifulSoup(r.content, "xml").find_all("item")[:num_headlines]
    news = []
    for item in items:
        title = item.title.text.strip()
        link = resolve_final_url(item.link.text.strip())

        raw = ""
        try:
            art = Article(link)
            art.download()
            art.parse()
            raw = art.text.strip()
        except:
            pass

        if not raw and item.description:
            raw = BeautifulSoup(item.description.text, "html.parser").get_text().strip()

        summary = summarize_text(raw, ticker, client, model, num_sentences) if raw else "No summary available."
        news.append((title, link, summary, raw))
    return news


def get_short_rating(ticker, text_for_ai, client, model):
    prompt = (
        f"Given this information about {ticker}, how good is this stock "
        f"as a shorting opportunity on a scale of 1 to 5 stars?\n\n"
        f"{text_for_ai}\n\nRespond with just a number from 1 to 5."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analyst rating stocks for shorting."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.3
        )
        digits = ''.join(filter(str.isdigit, resp.choices[0].message.content))
        n = max(1, min(5, int(digits or "1")))
        return "⭐" * n
    except:
        return "⭐❓"


def get_risk_tolerance_via_ai(ticker, name, summary_text, client, model):
    if not summary_text.strip():
        return "Unknown"

    prompt = (
        f"Given this summary of recent news about {ticker} ({name}):\n\n"
        f"{summary_text}\n\n"
        f"Rate the risk tolerance for this stock as one of the following: High, Medium, or Low.\n"
        f"Respond with exactly one of these words."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analyst assessing risk tolerance of stocks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        risk = response.choices[0].message.content.strip().capitalize()
        return risk if risk in ["High", "Medium", "Low"] else "Unknown"
    except:
        return "Unknown"


def format_market_cap(mcap):
    if mcap >= 1e9: return f"{mcap/1e9:.2f}B"
    if mcap >= 1e6: return f"{mcap/1e6:.2f}M"
    if mcap >= 1e3: return f"{mcap/1e3:.2f}K"
    return f"{mcap:.2f}"


def clean_market_cap(raw):
    s = raw.replace('\u202f','').replace(',','').replace(' USD','')
    if 'B' in s: return float(s.replace('B','')) * 1e9
    if 'M' in s: return float(s.replace('M','')) * 1e6
    if 'K' in s: return float(s.replace('K','')) * 1e3
    if s.strip()=='—': return 0.0
    return float(s)


def render_stock_card(ticker, name, price, change_pct, mcap, headlines, short_rating, risk_tolerance):
    news_html = "".join(
        f'<li class="headline-item"><a href="{link}" target="_blank">{title}</a><div class="summary-text">{summary}</div></li>'
        for title, link, summary in headlines
    )
    risk_data = risk_tolerance.lower() if risk_tolerance else "unknown"
    return f"""
<div class=\"card mb-4 shadow-sm result-card\" data-risk=\"{risk_data}\">
  <div class=\"card-body\">
    <h5 class=\"card-title text-primary\">🔹 {ticker} - {name}</h5>
    <p><strong>💰</strong> ${price:.2f} | {change_pct:.2f}% | Market Cap: {format_market_cap(mcap)}</p>
    <hr/>
    <h6 class=\"text-secondary\">📰 News:</h6>
    <ul class=\"headline-list\">{news_html}</ul>
    <hr/>
    <p><strong>🔻 Short Rating:</strong> {short_rating}</p>
    <p><strong>📊 Risk Tolerance:</strong> {risk_tolerance}</p>
  </div>
</div>
"""


def screen_best_shorts_web(api_key, model, settings):
    client = init_client(api_key)
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=opts)

    driver.get("https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/")
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    rows = soup.find_all("tr", class_="row-RdUXZpkv listRow")
    cards = []

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 8: continue

        ticker = row.find("span", class_="tickerCell-GrtoTeat").find("a").text.strip()
        name = row.find("span", class_="tickerCell-GrtoTeat").find("sup").text.strip()
        price = float(cols[2].text.replace(' USD','').replace(',','').strip())
        ch_el = cols[1].find("span", class_="positive-p_QIAEOQ")
        pct = float(ch_el.text.replace('%','').strip()) if ch_el else 0
        mcap = clean_market_cap(cols[5].text.strip())
        prev_p = price / (1 + pct/100)

        if (pct > settings["min_percent"]
            and settings["min_cap"] <= mcap <= settings["max_cap"]
            and prev_p > settings["min_price"]):
            news4 = fetch_news(ticker, client, model, settings["num_headlines"], settings["summary_sentences"])
            raw_block = "\n".join(r for _,_,__,r in news4)
            short_rt = get_short_rating(ticker, raw_block, client, model)
            risk = get_risk_tolerance_via_ai(ticker, name, raw_block, client, model)
            ui_head = [(t,l,ds) for t,l,ds,_ in news4]

            cards.append(render_stock_card(
                ticker, name, price, pct, mcap,
                ui_head, short_rt, risk
            ))

    return "\n".join(cards) or "<p class='text-muted'>🚫 No stocks met the screening criteria today.</p>"


def screen_single_stock_web(api_key, model, ticker, settings):
    ticker = ticker.upper()
    client = init_client(api_key)

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        name = info.get("shortName", ticker)
        price = info.get("regularMarketPrice", 0.0)
        prev_close = info.get("regularMarketPreviousClose", price)
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
        mcap = info.get("marketCap", 0.0)
    except:
        name = ticker
        price = 0.0
        change_pct = 0.0
        mcap = 0.0

    news4 = fetch_news(ticker, client, model, settings["num_headlines"], settings["summary_sentences"])
    raw_block = "\n".join(r for _,_,__,r in news4)
    short_rt = get_short_rating(ticker, raw_block, client, model)
    risk = get_risk_tolerance_via_ai(ticker, name, raw_block, client, model)
    ui_head = [(t, l, ds) for t, l, ds, _ in news4]

    return render_stock_card(
        ticker, name, price, change_pct, mcap,
        ui_head, short_rt, risk
    )
