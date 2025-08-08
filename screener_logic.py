import time
import math
import datetime
from urllib.parse import urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup
from newspaper import Article

import yfinance as yf
import numpy as np
import pandas as pd

from groq import Groq
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# === CONFIGURATION ===
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/115.0.0.0 Safari/537.36"
)

# ---------------------------
# Client init
# ---------------------------
def init_client(api_key):
    return Groq(api_key=api_key)


# ---------------------------
# Fetching & resolving URLs
# ---------------------------
def resolve_final_url(url):
    headers = {"User-Agent": USER_AGENT}
    try:
        if "news.google.com" in url and "url=" in url:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            actual_url = qs.get("url")
            if actual_url:
                return unquote(actual_url[0])
    except Exception:
        pass

    try:
        res = requests.get(url, headers=headers, timeout=7, allow_redirects=True)
        soup = BeautifulSoup(res.content, "html.parser")
        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            return canonical["href"]
        og = soup.find("meta", property="og:url")
        if og and og.get("content"):
            return og["content"]
        return res.url
    except Exception:
        return url


# ---------------------------
# Summarization via LLM
# ---------------------------
def summarize_text(raw, ticker, client, model, num_sentences):
    if not raw or not client or not model:
        return "No summary available."

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
        return summary + "." if summary and not summary.endswith(".") else summary
    except Exception:
        return "Summary generation failed."


def fetch_news(ticker, client, model, num_headlines, num_sentences):
    rss = f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(rss, timeout=7, headers=headers)
        if r.status_code != 200:
            return []
    except Exception:
        return []

    items = BeautifulSoup(r.content, "xml").find_all("item")[:max(0, int(num_headlines or 0))]
    news = []
    for item in items:
        try:
            title = item.title.text.strip()
            link = resolve_final_url(item.link.text.strip())
        except Exception:
            continue

        raw = ""
        # Try newspaper first
        try:
            art = Article(link)
            art.download()
            art.parse()
            raw = art.text.strip()
        except Exception:
            raw = ""

        try:
            if (not raw) and getattr(item, "description", None):
                raw = BeautifulSoup(item.description.text, "html.parser").get_text().strip()
        except Exception:
            pass

        summary = summarize_text(raw, ticker, client, model, num_sentences) if raw and client and model else (BeautifulSoup(item.description.text, "html.parser").get_text().strip() if getattr(item, "description", None) else raw[:800])
        news.append((title, link, summary, raw))
    return news


# ---------------------------
# News sentiment & event extraction
# ---------------------------
def analyze_news_sentiment(client, model, news_items):
    """
    Input: news_items list of tuples (title, link, summary, raw)
    Output: dict with:
      - sentiment: "Negative"/"Neutral"/"Positive"
      - score: [-1.0, 1.0]
      - event_flags: (e.g., 'earnings_miss','guidance_cut','layoffs','lawsuit','upgrade','downgrade')
    """
    # Combine headlines & summaries into a single prompt body
    combined = ""
    for title, link, summary, raw in news_items:
        piece = title + "\n" + (summary or raw or "")
        combined += piece + "\n---\n"

    # Default fallback
    def heuristic(items_text):
        negative_kw = ["miss", "downgrad", "cut guidance", "lawsuit", "investigation", "recall", "layoff", "bankrupt", "fraud", "profit warning", "sell-off", "plunge", "delist", "sue", "suspension", "fine"]
        positive_kw = ["beat", "upgrade", "raise guidance", "acquisition", "buyback", "partnership", "beat estimates", "record revenue", "record profit", "upgrade to", "positive outlook"]
        neg = 0
        pos = 0
        flags = set()
        text = combined.lower()
        for kw in negative_kw:
            if kw in text:
                neg += 1
                flags.add(kw.replace(" ", "_"))
        for kw in positive_kw:
            if kw in text:
                pos += 1
                flags.add(kw.replace(" ", "_"))
        score = 0.0
        if pos + neg > 0:
            score = (pos - neg) / max(1, pos + neg)
            score = max(-1.0, min(1.0, score))
        sentiment = "Negative" if score < -0.15 else ("Positive" if score > 0.15 else "Neutral")
        return {'sentiment': sentiment, 'score': score, 'event_flags': list(flags)}

    # If we don't have a client/model, use heuristic
    if not client or not model or not news_items:
        return heuristic(combined)

    # Build LLM prompt
    prompt = (
        "You are a financial news analyzer. Given the following concatenated news headlines and short summaries, "
        "produce a small JSON object with three fields: sentiment (one of Negative, Neutral, Positive), "
        "score (a number between -1.0 and 1.0 where -1 is very bearish and +1 is very bullish), "
        "and event_flags (a JSON array of short string flags about any important events that would affect shorting, e.g. "
        '["earnings_miss","guidance_cut","layoffs","lawsuit","upgrade","downgrade"]).'
        "Only output valid JSON and nothing else.\n\n"
        f"News:\n{combined}\n\nRespond with JSON only."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict JSON-outputting financial news classifier."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        raw_out = resp.choices[0].message.content.strip()
        # Sometimes the model adds backticks or text...try to extract JSON substring
        import json, re
        m = re.search(r'(\{.*\})', raw_out, re.DOTALL)
        json_text = m.group(1) if m else raw_out
        parsed = json.loads(json_text)
        # Normalize outputs
        sentiment = parsed.get('sentiment', 'Neutral')
        score = float(parsed.get('score', 0.0))
        event_flags = parsed.get('event_flags', []) if isinstance(parsed.get('event_flags', []), list) else []
        score = max(-1.0, min(1.0, score))
        return {'sentiment': sentiment, 'score': score, 'event_flags': event_flags}
    except Exception:
        # Fall back to heuristic if LLM fails
        return heuristic(combined)


# ---------------------------
# Market cap helpers
# ---------------------------
def format_market_cap(mcap):
    try:
        mcap = float(mcap or 0.0)
    except Exception:
        return "0.00"
    if mcap >= 1e9: return f"{mcap/1e9:.2f}B"
    if mcap >= 1e6: return f"{mcap/1e6:.2f}M"
    if mcap >= 1e3: return f"{mcap/1e3:.2f}K"
    return f"{mcap:.2f}"


def clean_market_cap(raw):
    if raw is None:
        return 0.0
    try:
        s = str(raw).replace('\u202f','').replace(',','').replace(' USD','')
        if 'B' in s: return float(s.replace('B','')) * 1e9
        if 'M' in s: return float(s.replace('M','')) * 1e6
        if 'K' in s: return float(s.replace('K','')) * 1e3
        if s.strip()=='—': return 0.0
        return float(s)
    except Exception:
        return 0.0


# ---------------------------
# Card rendering
# ---------------------------
def render_stock_card(ticker, name, price, change_pct, mcap, headlines, short_rating, risk_tolerance):
    news_html = "".join(
        f'<li class="headline-item"><a href="{link}" target="_blank">{title}</a><div class="summary-text">{summary}</div></li>'
        for title, link, summary in headlines
    )
    risk_data = (risk_tolerance or "Unknown").lower()
    return f"""
<div class="card mb-4 shadow-sm result-card" data-risk="{risk_data}">
  <div class="card-body">
    <h5 class="card-title text-primary">🔹 {ticker} - {name}</h5>
    <p><strong>💰</strong> ${price:.2f} | {change_pct:.2f}% | Market Cap: {format_market_cap(mcap)}</p>
    <hr/>
    <h6 class="text-secondary">📰 News:</h6>
    <ul class="headline-list">{news_html}</ul>
    <hr/>
    <p><strong>🔻 Short Rating:</strong> {short_rating}</p>
    <p><strong>📊 Risk Tolerance:</strong> {risk_tolerance}</p>
  </div>
</div>
"""


# ---------------------------
# Price / indicator helpers
# ---------------------------
def fetch_price_volume_data(ticker, period="90d", interval="1d"):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval, actions=False)
        if hist is None or hist.empty:
            return pd.DataFrame()
        hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        hist.dropna(inplace=True)
        return hist
    except Exception:
        return pd.DataFrame()


def compute_rsi(series, window=14):
    try:
        if series is None or len(series) < window + 1:
            return None
        delta = series.diff().dropna()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.ewm(alpha=1/window, adjust=False).mean()
        ma_down = down.ewm(alpha=1/window, adjust=False).mean()
        rs = ma_up / ma_down.replace({0: np.nan})
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    except Exception:
        return None


def pct_change_recent(series, days=1):
    try:
        if series is None or len(series) < days + 1:
            return 0.0
        return float((series.iloc[-1] / series.iloc[-(1 + days)] - 1) * 100)
    except Exception:
        return 0.0


def avg_volume(series, window=20):
    try:
        if series is None or series.empty:
            return 0.0
        if len(series) < window:
            return float(series.mean())
        return float(series[-window:].mean())
    except Exception:
        return 0.0


def normalize(x, lo, hi, invert=False):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 0.5
        if lo == hi:
            return 0.5
        v = (float(x) - float(lo)) / (float(hi) - float(lo))
        v = max(0.0, min(1.0, v))
        return 1.0 - v if invert else v
    except Exception:
        return 0.5


# ---------------------------
# Quantitative short rating
# ---------------------------
def get_short_rating_quant(ticker, news_sentiment_score=0.0, news_event_flags=None, settings=None):
    """
    Compute a quantitative short score and return (stars_string, details_dict).
    news_sentiment_score: float in [-1..1] where negative is bearish (supports short)
    news_event_flags: list for important flags like 'earnings_miss','guidance_cut','layoffs' etc.
    settings: optional dict to tune weights and thresholds
    """
    settings = settings or {}
    news_event_flags = news_event_flags or []

    # default weights (sum approx 1.0)
    weights = {
        'valuation':   settings.get('w_valuation', 0.15),
        'momentum':    settings.get('w_momentum', 0.18),
        'rsi':         settings.get('w_rsi', 0.15),
        'volume':      settings.get('w_volume', 0.10),
        'short_int':   settings.get('w_short', 0.10),
        'liquidity':   settings.get('w_liq', 0.10),
        'news':        settings.get('w_news', 0.22),   # news is now a significant component
    }

    # fetch basic info
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
    except Exception:
        info = {}

    trailingPE = info.get('trailingPE') or None
    forwardPE = info.get('forwardPE') or None
    shortPercentOfFloat = info.get('shortPercentOfFloat') or info.get('shortRatio') or 0.0
    avgVol3m = info.get('averageDailyVolume3Month') or info.get('averageDailyVolume10Day') or 0
    marketCap = info.get('marketCap') or 0
    next_earnings = info.get('nextEarningsDate') or info.get('earningsTimestamp') or None

    hist = fetch_price_volume_data(ticker, period="120d", interval="1d")
    if hist.empty:
        return "⭐❓", {'score': 0.0, 'reason': 'no_price_data'}

    closes = hist['Close']
    vols = hist['Volume']

    today_vol = float(vols.iloc[-1]) if len(vols) > 0 else 0.0
    vol_20 = avg_volume(vols, window=20) if len(vols) > 0 else 0.0
    vol_spike = (today_vol / vol_20) if vol_20 > 0 else 1.0

    gap_pct = 0.0
    if len(closes) >= 2:
        prev_close = closes.iloc[-2]
        gap_pct = 0.0 if prev_close == 0 else float((closes.iloc[-1] / prev_close - 1) * 100)

    ret_1 = pct_change_recent(closes, days=1)
    ret_3 = pct_change_recent(closes, days=3)
    ret_5 = pct_change_recent(closes, days=5)
    rsi = compute_rsi(closes, window=14) or 50.0

    # Valuation score: higher PE -> more likely overvalued
    pe_for_score = forwardPE if forwardPE is not None else trailingPE
    if pe_for_score is None:
        pe_for_score = 30.0
    val_score = normalize(pe_for_score, lo=5, hi=100, invert=False)

    # Momentum: recent positive returns -> candidate for mean reversion
    mom = max(0.0, max(ret_1, ret_3, ret_5))
    mom_score = normalize(min(mom, 30.0), lo=0, hi=30, invert=False)

    # RSI
    rsi_score = normalize(rsi, lo=30, hi=85, invert=False)

    # Volume spike
    vol_score = normalize(min(vol_spike, 5.0), lo=1.0, hi=4.0, invert=False)

    # Short interest (avoid extremely high short interest because of squeeze risk)
    short_pct = float(shortPercentOfFloat or 0.0)
    if short_pct <= 0:
        short_score = 0.5
    else:
        short_score = normalize(short_pct, lo=0, hi=40, invert=True)

    # Liquidity
    liq = 1.0
    if avgVol3m and avgVol3m > 0:
        liq = normalize(min(avgVol3m, 10_000_000), lo=1_000, hi=100_000, invert=False)
    else:
        liq = 0.3

    # News component: map news_sentiment_score [-1..1] to 0..1 where more negative -> higher short score
    # Negative news should strongly support shorting, positive news reduces short attractiveness.
    news_score = 0.5 - 0.5 * (news_sentiment_score)  # so -1 => 1.0 (very bearish), +1 => 0.0 (very bullish)

    # Event flags: if certain negative events appear, we'll enforce stronger penalties later
    event_flags = set([f.lower() for f in (news_event_flags or [])])

    comp_scores = {
        'valuation': val_score,
        'momentum':  mom_score,
        'rsi':       rsi_score,
        'volume':    vol_score,
        'short_int': short_score,
        'liquidity': liq,
        'news':      news_score
    }

    combined = sum(comp_scores[k] * weights[k] for k in weights)
    score_pct = combined * 100.0

    reason_notes = []
    # Penalties: short-squeeze and event-driven risk
    if short_pct >= 20.0:
        score_pct *= 0.6
        reason_notes.append(f"high_short_interest({short_pct:.1f}%)")
    if avgVol3m and avgVol3m < 2000:
        score_pct *= 0.6
        reason_notes.append(f"low_liq({int(avgVol3m)})")

    # Negative event flags that increase uncertainty (even if sentiment is negative),
    # because some events (e.g., major regulatory action) can cause unpredictable moves.
    squeeze_like_flags = {'short_squeeze', 'high_options_gamma', 'large_call_oi'}
    severe_negative_flags = {'lawsuit', 'investigation', 'bankrupt', 'receivership'}
    earnings_flags = {'earnings_miss', 'guidance_cut', 'earnings_warn'}

    # If severe negative flags present, we consider them supportive of the short (increase score)
    if event_flags.intersection(earnings_flags):
        # earnings miss or guidance cut -> more likely to fall next day
        score_pct *= 1.05
        reason_notes.append("earnings_related_flag")
    if event_flags.intersection(severe_negative_flags):
        score_pct *= 1.05
        reason_notes.append("severe_negative_flag")

    # But if "short_squeeze" or high options attention, reduce score because of squeeze risk
    if event_flags.intersection(squeeze_like_flags) or short_pct >= 15.0:
        score_pct *= 0.7
        reason_notes.append("squeeze_risk_flag")

    # earnings proximity penalty (if actual earnings date is imminent)
    days_to_earnings = None
    if next_earnings:
        try:
            if isinstance(next_earnings, (int, float)):
                e_dt = datetime.datetime.utcfromtimestamp(int(next_earnings))
            else:
                e_dt = pd.to_datetime(next_earnings)
            days_to_earnings = (e_dt - datetime.datetime.utcnow()).days
            if days_to_earnings <= 3:
                score_pct *= 0.6
                reason_notes.append(f"earnings_in_{days_to_earnings}d")
        except Exception:
            pass

    # runup without volume reduces confidence
    if mom_score > 0.25 and vol_score < 0.6:
        score_pct *= 0.85
        reason_notes.append("runup_without_volume")

    # If news is strongly positive, reduce score more
    if news_sentiment_score > 0.4:
        score_pct *= 0.7
        reason_notes.append("positive_news")

    score_pct = max(0.0, min(100.0, score_pct))

    # Map to stars
    if score_pct >= 80:
        stars = "⭐⭐⭐⭐⭐"
    elif score_pct >= 60:
        stars = "⭐⭐⭐⭐"
    elif score_pct >= 40:
        stars = "⭐⭐⭐"
    elif score_pct >= 20:
        stars = "⭐⭐"
    else:
        stars = "⭐"

    details = {
        'score': round(score_pct, 2),
        'components': comp_scores,
        'weights': weights,
        'pe': pe_for_score,
        'short_pct': short_pct,
        'avg_vol_3m': avgVol3m,
        'vol_spike': round(vol_spike, 2),
        'rsi': round(rsi, 2),
        'gap_pct': round(gap_pct, 2),
        'ret_1': round(ret_1, 2),
        'ret_3': round(ret_3, 2),
        'ret_5': round(ret_5, 2),
        'marketCap': marketCap,
        'days_to_earnings': days_to_earnings,
        'news_sentiment_score': news_sentiment_score,
        'news_event_flags': list(event_flags),
        'reason_notes': reason_notes
    }

    return stars, details


# ---------------------------
# Wrapper (maintain compatibility)
# ---------------------------
def get_short_rating(ticker, text_for_ai, client=None, model=None, settings=None):
    """
    Wrapper that will:
      - use the provided client+model to analyze news sentiment (if available),
      - then compute quantitative short rating that incorporates news score.
    Returns: star string for UI compatibility.
    """
    # Attempt to parse news_text (text_for_ai) to list of headlines if possible
    # In normal usage we pass raw_block as text_for_ai — but we also have the fetch_news results
    # so caller (screen_* functions) will call analyze_news_sentiment before calling this wrapper.
    # For safety, attempt heuristic fallback:
    try:
        # If caller passed settings.news_context (not standard) use it
        news_sentiment_score = 0.0
        news_event_flags = []
        # allow settings to carry analyzed news info
        if settings and isinstance(settings, dict) and 'news_analysis' in settings:
            na = settings.get('news_analysis') or {}
            news_sentiment_score = float(na.get('score', 0.0))
            news_event_flags = na.get('event_flags', []) or []
        else:
            # try lightweight heuristic on text_for_ai if present
            if text_for_ai:
                # simple heuristic: negative words - positive words normalized
                negative_kw = ["miss", "downgrad", "cut guidance", "lawsuit", "investigation", "recall", "layoff", "bankrupt", "fraud", "warning", "sell-off", "plunge", "fine"]
                positive_kw = ["beat", "upgrade", "raise guidance", "acquisition", "buyback", "partnership", "beat estimates", "record revenue", "upgrade"]
                txt = text_for_ai.lower()
                neg = sum(txt.count(k) for k in negative_kw)
                pos = sum(txt.count(k) for k in positive_kw)
                if pos + neg > 0:
                    news_sentiment_score = (pos - neg) / max(1, pos + neg)
                else:
                    news_sentiment_score = 0.0
                # flags
                flags = set()
                for kw in negative_kw + positive_kw:
                    if kw in txt:
                        flags.add(kw.replace(" ", "_"))
                news_event_flags = list(flags)

        stars, details = get_short_rating_quant(ticker, news_sentiment_score=news_sentiment_score, news_event_flags=news_event_flags, settings=settings)
        return stars
    except Exception:
        # fallback to original LLM approach if provided
        try:
            if client and model:
                prompt = (
                    f"Given this information about {ticker}, how good is this stock "
                    f"as a shorting opportunity on a scale of 1 to 5 stars?\n\n"
                    f"{text_for_ai}\n\nRespond with just a number from 1 to 5."
                )
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
        except Exception:
            pass
    return "⭐❓"


# ---------------------------
# Risk tolerance: (High/Medium/Low)
# ---------------------------
def get_risk_tolerance_via_ai(ticker, name, summary_text, client=None, model=None):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
    except Exception:
        info = {}

    short_pct = float(info.get('shortPercentOfFloat') or info.get('shortRatio') or 0.0)
    avgVol3m = info.get('averageDailyVolume3Month') or info.get('averageDailyVolume10Day') or 0
    marketCap = info.get('marketCap') or 0
    next_earnings = info.get('nextEarningsDate') or info.get('earningsTimestamp') or None

    days_to_earn = None
    if next_earnings:
        try:
            if isinstance(next_earnings, (int, float)):
                e_dt = datetime.datetime.utcfromtimestamp(int(next_earnings))
            else:
                e_dt = pd.to_datetime(next_earnings)
            days_to_earn = (e_dt - datetime.datetime.utcnow()).days
        except Exception:
            days_to_earn = None

    if short_pct >= 20.0:
        return "High"
    if avgVol3m and avgVol3m < 2000:
        return "High"
    if days_to_earn is not None and days_to_earn <= 3:
        return "High"
    if marketCap < 500_000_000 or (avgVol3m and avgVol3m < 20_000):
        return "Medium"
    return "Low"


# ---------------------------
# Web scrapers & Screens
# ---------------------------
def screen_best_shorts_web(api_key, model, settings):
    client = init_client(api_key) if api_key else None

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--log-level=3")

    try:
        driver = webdriver.Chrome(options=opts)
    except Exception as e:
        print(f"[ERROR] Selenium Chrome driver init failed: {e}")
        return "<p class='text-danger'>Selenium driver failed to start on server. Ensure chromedriver is installed and accessible.</p>"

    try:
        driver.get("https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/")
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
    except Exception as e:
        print(f"[ERROR] Failed to fetch TradingView page: {e}")
        soup = BeautifulSoup("", "html.parser")
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    rows = soup.find_all("tr", class_="row-RdUXZpkv listRow")
    if not rows:
        rows = soup.find_all("tr")

    cards = []
    for row in rows:
        try:
            cols = row.find_all("td")
            if len(cols) < 3:
                continue

            ticker_el = row.find("span", class_="tickerCell-GrtoTeat")
            if ticker_el:
                t_a = ticker_el.find("a")
                t_sup = ticker_el.find("sup")
                ticker = t_a.text.strip() if t_a else None
                name = t_sup.text.strip() if t_sup else (ticker or "Unknown")
            else:
                ticker = row.get('data-symbol') or (cols[0].get_text().strip() if cols else None)
                name = ticker

            if not ticker:
                continue

            price_text = cols[2].text if len(cols) > 2 else "0"
            price = 0.0
            try:
                price = float(price_text.replace(' USD', '').replace(',', '').strip())
            except Exception:
                price = 0.0

            pct = 0.0
            try:
                ch_el = cols[1].find("span")
                if ch_el:
                    pct = float(ch_el.text.replace('%', '').replace('+', '').replace(',', '').strip())
                else:
                    pct = float(cols[1].text.replace('%', '').replace('+', '').strip())
            except Exception:
                pct = 0.0

            mcap = clean_market_cap(cols[5].text.strip()) if len(cols) > 5 else 0.0
            prev_p = price / (1 + pct / 100) if (1 + pct / 100) != 0 else price

            meets = (
                (pct > settings.get("min_percent", 20)) and
                (settings.get("min_cap", 25_000_000) <= mcap <= settings.get("max_cap", 250_000_000_000)) and
                (prev_p > settings.get("min_price", 3))
            )

            if not meets:
                continue

            news4 = fetch_news(ticker, client, model, settings.get("num_headlines", 1), settings.get("summary_sentences", 3))
            # Analyze news sentiment using LLM if available
            news_analysis = analyze_news_sentiment(client, model, news4)
            news_score = news_analysis.get('score', 0.0)
            news_flags = news_analysis.get('event_flags', [])

            raw_block = "\n".join(r for _, _, __, r in news4)
            combined_settings = dict(settings or {})
            combined_settings['news_analysis'] = news_analysis

            short_rt = get_short_rating(ticker, raw_block, client, model, settings=combined_settings)
            risk = get_risk_tolerance_via_ai(ticker, name, raw_block, client, model)
            ui_head = [(t, l, ds) for t, l, ds, _ in news4]

            cards.append(render_stock_card(
                ticker, name, price, pct, mcap,
                ui_head, short_rt, risk
            ))

        except Exception as e:
            print(f"[WARN] Error processing row: {e}")
            continue

    return "\n".join(cards) or "<p class='text-muted'>🚫 No stocks met the screening criteria today.</p>"


def screen_single_stock_web(api_key, model, ticker, settings):
    ticker = ticker.upper().strip()
    client = init_client(api_key) if api_key else None

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        name = info.get("shortName", ticker)
        price = info.get("regularMarketPrice", 0.0) or 0.0
        prev_close = info.get("regularMarketPreviousClose", price) or price
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
        mcap = info.get("marketCap", 0.0) or 0.0
    except Exception:
        name = ticker
        price = 0.0
        change_pct = 0.0
        mcap = 0.0

    news4 = fetch_news(ticker, client, model, settings.get("num_headlines", 1), settings.get("summary_sentences", 3))
    news_analysis = analyze_news_sentiment(client, model, news4)
    raw_block = "\n".join(r for _, _, __, r in news4)

    combined_settings = dict(settings or {})
    combined_settings['news_analysis'] = news_analysis

    short_rt = get_short_rating(ticker, raw_block, client, model, settings=combined_settings)
    risk = get_risk_tolerance_via_ai(ticker, name, raw_block, client, model)
    ui_head = [(t, l, ds) for t, l, ds, _ in news4]

    return render_stock_card(
        ticker, name, price, change_pct, mcap,
        ui_head, short_rt, risk
    )
