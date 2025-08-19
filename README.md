# 📈 DECA SMG Stock Screener
🚀 Features
🔍 Screen All Stocks: Filter based on percent gain, price, market cap, and headline count

🧠 Single Stock Insights: Analyze and summarize headlines for a specified ticker

🗞️ Headline Summarization: Uses Groq LLMs for quick news digests

⚙️ Advanced Settings: Customize summary length and headlines per ticker

⭐ Interactive Sorting: Rank results by star count or risk tolerance

⛔ Drawbacks: Initial setup will take time; Currently only supported on Windows; Program will take A LOT of time to process the information, so it will take a lot of time to give you an output, especially if you tweak the advanced settings

If you liked this program:

[![Star this repo](https://img.shields.io/github/stars/abam55/DECA-ShortScanner-SMG?style=social)](https://github.com/abam55/DECA-ShortScanner-SMG/stargazers)

# 📚 Resources needed
Chrome Web Browser

Python IDE or Code Editor (PyCharm or VS Code works)

Groq API(Will cover later)


# 📂 Project Structure

├── app.py 

├── templates/

│   └── index.html      

├── screener_logic.py    

└── README.md          

# 🛠️ Installation

In order to grab my code, you will need to install Git at https://git-scm.com/downloads/win (If you have any IDE or Code editor open, close and reopen once finished with installation)

Clone the repository:

In the terminal of your IDE/Code editor, copy and paste:

git clone https://github.com/abam55/DECA-ShortScanner-SMG.git

cd DECA-ShortScanner-SMG

# Install dependencies
In the same terminal, type: pip install flask yfinance requests beautifulsoup4 selenium newspaper3k groq chromedriver-autoinstaller lxml_html_clean

# 🔐 Configuration
Head over to https://console.groq.com and create an account to generate your API key.

Once you have it:

Enter your key and model name (e.g., llama3-8b-8192) into the form on the homepage

Adjust advanced settings as needed before launching a scan

# 🚦 How to Use
In the same terminal, type: flask python app.py

Navigate to http://127.0.0.1:5000/ in your browser. From there:

Use the Run Screener for All Stocks button to filter stocks based on your criteria

Use the Screen Single Stock button to analyze a specific ticker

Results will be summarized using the selected LLM and dynamically sortable.

# 🙏 Credits
Project Author: abam55 — Designed, developed, and tested the DECA SMG Stock Screener.

Groq API — Used for large language model-based news summarization and stock analysis.
Website: https://console.groq.com

TradingView — Source of stock market gainers data for real-time screening.
Website: https://www.tradingview.com

Yahoo Finance (via yfinance) — Used to fetch real-time stock prices and market cap data.
Library: https://github.com/ranaroussi/yfinance

Newspaper3k — For extracting full text from news articles.
Library: https://github.com/codelucas/newspaper

Bootstrap + Bootstrap Icons — Frontend styling and UI components.
CDN: https://getbootstrap.com
