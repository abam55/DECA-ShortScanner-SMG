from flask import Flask, render_template, request, session
from screener_logic import screen_best_shorts_web, screen_single_stock_web
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    error = None

    if request.method == 'POST':
        api_key = request.form.get('api_key') or session.get('api_key')
        action = request.form.get('action', 'screen_all')
        stock = request.form.get('stock', '').strip()

        advanced_settings = {
            'min_percent': float(request.form.get('min_percent', 20)),
            'min_cap': float(request.form.get('min_cap', 25_000_000)),
            'max_cap': float(request.form.get('max_cap', 250_000_000)),
            'min_price': float(request.form.get('min_price', 3)),
            'num_headlines': int(request.form.get('num_headlines', 1)),
            'summary_sentences': int(request.form.get('summary_sentences', 3))
        }

        if not api_key:
            error = "API key required."
            return render_template('index.html', results=results, error=error)

        session['api_key'] = api_key

        try:
            if action == 'screen_single':
                if not stock:
                    error = "Please enter a ticker symbol for single stock screening."
                    return render_template('index.html', results=results, error=error)
                raw_output = screen_single_stock_web(api_key, stock, advanced_settings)
            else:
                raw_output = screen_best_shorts_web(api_key, advanced_settings)
        except Exception as e:
            error = str(e)
            raw_output = ""

        results = raw_output

    return render_template('index.html', results=results, error=error)


if __name__ == "__main__":
    app.run(debug=True)
