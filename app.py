from flask import Flask, render_template, request
from screener_logic import screen_best_shorts_web, screen_single_stock_web

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    error = None

    if request.method == "POST":
        api_key = request.form.get("api_key", "").strip()
        model_name = request.form.get("model_name", "").strip()
        stock = request.form.get("stock", "").strip()
        action = request.form.get("action", "")

        # Parse advanced settings
        try:
            if action == "screen_all":
                settings = {
                    "min_percent": float(request.form.get("min_percent", 20)),
                    "min_price": float(request.form.get("min_price", 3)),
                    "min_cap": float(request.form.get("min_cap", 25_000_000)),
                    "max_cap": float(request.form.get("max_cap", 250_000_000_000)),
                    "num_headlines": int(request.form.get("num_headlines", 1)),
                    "summary_sentences": int(request.form.get("summary_sentences", 3))
                }
            elif action == "screen_single":
                settings = {
                    "num_headlines": int(request.form.get("num_headlines_single", 1)),
                    "summary_sentences": int(request.form.get("summary_sentences_single", 3))
                }
            else:
                settings = {}
        except ValueError:
            error = "Invalid input in advanced settings."
            return render_template("index.html", results=None, error=error)

        # Validate API key
        if not api_key:
            error = "Groq API key is required."
        else:
            if action == "screen_all":
                results = screen_best_shorts_web(api_key, model_name, settings)
            elif action == "screen_single" and stock:
                results = screen_single_stock_web(api_key, model_name, stock, settings)
            else:
                error = "No action selected or missing stock ticker."

    return render_template("index.html", results=results, error=error)

if __name__ == "__main__":
    app.run(debug=True)
