FROM python:3.11-slim

# Install system dependencies for Chrome/Chromium + fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg ca-certificates curl unzip \
    fonts-liberation locales \
    libglib2.0-0 libnss3 libx11-6 libx11-xcb1 \
    libxcomposite1 libxdamage1 libxext6 libxfixes3 \
    libxrandr2 libxkbcommon0 libdrm2 libgbm1 \
    libpango-1.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# (Optional) Install Chromium (if your app needs a browser)
RUN apt-get update && apt-get install -y chromium && rm -rf /var/lib/apt/lists/*

# Set Python environment
ENV PYTHONUNBUFFERED=1

# Copy requirements + install
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .

# Expose port for Flask
EXPOSE 10000

# Start app with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:10000", "app:app"]
