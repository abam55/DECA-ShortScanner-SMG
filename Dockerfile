# Use slim Python image
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg ca-certificates curl \
    fonts-liberation locales \
    libglib2.0-0 libnss3 libgconf-2-4 libatk1.0-0 libatk-bridge2.0-0 \
    libdrm2 libx11-xcb1 libxcb1 libxcomposite1 libxdamage1 libxext6 \
    libxfixes3 libxkbcommon0 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome (stable)
RUN mkdir -p /usr/share/keyrings && \
    wget -qO- https://dl.google.com/linux/linux_signing_key.pub \
      | gpg --dearmor -o /usr/share/keyrings/google-linux-signing-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-linux-signing-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
      > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y --no-install-recommends google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Python deps first (better caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Env so Selenium can find Chrome (your code already sets headless)
ENV CHROME_BIN=/usr/bin/google-chrome

# Render sets $PORT; bind gunicorn to it
# Use sh -c so $PORT gets expanded
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT app:app"]
