FROM python:3.12-slim

# Install system dependencies for OpenCV (EasyOCR) and XGBoost
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .

# Convert requirements.txt from UTF-16 to UTF-8 if necessary
RUN python -c "import codecs; content = codecs.open('requirements.txt', 'r', 'utf-16').read(); open('requirements.txt', 'w', 'utf-8').write(content)" || true

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
