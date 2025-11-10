FROM python:3.10-slim-bullseye
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port your app runs on
EXPOSE 8080

CMD ["python3", "app.py"]