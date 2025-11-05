FROM python:3.10-slim-bullseye

WORKDIR /app

# Install system dependencies (build tools + runtime libs + awscli)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libgomp1 \
    awscli && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and setup files first (better caching)
COPY requirements.txt Setup.py ./
COPY networksecurity/__init__.py networksecurity/__init__.py

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]