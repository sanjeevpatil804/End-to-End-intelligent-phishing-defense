FROM python:3.10-slim-bullseye
WORKDIR /app

# Copy application code first (needed for -e . in requirements)
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on
EXPOSE 8080

CMD ["python3", "app.py"]