
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gcc \
      libgl1-mesa-glx \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for Flask
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
