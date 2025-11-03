# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file (use Docker-specific requirements if available)
COPY requirements.docker.txt requirements.txt* ./

# Install Python dependencies
RUN if [ -f requirements.docker.txt ]; then \
        pip install --no-cache-dir -r requirements.docker.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads outputs/web data/backgrounds data/logos data/products

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Expose port 8080 (Cloud Run standard) and 5000 (Flask default)
EXPOSE 8080
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run the application
# Use gunicorn for production (you can switch between CMD options)
# CMD ["python", "app.py"]
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]
