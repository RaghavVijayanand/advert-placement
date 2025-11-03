FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (use CPU-friendly runtimes for Linux)
RUN pip install --no-cache-dir \
    opencv-python>=4.8.0 \
    pillow>=10.0.0 \
    numpy>=1.24.0 \
    scikit-image>=0.21.0 \
    scipy>=1.11.0 \
    paddleocr>=2.7.0 \
    timm>=0.9.0 \
    torch>=2.2.0 \
    torchvision>=0.17.0 \
    lightgbm>=4.0.0 \
    onnxruntime>=1.16.0 \
    pyyaml

# Copy application code
COPY . .

# Expose port 8080 (adjust if needed)
EXPOSE 8080

# Default command
CMD ["python", "main.py"]
