# Base Image
FROM python:3.10-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install System Dependencies (Minimal for headless OpenCV)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Common Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Application Code
# We assume the context is the 'python-worker' directory
COPY . .

# Add the current directory to PYTHONPATH so imports work correctly
ENV PYTHONPATH=/app

# Expose the gRPC port
EXPOSE 50051

# Run the gRPC server
CMD ["python", "app/main.py"]
