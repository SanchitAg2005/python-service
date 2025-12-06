FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by insightface + ONNXRuntime
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install insightface with environment variable to skip building 3D mesh
ENV INSIGHTFACE_SKIP_CYTHON=1

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
