FROM python:3.10-slim

WORKDIR /app

# system deps for insightface + onnxruntime
RUN apt-get update && apt-get install -y \
    git \
    g++ \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# skip heavy 3D builds
ENV INSIGHTFACE_SKIP_CYTHON=1
ENV INSIGHTFACE_DISABLE_TRT=1
ENV OMP_NUM_THREADS=1

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
