FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (lightweight)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Disable heavy parts of insightface
ENV INSIGHTFACE_SKIP_CYTHON=1
ENV INSIGHTFACE_DISABLE_TRT=1
ENV OMP_NUM_THREADS=1
ENV OMP_WAIT_POLICY=PASSIVE

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
