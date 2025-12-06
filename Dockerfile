FROM python:3.10-slim

WORKDIR /app

# Install system build deps (gcc, g++, python headers) and runtime libs
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY requirements.txt .

# these env vars reduce optional heavy features
ENV INSIGHTFACE_SKIP_CYTHON=1
ENV INSIGHTFACE_DISABLE_TRT=1
ENV OMP_NUM_THREADS=1
ENV OMP_WAIT_POLICY=PASSIVE
ENV PIP_NO_BUILD_ISOLATION=0

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
