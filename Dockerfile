FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_api.txt .

RUN pip install --no-cache-dir --timeout 300 \
    fastapi==0.138.2 \
    uvicorn==0.49.0 \
    pydantic==2.13.4 \
    pandas==2.2.2 \
    numpy==2.2.6 \
    scikit-learn==1.5.0 \
    pyarrow==20.0.0

RUN pip install --no-cache-dir --timeout 600 \
    torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY data/models/ ./data/models/
COPY data/raw/ml-latest-small/movies.csv ./data/raw/ml-latest-small/movies.csv

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
