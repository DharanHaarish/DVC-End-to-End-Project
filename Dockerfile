FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .git/ ./.git/
COPY .dvc/ ./.dvc/
COPY .dvcignore ./.dvcignore
COPY dvc.yaml .
COPY dvc.lock .
COPY params.yaml .

COPY data/raw.dvc ./data/
COPY models.dvc .
COPY metrics.json.dvc .
COPY src/ ./src/

COPY dvc_e2e.json .

ENV GOOGLE_APPLICATION_CREDENTIALS=/app/dvc_e2e.json
ENV PYTHONUNBUFFERED=1

CMD dvc pull && dvc repro && echo "" && echo "Final Metrics:" && cat metrics.json && echo ""