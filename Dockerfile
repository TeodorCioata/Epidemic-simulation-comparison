
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app

WORKDIR /app

COPY pyproject.toml README.md ./
COPY episim/ ./episim/
COPY main.py ./
COPY countermeasures.py ./
COPY run_experiments.py ./

RUN pip install --no-cache-dir .

COPY frontend/ ./frontend/

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
