FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=9009 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py bow.py similarity.py storage.py text.py topics.py md_service_boilerplate.py service.json index.md ./

EXPOSE 9009

CMD ["python", "api.py"]
