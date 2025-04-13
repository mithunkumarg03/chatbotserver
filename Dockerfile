FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y git && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "app.py"]
