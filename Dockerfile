FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    pip install --no-cache-dir fastapi uvicorn pandas openpyxl requests python-multipart

COPY . /app
WORKDIR /app
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]