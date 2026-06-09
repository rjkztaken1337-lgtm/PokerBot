FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Пропускаем переобучение — используем готовые файлы модели из репозитория
# RUN python retrain_model.py

CMD ["python", "app.py"]