FROM python:3.12-slim

WORKDIR /app

# Устанавливаем системные зависимости для компиляции некоторых пакетов,
# а также setuptools и wheel (необходимы для сборки)
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && apt-get clean

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Переобучаем модель (чтобы она была совместима с окружением)
RUN python retrain_model.py

CMD ["python", "app.py"]