FROM python:3.12-slim

WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной код
COPY . .

# Переобучаем модель в среде контейнера
RUN python retrain_model.py

# Запускаем бота
CMD ["python", "app.py"]