FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --only-binary :all: -r requirements.txt

COPY . .

# Переобучаем модель (poker_hands_fixed.csv должен быть в репозитории)
RUN python retrain_model.py

CMD ["python", "app.py"]