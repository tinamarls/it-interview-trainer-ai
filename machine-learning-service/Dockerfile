FROM python:3.10-slim

# Установка зависимостей
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Пробрасываем порт, если нужен (например, Flask 5000)
EXPOSE 5001

# Запуск (если Flask)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]
