# backend/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code
COPY . .

# expose port
EXPOSE 7070

# run the Flask app
CMD ["python", "app.py"]
