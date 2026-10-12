
FROM python:3.13-slim

WORKDIR /app

# Set the model version
ENV MODEL_VERSION=v0.1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY train.py .
COPY app/ ./app/

# Train and save the model inside Docker
RUN python train.py

# Expose the required port
EXPOSE 80

# Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
