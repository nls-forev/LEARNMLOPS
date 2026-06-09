FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Copy dependency list first (better Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY src/ src/
COPY output/model.onnx output/model.onnx

# Start FastAPI
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
