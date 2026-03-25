FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variable for the run ID
ENV RUN_ID=${RUN_ID}

# Default command
CMD ["python", "check_threshold.py"]