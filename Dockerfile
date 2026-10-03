# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create data directory for mounting
RUN mkdir -p /app/data

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY api.py similarity.py md_storage.py ./

# Expose the port the app runs on
EXPOSE 9009

# Command to run the application
CMD ["python", "api.py"] 