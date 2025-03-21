# Use Python 3.9 slim image
FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Copy trained models
COPY models/ ./models/

# Create templates directory and copy HTML template
RUN mkdir -p src/templates
COPY src/templates/index.html src/templates/

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=src/web_app.py
ENV FLASK_ENV=production

# Run the application
CMD ["flask", "run", "--host=0.0.0.0"] 