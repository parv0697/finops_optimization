# Stage 1: Build stage (if any build steps were needed, not critical for this Python app)
# For Python, we mostly copy files and install dependencies.

# Stage 2: Final stage
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies that might be needed by Python packages
# e.g., for Prophet, it might need build-essential, libstdc++6 on some systems
# For now, keeping it minimal. Add if pip install fails for certain packages.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size slightly
# Ensure pip is upgraded
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./app /app/app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# Use Gunicorn for production if desired, or Uvicorn for development/simpler setup.
# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
