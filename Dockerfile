# Use a lightweight Python image as the base
FROM python:3.11-slim-buster

# Expose the Streamlit default port
EXPOSE 8501

# Install necessary system packages and clean up to keep the image slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /Stockpredictionapp

# Copy all the application files to the working directory
COPY . /Stockpredictionapp

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point to start the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]