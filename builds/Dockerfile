# --- STAGE 1: Builder ---
# This stage installs all dependencies and builds our Python environment.
FROM python:3.9-slim-bullseye AS builder

# Set environment variables to prevent interactive prompts during installation
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    DEBIAN_FRONTEND=noninteractive

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install system-level build tools BEFORE installing python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
# --- OPTIMIZATION: Install CPU-only versions of torch and torchvision ---
# This is the single biggest size reduction.
RUN pip install "numpy<2" torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt


# --- STAGE 2: Final Production Image ---
# This stage creates the final, lean image.
FROM python:3.9-slim-bullseye

# Set the working directory
WORKDIR /app

# Install ONLY the necessary runtime system dependencies.
# No build tools are included, making the image smaller and more secure.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libx264-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application code
COPY . .

RUN mkdir -p /app/assets/predictions
RUN mkdir -p /app/assets/uploads

# Set the PATH to include the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port the app runs on
EXPOSE 8000

# The command to run when the container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
