# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniforge
RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    && bash Miniforge3.sh -b -p "${HOME}/conda" \
    && rm Miniforge3.sh

# Set up the shell to use conda
SHELL ["bash", "-c"]

# Copy the environment file from the parent directory
COPY . .

# Source conda scripts and create the conda environment
RUN source "${HOME}/conda/etc/profile.d/conda.sh" \
    && conda env create -f env.yml

# Activate the environment and ensure it's activated in future RUN commands
# Using `conda run` to activate the environment for the next RUN commands
SHELL ["conda", "run", "-n", "mle-dev", "/bin/bash", "-c"]




# Expose the port for MLflow
EXPOSE 5000

# Command to run your script using conda's active environment
CMD ["conda", "run", "-n", "mle-dev", "python", "scripts/main.py", "--data_folder", "./data", "--model_folder", "./model", "--result_folder", "./results", "--log_path", "./logs"]
