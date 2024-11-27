# Use the official Miniforge3 image as the base image
FROM condaforge/miniforge3

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment file
COPY ../env.yml .

# Create the conda environment
RUN conda env create -f env.yml

# Activate the environment and ensure it's activated in future RUN commands
SHELL ["conda", "run", "-n", "mle-dev", "/bin/bash", "-c"]

COPY ../. .
# Install MLflow
RUN pip install setuptools build wheel mlflow \
    && python -m build \
    && pip install dist/*.whl

# Expose the port for MLflow
EXPOSE 5000

# Command to run your script with MLflow
CMD ["conda", "run", "-n", "mle-dev", "bash", "-c", "mlflow server --host 0.0.0.0 --port 5000 & python scripts/main.py --data_folder ./data --model_folder ./model --result_folder ./results --log_path ./logs"]