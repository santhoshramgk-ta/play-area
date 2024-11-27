# Use the official Miniforge3 image as the base image
FROM condaforge/miniforge3

# Copy the environment file
COPY ../. .

# Create the conda environment
RUN conda env create -f env.yml

# Activate the environment and ensure it's activated in future RUN commands
SHELL ["conda", "run", "-n", "mle-dev", "/bin/bash", "-c"]

# Install MLflow
RUN pip install mlflow

# Expose the port for MLflow
EXPOSE 5000

# Command to run your script
CMD ["conda", "run", "-n", "mle-dev", "python", "scripts/main.py", "--data_folder", "./data", "--model_folder", "./model", "--result_folder", "./results", "--log_path", "./logs"]