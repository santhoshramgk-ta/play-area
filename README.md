# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.



## Installation

1. **Clone the repository**:
    ```bash
    git clone git@github.com:santhoshramgk-ta/mle-training.git
    cd your-repo
    ```

2. **Set up the environment**:
    - Ensure you have Miniconda installed. If not, you can download and install it from here.
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    - Create and activate the environment:
    ```bash
    conda env create -f env.yml
    conda activate mle-dev
    ```

3. **Install the package**:
    ```bash
    pip install -e .
    ```

## Running the Script

To run the main script, use:
bash
python scripts/ingest_data.py --data_output_folder ./data --log-level INFO --log-path ./logs/ingest_data.log

python scripts/train.py --dataset_folder ./data --model_output_folder ./models --log-level INFO --log-path ./logs/train.log

python scripts/score.py --model_folder ./models --dataset_folder ./data --output_folder ./results --log-level INFO --log-path ./logs/score.log

# Fixing Z

