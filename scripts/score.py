import argparse
import importlib
import logging
import os
import pickle

import mlflow
import pandas as pd
import setLogging

training = data_ingestion = importlib.import_module(
    "house_price_prediction.training"
)
scoring = data_ingestion = importlib.import_module(
    "house_price_prediction.scoring"
)


def score(model_folder, dataset_folder, output_folder, parent_run_id=None):
    # with mlflow.start_run(
    #     run_name="Model Scoring", nested=True, parent_run_id=parent_run_id
    # ) as run:
    #     run_id = run.info.run_id
    logger = logging.getLogger(__name__)
    # logger.info(f"MLFlow run id {run_id}")
    logger.info("Starting training process...")

    # Load the pre-trained models
    models = {}
    for model_name in [
        "linear_regression_model",
        "decision_tree_model",
        "random_forest_model",
        "best_random_forest_model",
    ]:
        with open(os.path.join(model_folder, f"{model_name}.pkl"), "rb") as f:
            models[model_name] = pickle.load(f)

    # Load the test dataset
    X_test_prepared = pd.read_csv(os.path.join(dataset_folder, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(dataset_folder, "y_test.csv"))
    y_test = y_test.values.ravel()

    # Evaluate models
    scores = {}
    for model_name, model in models.items():
        scores[model_name] = scoring.evaluate_model(
            model, X_test_prepared, y_test
        )
        logger.info(f"{model_name} RMSE: {scores[model_name]}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for model_name, model in models.items():
        scores[model_name] = scoring.evaluate_model(
            model, X_test_prepared, y_test
        )
        logger.info(f"{model_name} RMSE: {scores[model_name]}")
        mlflow.log_metric(f"{model_name}_rmse", scores[model_name])
        if parent_run_id:
            mlflow.log_metric(f"{model_name}_rmse", scores[model_name])

    # Save the results
    with open(os.path.join(output_folder, "model_scores.txt"), "w") as f:
        for model_name, score in scores.items():
            f.write(f"{model_name} RMSE: {score}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the trained models.")
    parser.add_argument(
        "--model_folder",
        type=str,
        help="Folder containing the trained models",
        required=True,
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        help="Folder containing the dataset",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Folder to save the output results",
        required=True,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level (e.g., DEBUG, INFO, WARNING)",
    )
    parser.add_argument(
        "--log-path", type=str, default=None, help="Path to save log file"
    )
    parser.add_argument(
        "--no-console-log", action="store_true", help="Disable console logging"
    )
    parser.add_argument(
        "--parent_run_id", type=str, help="Parent MLflow run ID", required=True
    )

    args = parser.parse_args()

    setLogging.setup_logging(
        args.log_level.upper(), args.log_path, not args.no_console_log
    )
    score(
        args.model_folder,
        args.dataset_folder,
        args.output_folder,
        args.parent_run_id,
    )
