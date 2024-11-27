import argparse
import importlib
import logging
import os
import pickle

import mlflow

# import numpy as np
import pandas as pd
import setLogging

data_ingestion = importlib.import_module(
    "house_price_prediction.data_ingestion"
)
training = importlib.import_module("house_price_prediction.training")


def train(dataset_folder, model_output_folder, parent_run_id=None):
    # with mlflow.start_run(
    #     run_name="Model Training", nested=True, parent_run_id=parent_run_id
    # ) as run:
    #     run_id = run.info.run_id
    logger = logging.getLogger(__name__)
    # logger.info(f"MLflow run ID: {run_id}")
    logger.info("Starting training process...")

    # Pipeline
    housing_prepared = pd.read_csv(os.path.join(dataset_folder, "X_train.csv"))
    housing_labels = pd.read_csv(os.path.join(dataset_folder, "y_train.csv"))
    housing_labels = housing_labels.values.ravel()

    print("Linear Reg")
    # Train models
    lin_reg = training.train_linear_regression(
        housing_prepared, housing_labels
    )

    print("Decision Tree")
    tree_reg = training.train_decision_tree(housing_prepared, housing_labels)

    print("Random F")
    forest_reg, f_cvres = training.train_random_forest(
        housing_prepared, housing_labels
    )
    """
        for mean_score, params in zip(
            f_cvres["mean_test_score"], f_cvres["params"]
        ):
            # Print to check the results
            # print(np.sqrt(-mean_score), params)

            # Log the hyperparameters as params in MLflow
            # Start a nested run for each hyperparameter combination

            with mlflow.start_run(nested=True):
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

                    # Log the mean score as a metric in MLflow
                    mlflow.log_metric("mean_score", np.sqrt(-mean_score))"""
    # mlflow.log_param(
    #     "random_forest_n_estimators",
    #     forest_reg.get_params()["n_estimators"],
    # )
    # mlflow.log_param(
    #     "random_forest_max_depth", forest_reg.get_params()["max_depth"]
    # )

    print("Grid Search Random F")
    # Perform grid search for Random Forest
    best_forest_reg, bf_cvres = training.grid_search_random_forest(
        housing_prepared, housing_labels
    )
    # Start a nested run for each hyperparameter combination
    """
        with mlflow.start_run(nested=True):
            for mean_score, params in zip(
                bf_cvres["mean_test_score"], bf_cvres["params"]
            ):
                # Print to check the results
                # print(np.sqrt(-mean_score), params)

                # Log the hyperparameters as params in MLflow
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

                    # Log the mean score as a metric in MLflow
                    mlflow.log_metric("mean_score", np.sqrt(-mean_score))"""

    if not os.path.exists(model_output_folder):
        os.makedirs(model_output_folder)

    # Save the models
    with open(
        os.path.join(model_output_folder, "linear_regression_model.pkl"),
        "wb",
    ) as f:
        pickle.dump(lin_reg, f)
    with open(
        os.path.join(model_output_folder, "decision_tree_model.pkl"), "wb"
    ) as f:
        pickle.dump(tree_reg, f)
    with open(
        os.path.join(model_output_folder, "random_forest_model.pkl"), "wb"
    ) as f:
        pickle.dump(forest_reg, f)
    with open(
        os.path.join(model_output_folder, "best_random_forest_model.pkl"),
        "wb",
    ) as f:
        pickle.dump(best_forest_reg, f)

    mlflow.log_param("dataset_folder", dataset_folder)
    mlflow.log_param("model_output_folder", model_output_folder)
    mlflow.log_param(
        "linear_regression_fit_intercept",
        lin_reg.get_params()["fit_intercept"],
    )
    mlflow.log_param(
        "decision_tree_criterion", tree_reg.get_params()["criterion"]
    )
    mlflow.log_param(
        "decision_tree_max_depth", tree_reg.get_params()["max_depth"]
    )
    mlflow.log_artifact(
        os.path.join(model_output_folder, "linear_regression_model.pkl")
    )
    mlflow.log_artifact(
        os.path.join(model_output_folder, "decision_tree_model.pkl")
    )
    mlflow.log_artifact(
        os.path.join(model_output_folder, "random_forest_model.pkl")
    )
    mlflow.log_artifact(
        os.path.join(model_output_folder, "best_random_forest_model.pkl")
    )

    if parent_run_id:
        mlflow.log_param(
            "linear_regression_fit_intercept",
            lin_reg.get_params()["fit_intercept"],
        )
        mlflow.log_param(
            "decision_tree_criterion",
            tree_reg.get_params()["criterion"],
        )
        mlflow.log_param(
            "decision_tree_max_depth",
            tree_reg.get_params()["max_depth"],
        )
        mlflow.log_param(
            "dataset_folder",
            dataset_folder,
        )
        mlflow.log_param(
            "model_output_folder",
            model_output_folder,
        )
        mlflow.log_artifact(
            os.path.join(model_output_folder, "linear_regression_model.pkl"),
            run_id=parent_run_id,
        )
        mlflow.log_artifact(
            os.path.join(model_output_folder, "decision_tree_model.pkl"),
            run_id=parent_run_id,
        )
        mlflow.log_artifact(
            os.path.join(model_output_folder, "random_forest_model.pkl"),
            run_id=parent_run_id,
        )
        mlflow.log_artifact(
            os.path.join(model_output_folder, "best_random_forest_model.pkl"),
            run_id=parent_run_id,
        )

    logger.info(f"Models saved to {model_output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train machine learning models."
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        help="Folder containing the dataset",
        required=True,
    )
    parser.add_argument(
        "--model_output_folder",
        type=str,
        help="Folder to save the trained models",
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
    train(args.dataset_folder, args.model_output_folder, args.parent_run_id)
