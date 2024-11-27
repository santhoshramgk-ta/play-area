import argparse
import importlib
import logging
import os

import mlflow
import pandas as pd
import setLogging

data_ingestion = importlib.import_module(
    "house_price_prediction.data_ingestion"
)


def ingest_data(data_output_folder, parent_run_id=None):
    # with mlflow.start_run(
    #     run_name="Data Ingestion", nested=True, parent_run_id=parent_run_id
    # ) as run:
    #     run_id = run.info.run_id
    logger = logging.getLogger(__name__)
    logger.info("Starting data ingestion process...")

    # Download and extract data
    data_ingestion.fetch_housing_data(housing_path=data_output_folder)

    # Load the housing data
    housing = data_ingestion.load_housing_data(housing_path=data_output_folder)

    # Perform stratified split
    strat_train_set, strat_test_set = data_ingestion.stratified_split(housing)

    X_train = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    y_train = strat_train_set["median_house_value"].copy()
    pipeline = data_ingestion.prepare_pipeline(X_train)
    X_train = data_ingestion.prepare_data_with_pipeline(X_train, pipeline)

    feature_names = pipeline.get_feature_names_out()
    X_train = pd.DataFrame(X_train, columns=feature_names)
    print(X_train.columns.tolist())

    # Save the train and test data
    X_train.to_csv(
        os.path.join(data_output_folder, "X_train.csv"), index=False
    )
    y_train.to_csv(
        os.path.join(data_output_folder, "y_train.csv"), index=False
    )
    y_test = strat_test_set["median_house_value"].copy()
    X_test = strat_test_set.drop("median_house_value", axis=1)

    X_test = pipeline.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    X_test.to_csv(os.path.join(data_output_folder, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(data_output_folder, "y_test.csv"), index=False)

    logger.info(f"Training and testing data saved in {data_output_folder}")
    # logger.info(f"MLflow run ID: {run_id}")
    if parent_run_id:
        mlflow.log_artifact(
            os.path.join(data_output_folder, "X_train.csv"),
            run_id=parent_run_id,
        )
        mlflow.log_artifact(
            os.path.join(data_output_folder, "y_train.csv"),
            run_id=parent_run_id,
        )
        mlflow.log_artifact(
            os.path.join(data_output_folder, "X_test.csv"),
            run_id=parent_run_id,
        )
        mlflow.log_artifact(
            os.path.join(data_output_folder, "y_test.csv"),
            run_id=parent_run_id,
        )
    mlflow.log_artifact(os.path.join(data_output_folder, "X_train.csv"))
    mlflow.log_artifact(os.path.join(data_output_folder, "y_train.csv"))
    mlflow.log_artifact(os.path.join(data_output_folder, "X_test.csv"))
    mlflow.log_artifact(os.path.join(data_output_folder, "y_test.csv"))

    print(f"Training and testing data saved in {data_output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare the housing dataset."
    )
    parser.add_argument(
        "--data_output_folder",
        type=str,
        help="Folder to save the dataset",
        required=True,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level (e.g., DEBUG, INFO, WARNING)",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path to save log file",
        nargs="?",
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
    ingest_data(args.data_output_folder, args.parent_run_id)
