# import subprocess

import ingest_data
import mlflow
import score
import train

remote_server_uri = "mlruns/"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)


def main():
    exp_name = "Housing Price Prediction"
    mlflow.set_experiment(exp_name)
    print(mlflow.get_tracking_uri())

    with mlflow.start_run(
        run_name="House Price Prediction Pipeline"
    ) as main_run:
        parent_run_id = main_run.info.run_id
        print(f"Parent run id: {parent_run_id}")

        # Data Ingestion
        with mlflow.start_run(
            run_name="Data Ingestion", nested=True, parent_run_id=parent_run_id
        ) as ingestion_run:
            print(f"Starting Data Ingestion {ingestion_run.info.run_id}")

            data_output_folder = "./data"
            mlflow.log_param("data_output_folder", data_output_folder)
            ingest_data.ingest_data(data_output_folder=data_output_folder)
            print("Completed Data Ingestion")

        # Model Training
        with mlflow.start_run(
            run_name="Model Training", nested=True, parent_run_id=parent_run_id
        ) as training_run:
            dataset_folder = "./data"
            model_output_folder = "./models"
            print(f"Starting Model Training {training_run.info.run_id}")
            mlflow.log_param("dataset_folder", dataset_folder)
            mlflow.log_param("model_output_folder", model_output_folder)
            train.train(
                dataset_folder=dataset_folder,
                model_output_folder=model_output_folder,
            )
            print("Completed Model Training")

        # Model Scoring
        with mlflow.start_run(
            run_name="Model Scoring", nested=True, parent_run_id=parent_run_id
        ) as scoring_run:
            model_folder = "./models"
            output_folder = "./results"
            dataset_folder = "./data"
            print(f"Starting Model Scoring {scoring_run.info.run_id}")
            mlflow.log_param("model_folder", model_folder)
            mlflow.log_param("dataset_folder", dataset_folder)
            mlflow.log_param("output_folder", output_folder)
            score.score(
                model_folder=model_folder,
                dataset_folder=dataset_folder,
                output_folder=output_folder,
            )
            print("Completed Model Scoring")


"""
def main():
    exp_name = "Housing Price Prediction"
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(
        run_name="House Price Prediction Pipeline"
    ) as main_run:
        parent_run_id = main_run.info.run_id
        print(f"Parent run id: {parent_run_id}")

        # Data Ingestion
        with mlflow.start_run(
            run_name="Data Ingestion", nested=True, parent_run_id=parent_run_id
        ):
            print("Starting Data Ingestion")
            data_output_folder = "./data"
            mlflow.log_param("data_output_folder", data_output_folder)
            ingest_data.ingest_data(data_output_folder=data_output_folder)
            print("Completed Data Ingestion")

        # subprocess.run(
        #     [
        #         "python",
        #         "ingest_data.py",
        #         "--data_output_folder",
        #         "./data",
        #         "--parent_run_id",
        #         parent_run_id,
        #     ]
        # )

        # Model Training
        with mlflow.start_run(
            run_name="Model Training", nested=True, parent_run_id=parent_run_id
        ):
            dataset_folder = "./data"
            model_output_folder = "./models"
            print("Starting Model Training")
            mlflow.log_param("dataset_folder", dataset_folder)
            mlflow.log_param("model_output_folder", model_output_folder)
            train.train(
                dataset_folder=dataset_folder,
                model_output_folder=model_output_folder,
            )
            print("Completed Model Training")

        # subprocess.run(
        #     [
        #         "python",
        #         "train.py",
        #         "--dataset_folder",
        #         "./data",
        #         "--model_output_folder",
        #         "./models",
        #         "--parent_run_id",
        #         parent_run_id,
        #     ]
        # )

        # # Model Scoring
        with mlflow.start_run(
            run_name="Model Scoring", nested=True, parent_run_id=parent_run_id
        ):
            model_folder = "./models"
            output_folder = "./results"
            dataset_folder = "./data"
            print("Starting Model Scoring")
            mlflow.log_param("model_folder", model_folder)
            mlflow.log_param("dataset_folder", dataset_folder)
            mlflow.log_param("output_folder", output_folder)
            score.score(
                model_folder=model_folder,
                dataset_folder=dataset_folder,
                output_folder=output_folder,
            )
            print("Completed Model Scoring")

        # subprocess.run(
        #     [
        #         "python",
        #         "score.py",
        #         "--model_folder",
        #         "./models",
        #         "--dataset_folder",
        #         "./data",
        #         "--output_folder",
        #         "./results",
        #         "--parent_run_id",
        #         parent_run_id,
        #     ]
        # )
"""

if __name__ == "__main__":
    main()
