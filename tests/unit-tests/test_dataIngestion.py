import importlib
import os

import pandas as pd
import pytest

# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline

data_ingestion = importlib.import_module(
    "house_price_prediction.data_ingestion"
)


def test_fetch_housing_data():
    data_ingestion.fetch_housing_data()

    # Check if the file was fetched and extracted correctly
    assert os.path.exists(
        "datasets/housing/"
    ), "The housing.csv file does not exist in the directory."
    assert os.path.isfile("datasets/housing/housing.csv")


def test_load_housing_data():

    housing = data_ingestion.load_housing_data()
    assert isinstance(housing, pd.DataFrame)

    # Check if the DataFrame is loaded correctly
    expected_columns = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "median_house_value",
        "ocean_proximity",
    ]
    assert (
        list(housing.columns) == expected_columns
    ), "The DataFrame columns do not match the expected columns."


# def test_pipeline():
#     sample_housing = pd.DataFrame(
#         {
#             "longitude": [-122.23, -122.22, -122.24],
#             "latitude": [37.88, 37.86, 37.85],
#             "housing_median_age": [41, 21, 52],
#             "total_rooms": [880, 7099, 1467],
#             "total_bedrooms": [129, 1106, 190],
#             "population": [322, 2401, 496],
#             "households": [126, 1138, 177],
#             "median_income": [8.3252, 8.3014, 7.2574],
#             "median_house_value": [452600, 358500, 352100],
#             "ocean_proximity": ["NEAR BAY", "NEAR BAY", "NEAR BAY"],
#         }
#     )
#     pipeline = data_ingestion.prepare_pipeline(sample_housing)

#     assert isinstance(
#         pipeline, ColumnTransformer
#     ), "The pipeline should be a ColumnTransformer."
#     assert "num" in [
#         step[0] for step in pipeline.transformers
#     ], "Numerical pipeline missing."
#     assert "cat" in [
#         step[0] for step in pipeline.transformers
#     ], "Categorical pipeline missing."

#     num_pipeline = pipeline.transformers[0][1]
#     assert isinstance(
#         num_pipeline, Pipeline
#     ), "Numerical part of the pipeline should be a sklearn Pipeline."


# Run the tests
if __name__ == "__main__":
    pytest.main()
