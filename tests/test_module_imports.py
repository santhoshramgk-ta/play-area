import importlib

import pytest


def test_data_ingestion_import():
    try:
        data_ingest = importlib.import_module(
            "house_price_prediction.data_ingestion"
        )
        assert data_ingest is not None
    except ImportError:
        pytest.fail("Package is not installed properly")


def test_training_import():
    try:
        training = importlib.import_module("house_price_prediction.training")
        assert training is not None

    except ImportError:
        pytest.fail("Package is not installed properly")


def test_scoring_import():
    try:
        scoring = importlib.import_module("house_price_prediction.scoring")
        assert scoring is not None

    except ImportError:
        pytest.fail("Package is not installed properly")


# Run the tests
if __name__ == "__main__":
    pytest.main()
