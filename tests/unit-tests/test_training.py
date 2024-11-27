import importlib

import numpy as np

# Convert to a pandas DataFrame for compatibility with your code (if needed)
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

training = importlib.import_module("house_price_prediction.training")


def test_train_linear_regression():

    # Test that the function completes without errors and
    #  returns a LinearRegression model
    X = np.array([[1], [2], [3], [4], [5]])  # Features
    y = np.array([3, 6, 9, 12, 15])  # Labels (y = 3 * X)

    # Convert to DataFrame for compatibility
    X = pd.DataFrame(X, columns=["feature_1"])
    model = training.train_linear_regression(X, y)

    # Check if the returned model is an instance of LinearRegression
    assert isinstance(model, LinearRegression)
    X_new = pd.DataFrame([[10]], columns=["feature_1"])
    prediction = model.predict(X_new)

    # Expected prediction is 30 (since y = 3 * x)
    expected_prediction = 30

    assert np.isclose(
        prediction, expected_prediction
    ), f"Prediction: {prediction}, Expected: {expected_prediction}"

    # Optionally, check that the model can make predictions
    predictions = model.predict(X)
    assert (
        predictions.shape[0] == X.shape[0]
    ), "Number of predictions should match number of samples"


def test_train_random_forest():
    # Test that the function completes without errors and returns
    #  a RandomForestRegressor model
    # Generate a random regression dataset with 100 samples and 5 features
    X, y = make_regression(
        n_samples=100, n_features=5, noise=0.1, random_state=42
    )

    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, 6)])
    model, cvres = training.train_random_forest(X, y)

    # Check if the returned model is an instance of RandomForestRegressor
    assert isinstance(model, RandomForestRegressor)

    # Optionally, check if the model is fitted (i.e., check if it has the
    # 'feature_importances_' attribute)
    assert hasattr(
        model, "feature_importances_"
    ), "RandomForestRegressor should have 'feature_importances_' attribute"


if __name__ == "__main__":
    pytest.main()
