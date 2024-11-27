import importlib

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

model_eval = importlib.import_module("house_price_prediction.scoring")


def test_modelEval():
    X = np.array([[1], [2], [3], [4], [5]])  # Features
    y = np.array([1, 2, 3, 4, 5])  # Labels (perfectly linear)

    # Split data into training and testing sets
    # (using 4 for training, 1 for testing)
    X_train, X_test = X[:4], X[4:]
    y_train, y_test = y[:4], y[4:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Use the evaluate_model function to compute RMSE
    rmse = model_eval.evaluate_model(model, X_test, y_test)
    assert rmse == 0


if __name__ == "__main__":
    pytest.main()
