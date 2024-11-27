import logging

import numpy as np
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of a regression model
    using Root Mean Squared Error (RMSE).

    Parameters
    ----------
    model : sklearn.base.RegressorMixin
        The trained model that implements the `predict` method.
    X_test : numpy.ndarray or pandas.DataFrame
        The test features used for making predictions.
    y_test : numpy.ndarray or pandas.Series
        The true target values for the test set.

    Returns
    -------
    float
        The Root Mean Squared Error (RMSE) of the model on the test set.
    """
    logger.info(f"Evaluating the model {model}")
    final_predictions = model.predict(X_test)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    return final_rmse
