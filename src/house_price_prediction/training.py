import logging

# import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)


def train_linear_regression(housing_prepared, housing_labels):
    """
    Train a linear regression model.

    Parameters
    ----------
    housing_prepared : numpy.ndarray or pandas.DataFrame
        The feature matrix used to train the model.
    housing_labels : numpy.ndarray or pandas.Series
        The target values for training the model.

    Returns
    -------
    sklearn.linear_model.LinearRegression
        The trained Linear Regression model.
    """
    logger.info("Training Linear Regression")
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    return lin_reg


def train_decision_tree(housing_prepared, housing_labels):
    """
    Train a decision tree regressor.

    Parameters
    ----------
    housing_prepared : numpy.ndarray or pandas.DataFrame
        The feature matrix used to train the model.
    housing_labels : numpy.ndarray or pandas.Series
        The target values for training the model.

    Returns
    -------
    sklearn.tree.DecisionTreeRegressor
        The trained Decision Tree Regressor model.
    """
    logger.info("Training Decision Tree")
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    return tree_reg


def train_random_forest(housing_prepared, housing_labels):
    """
    Train a random forest regressor using randomized search.

    Parameters
    ----------
    housing_prepared : numpy.ndarray or pandas.DataFrame
        The feature matrix used to train the model.
    housing_labels : numpy.ndarray or pandas.Series
        The target values for training the model.

    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
        The best Random Forest model obtained from the randomized search.
    """
    logger.info("Training Random Forest")
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)

    # Print cross-validation results
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    return rnd_search.best_estimator_, cvres


def grid_search_random_forest(housing_prepared, housing_labels):
    """
    Perform grid search to tune the hyperparameters of
    a random forest regressor.

    Parameters
    ----------
    housing_prepared : numpy.ndarray or pandas.DataFrame
        The feature matrix used to train the model.
    housing_labels : numpy.ndarray or pandas.Series
        The target values for training the model.

    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
        The best Random Forest model obtained from the grid search.
    """
    logger.info("Training Grid Search Random Forest")
    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
    ]
    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    # Print feature importances
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    return grid_search.best_estimator_, cvres
