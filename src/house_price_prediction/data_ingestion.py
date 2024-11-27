import logging
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration class for housing data.

    Attributes
    ----------
    DOWNLOAD_ROOT : str
        The root URL for downloading data.
    HOUSING_PATH : str
        Local path to store housing data.
    HOUSING_URL : str
        Complete URL to fetch the housing dataset.
    """

    DOWNLOAD_ROOT = (
        "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    )
    HOUSING_PATH = os.path.join("datasets", "housing")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# Usage example
config = Config()


def fetch_housing_data(
    housing_url=config.HOUSING_URL, housing_path=config.HOUSING_PATH
):
    """
    Fetch housing data from the specified URL and extract it to the
    specified path.

    Parameters
    ----------
    housing_url : str, optional
        The URL to download the housing data from
        (default is `config.HOUSING_URL`).
    housing_path : str, optional
        The local path where the housing data will be stored
        (default is `config.HOUSING_PATH`).

    Returns
    -------
    None
    """
    logger.info(f"Fetching housing data into {housing_path}")
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)


def load_housing_data(housing_path=config.HOUSING_PATH):
    """
    Load the housing data from the CSV file into a Pandas DataFrame.

    Parameters
    ----------
    housing_path : str, optional
        The path where the housing data is stored
        (default is `config.HOUSING_PATH`).

    Returns
    -------
    pandas.DataFrame
        The loaded housing data.
    """
    logger.info(f"Loading housing data from {housing_path}")
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def stratified_split(housing):
    """
    Perform a stratified split of the housing data based on income categories.

    Parameters
    ----------
    housing : pandas.DataFrame
        The housing data to split.

    Returns
    -------
    tuple of pandas.DataFrame
        The stratified train and test datasets.
    """
    logger.info("Performing stratified split on the dataset")
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    test_set = housing.sample(frac=0.2, random_state=42)
    # Compare income category proportions
    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


class FeatureAdder(BaseEstimator, TransformerMixin):
    """
    Custom transformer that adds new features to the dataset.

    The new features are:
    - rooms_per_household: ratio of total rooms to households
    - population_per_household: ratio of population to households
    - bedrooms_per_room: ratio of total bedrooms to total rooms
    """

    def __init__(self):
        self.total_rooms_idx = None
        self.total_bedrooms_idx = None
        self.population_idx = None
        self.households_idx = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.total_rooms_idx = list(X.columns).index("total_rooms")
            self.total_bedrooms_idx = list(X.columns).index("total_bedrooms")
            self.population_idx = list(X.columns).index("population")
            self.households_idx = list(X.columns).index("households")
        else:
            self.total_rooms_idx = 3
            self.total_bedrooms_idx = 4
            self.population_idx = 5
            self.households_idx = 6

        return self

    def transform(self, X):
        """
        Transform method to add new features to the dataset.

        Parameters:
        X : array-like, shape [n_samples, n_features]
            The data to transform.

        Returns:
        X_transformed : array-like, shape [n_samples, n_features + 3]
            The transformed data with new features added.
        """
        if isinstance(X, pd.DataFrame):
            rooms_per_household = (
                X.iloc[:, self.total_rooms_idx]
                / X.iloc[:, self.households_idx]
            )
            population_per_household = (
                X.iloc[:, self.population_idx] / X.iloc[:, self.households_idx]
            )
            bedrooms_per_room = (
                X.iloc[:, self.total_bedrooms_idx]
                / X.iloc[:, self.total_rooms_idx]
            )

        else:
            rooms_per_household = (
                X[:, self.total_rooms_idx] / X[:, self.households_idx]
            )
            population_per_household = (
                X[:, self.population_idx] / X[:, self.households_idx]
            )
            bedrooms_per_room = (
                X[:, self.total_bedrooms_idx] / X[:, self.total_rooms_idx]
            )

        X_new = np.c_[
            X, rooms_per_household, population_per_household, bedrooms_per_room
        ]
        return X_new

    def get_feature_names_out(self, input_features=None):
        """
        Returns the names of the output features, including the newly added
        features.

        Parameters
        ----------
        input_features : list of str, optional
            The names of the input features. If None, uses default column
            names.

        Returns
        -------
        list of str
            The names of the output features.
        """
        if input_features is None:
            # Default feature names if input_features is not provided
            input_features = [
                f"feature_{i}" for i in range(self.total_rooms_idx + 4)
            ]
        else:
            input_features = list(input_features)

        # Add new feature names
        new_features = [
            "rooms_per_household",
            "bedrooms_per_room",
            "population_per_household",
        ]
        return input_features + new_features


def prepare_pipeline(housing=None):
    """
    Prepare a data processing pipeline for housing data.

    Parameters:
    housing : DataFrame, optional
        The housing data to determine numerical features. If not provided,
        default numerical features are used.

    Returns:
    full_pipeline : ColumnTransformer
        The full data processing pipeline.
    """
    if housing is not None:
        num_features = housing.select_dtypes(
            include=["float64", "int64"]
        ).columns.tolist()

        # Get indices for specific columns to pass to FeatureAdder

    else:
        num_features = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
        ]

    cat_features = ["ocean_proximity"]

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("feature_adder", FeatureAdder()),
            ("scaler", StandardScaler()),
        ]
    )

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_features),
            ("cat", OneHotEncoder(), cat_features),
        ]
    )
    return full_pipeline


def prepare_data_with_pipeline(housing, pipeline):
    """
    Prepare the housing data using the provided pipeline.

    Parameters:
    housing : DataFrame
        The housing data to transform.
    pipeline : ColumnTransformer
        The data processing pipeline.

    Returns:
    transformed_data : array-like
        The transformed housing data.
    """
    return pipeline.fit_transform(housing)


def income_cat_proportions(data):
    """
    Calculate the proportions of each income category in the dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        The housing data containing the 'income_cat' column.

    Returns
    -------
    pandas.Series
        The proportions of each income category.
    """
    return data["income_cat"].value_counts() / len(data)


def plot_housing_data(housing):
    """
    Plot the housing data showing the geographic distribution of the data.

    Parameters
    ----------
    housing : pandas.DataFrame
        The housing data containing 'longitude' and 'latitude' columns.

    Returns
    -------
    None
    """
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    # plt.show()


def calculate_correlation_matrix(housing):
    """
    Calculate the correlation matrix for the numerical features in the
    housing data.

    Parameters
    ----------
    housing : pandas.DataFrame
        The housing data.

    Returns
    -------
    pandas.Series
        The correlation of each numerical feature with
        the 'median_house_value'.
    """
    corr_matrix = housing.corr(numeric_only=True)
    return corr_matrix["median_house_value"].sort_values(ascending=False)
