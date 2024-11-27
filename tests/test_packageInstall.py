import importlib

import pytest


def test_package_installation():
    try:
        house_price_prediction = importlib.import_module(
            "house_price_prediction"
        )

        assert house_price_prediction is not None
    except ImportError:
        pytest.fail("Package is not installed properly")
