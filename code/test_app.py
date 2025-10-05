import numpy as np
import pickle
import sys

# Import the class BEFORE unpickling
from class_implementation import RidgeRegression
from app import handle_prediction_or_reset,predict_or_reset_logic

# Tell pickle that '__main__' refers to our class_implementation module
#sys.modules["__main__"] = sys.modules["class_implementation"]

# Now load the model safely
#model = pickle.load(open("code/best_ridge_model.pkl", "rb"))

import os
import pickle

BASE_DIR = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(BASE_DIR, "best_ridge_model.pkl"), "rb"))

# Load other components
scaler = pickle.load(open("code/scaler.pkl", "rb"))
defaults = pickle.load(open("code/defaults.pkl", "rb"))


def predict_car_price(sample: np.ndarray) -> int:
    predicted_class = model.predict(sample)
    return int(predicted_class[0])


def fill_missing_values(transmission, max_power):
    if transmission is None:
        probs = defaults["transmission_ratio"]
        transmission = np.random.choice(list(probs.keys()), p=list(probs.values()))
    if max_power is None:
        max_power = defaults["mean_max_power"]
    return transmission, max_power


# Test that the model loads and predicts correctly.
def test_model_loads_and_predicts():
    """Test if the model loads and predicts correctly."""
    sample = np.array([[1,0, 120]])
    pred = predict_car_price(sample)
    assert isinstance(pred, int)
    assert pred in [0, 1, 2, 3]  # expected class range

# Test that missing values are correctly filled.
def test_fill_missing_values_returns_valid_defaults():
    
    transmission, max_power = fill_missing_values(None, None)
    assert transmission in [0, 1]
    assert isinstance(max_power, (float, int))

import pytest
from dash.testing.application_runners import import_app

# Make dash_app a proper pytest fixture
@pytest.fixture
def dash_app():
    # Import your app
    app = import_app("app")  # assumes app.py in same folder
    return app

def test_routing_callback(dash_app):
    # Test routing callback directly
    from app import display_page, main_page, prediction_page
    assert display_page("/") == main_page
    assert display_page("/predict") == prediction_page
    assert display_page("/unknown") == main_page


def test_prediction_logic_submission():
    
    transmission, max_power, output = predict_or_reset_logic("submit-btn", 0, 120)
    assert transmission == 0
    assert max_power == 120
    assert "Classified Car Category:" in output

def test_prediction_logic_reset():
    
    transmission, max_power, output = predict_or_reset_logic("try-again-btn", 0, 120)
    assert transmission is None
    assert max_power is None
    assert output == ""
