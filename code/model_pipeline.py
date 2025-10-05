import sys
import joblib
import numpy as np
import pickle
import sys
import os
#from ridge_model import RidgeRegression

# Load the model, scaler, and defaults
from class_implementation import RidgeRegression


# Tell pickle that '__main__' refers to our class_implementation module
sys.modules["__main__"] = sys.modules["class_implementation"]

# Now load the model safely
#model = pickle.load(open("./code/best_ridge_model.pkl", "rb"))
#scaler = pickle.load(open("./code/scaler.pkl", "rb"))  # or pickle.load, both fine
#defaults = pickle.load(open("./code/defaults.pkl", "rb"))

BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "best_ridge_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
defaults_path = os.path.join(BASE_DIR, "defaults.pkl")

# Load the model, scaler, defaults
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(defaults_path, "rb") as f:
    defaults = pickle.load(f)

def predict_car_price(sample: np.ndarray) -> float:
    predicted_class = model.predict(sample)
    return int(predicted_class[0])


def fill_missing_values(transmission, max_power):
    # Handle missing categorical values of transmission
    if transmission is None:
        probs = defaults["transmission_ratio"]
        transmission = np.random.choice(list(probs.keys()), p=list(probs.values()))

    # Handle missing numeric values of max_power
    if max_power is None:
        max_power = defaults["mean_max_power"]  

    return transmission, max_power


# Example
if __name__ == "__main__":
    # Replace with actual user input
    #sample = np.array([[1, 67.05, 998.00, 2009.00]])  # example
    print("Predicted Car Price:", predict_car_price(sample))
