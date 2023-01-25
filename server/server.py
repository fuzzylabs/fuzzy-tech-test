import json
import logging
import pickle

import numpy as np
import requests
from flask import Flask, request
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

@app.before_first_request
def load_model() -> RandomForestRegressor:
    """Load the trained model from the specified path.

    Returns:
        RandomForestRegressor: the trained model
    """
    model_path = "models/model.pkl"
    logging.info(f"Loading model in path: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded")

    return model


@app.route("/predict", methods=["POST"])
def predict() -> str:
    """Process the JSON payload and select features that can be used by the model to make predictions.

    Returns:
        str: the price prediction as a string
    """
    features = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "yr_built",
    ]

    logging.info(f"Received inference request {request.json}")
    request_data = dict(
        (f, request.json[f]) for f in features if f in request.json
    )
    features = list(request_data.values())
    features = np.array(features)
    features = features.reshape(1, -1)

    pred = model.predict(features)
    pred_price = float(pred[0])

    return str(pred[0])

if __name__ == "__main__":
    model = load_model()
    app.run(host="0.0.0.0", port="5050", debug=True)
