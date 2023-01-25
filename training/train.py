import pandas as pd
import logging
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

RANDOM_STATE = 28

def setup_logger():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

def create_training_set():
    """
    Read the data from the filesystem and prepare a training set
    The training set includes train and test data
    """
    logging.info("Preparing data for train and test")
    df = pd.read_csv("data/houses.csv", index_col="date")

    target = "price"
    prediction = "predicted_price"
    datetime = "date"

    features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
        "waterfront", "view", "condition", "grade", "yr_built"]

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    return X_train, X_test, y_train, y_test

def create_model():
    logging.info("Creating Random Forest Regressor model")
    model = RandomForestRegressor(random_state=RANDOM_STATE, verbose=1)
    return model

def train(model, X_train, y_train):
    logging.info("Training model")
    model.fit(X_train, y_train)

def evaluate(model, X_test, y_test):
    logging.info("Evaluating model on test set")
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    logging.info(f"R-Squared score: {r2}")

def save_model(model):
    with open("models/model.pkl","wb") as f:
        pickle.dump(model, f)

def pipeline():
    X_train, X_test, y_train, y_test = create_training_set()
    model = create_model()
    train(model, X_train, y_train)
    evaluate(model, X_test, y_test)
    save_model(model)

if __name__ == "__main__":
    setup_logger()
    pipeline()
