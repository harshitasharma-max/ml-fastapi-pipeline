import pandas as pd
from app.model import LinearRegressionScratch

model = None

def train_model(file_path):

    global model

    data = pd.read_csv(file_path)

    X = data.iloc[:, :-1].values.tolist()
    y = data.iloc[:, -1].values.tolist()

    model = LinearRegressionScratch()

    model.fit(X, y)

    return "Model trained successfully"


def predict(features):

    global model

    if model is None:
        return "Model not trained"

    prediction = model.predict([features])

    return prediction[0]
