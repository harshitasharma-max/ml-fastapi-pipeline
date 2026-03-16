import pandas as pd

model = None


class LinearRegressionScratch:

    def __init__(self):
        self.weights = None
        self.bias = 0

    def fit(self, X, y, lr=0.00000001, epochs=1000):
        n_samples = len(X)
        n_features = len(X[0])

        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(epochs):

            for i in range(n_samples):

                prediction = sum(
                    self.weights[j] * X[i][j] for j in range(n_features)
                ) + self.bias

                error = prediction - y[i]

                for j in range(n_features):
                    self.weights[j] -= lr * error * X[i][j]

                self.bias -= lr * error

    def predict(self, X):

        predictions = []

        for row in X:
            pred = sum(
                self.weights[i] * row[i] for i in range(len(row))
            ) + self.bias

            predictions.append(pred)

        return predictions


def train_model(file_path):

    global model

    data = pd.read_csv(file_path)

    # Features = first columns
    X = data.iloc[:, :-1].values.tolist()

    # Target = last column
    y = data.iloc[:, -1].values.tolist()

    model = LinearRegressionScratch()

    model.fit(X, y)

    return "Model trained successfully"


def predict(features):

    global model

    if model is None:
        return None

    prediction = model.predict([features])

    return float(prediction[0])
