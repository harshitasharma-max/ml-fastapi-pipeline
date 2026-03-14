class LinearRegressionScratch:

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):

        m = len(X)
        n = len(X[0])

        self.weights = [0] * n
        self.bias = 0

        for _ in range(self.epochs):

            dw = [0] * n
            db = 0

            for i in range(m):

                pred = self.predict_row(X[i])
                error = pred - y[i]

                for j in range(n):
                    dw[j] += error * X[i][j]

                db += error

            for j in range(n):
                self.weights[j] -= self.lr * dw[j] / m

            self.bias -= self.lr * db / m


    def predict_row(self, x):

        pred = self.bias

        for i in range(len(x)):
            pred += self.weights[i] * x[i]

        return pred


    def predict(self, X):

        predictions = []

        for x in X:
            predictions.append(self.predict_row(x))

        return predictions
