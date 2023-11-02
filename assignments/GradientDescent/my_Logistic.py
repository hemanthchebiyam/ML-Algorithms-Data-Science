# This code is written by Hemanth Chebiyam.
# Email: hc3746@rit.edu

import pandas as pd
import numpy as np

class my_Logistic:

    def __init__(self, learning_rate=0.1, batch_size=10, max_iter=100, shuffle=False):
        # Logistic regression: f(x) = 1 / (1+exp(-(w0+w*x))})
        # Loss function is sum (f(x)-y)**2
        # learning_rate: Learning rate for each weight update.
        # batch_size: Number of training data points in each batch.
        # max_iter: The maximum number of passes over the training data (aka epochs). Note that this is not max batches.
        # shuffle: Whether to shuffle the data in each epoch.
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.shuffle = shuffle

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables
        # y: list, np.array or pd.Series, dependent variables
        data = X.to_numpy()
        d = data.shape[1]
        # Initialize weights as all zeros
        self.w = np.array([0.0] * d)
        self.w0 = 0.0

        num_samples = data.shape[0]
        num_batches = num_samples // self.batch_size

        for _ in range(self.max_iter):
            if self.shuffle:
                # Shuffle the data if shuffle is True
                indices = np.random.permutation(num_samples)
                data = data[indices]
                y = y[indices]

            for i in range(num_batches):
                # Get the current batch
                start_idx = i * self.batch_size
                end_idx = (i + 1) * self.batch_size
                X_batch = data[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]

                # Compute predictions
                predictions = self._sigmoid(np.dot(X_batch, self.w) + self.w0)

                # Compute gradients
                grad_w = np.dot(X_batch.T, predictions - y_batch) / self.batch_size
                grad_w0 = np.sum(predictions - y_batch) / self.batch_size

                # Update weights
                self.w -= self.learning_rate * grad_w
                self.w0 -= self.learning_rate * grad_w0

    def _sigmoid(self, z):
        # Sigmoid function
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables
        # prob is a dict of prediction probabilities belonging to each category
        # return probs = f(x) = 1 / (1+exp(-(w0+w*x))); a list of float values in [0.0, 1.0]
        data = X.to_numpy()
        return self._sigmoid(np.dot(data, self.w) + self.w0)

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list of int values in {0, 1}
        probs = self.predict_proba(X)
        predictions = [1 if prob >= 0.5 else 0 for prob in probs]
        return predictions