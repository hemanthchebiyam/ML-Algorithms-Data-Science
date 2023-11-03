# This code is written by Hemanth Chebiyam.
# Email: hc3746@rit.edu

import pandas as pd
import numpy as np
import pdb

# I did not use the hint file
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
        data = np.concatenate([data, np.ones(shape=(data.shape[0], 1))], axis=1)
        d = data.shape[1]
        # Initialize weights as all zeros
        self.w = np.array([0.0] * d)
        self.w0 = 0.0
        # write your code below
        for i in range(self.max_iter):

            if self.shuffle:
                inds = np.random.permutation(range(len(data)))
            else:
                inds = np.array(range(len(data)))

            num_inds = len(inds)

            while num_inds > 0:

                num_inds -= self.batch_size

                if num_inds > self.batch_size:
                    b_inds = inds[:self.batch_size]
                    inds = inds[self.batch_size:]

                else:
                    b_inds = inds

                Xb, yb = data[b_inds], y[b_inds]
                self.batch_update(Xb, yb)

    def batch_update(self, Xb, yb):

        gradient = grad(wts=self.w, Xb=Xb, yb=yb)
        self.w -= self.learning_rate * gradient

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = f(x) = 1 / (1+exp(-(w0+w*x))}); a list of float values in [0.0, 1.0]
        # write your code below
        X = np.concatenate([X, np.ones(shape=(X.shape[0], 1))], axis=1)
        probs = sigmoid(X @ self.w)
        return probs

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list of int values in {0, 1}

        probs = self.predict_proba(X)
        predictions = [1 if prob >= 0.5 else 0 for prob in probs]
        return predictions

def grad(wts, Xb, yb):
    m = len(yb)

    return 2 * (1 / m) * Xb.T @ ((sigmoid(Xb @ wts) - yb) * sigmoid(Xb @ wts) * (1 - sigmoid(Xb @ wts)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))