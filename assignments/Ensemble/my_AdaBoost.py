# This code is written by Hemanth Chebiyam.
# Email: hc3746@rit.edu

import pandas as pd
import numpy as np
from copy import deepcopy

# I did not use the hint file.
class my_AdaBoost:

    def __init__(self, base_estimator=None, n_estimators=50):
        # Multi-class Adaboost algorithm (SAMME)
        # alpha = ln((1-error)/error)+ln(K-1), K is the number of classes.
        # base_estimator: the base classifier class, e.g. my_DT
        # n_estimators: # of base_estimator rounds
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.estimators = [deepcopy(self.base_estimator) for i in range(self.n_estimators)]

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str

        self.classes_ = list(set(list(y)))
        k = len(self.classes_)
        n_samples, n_features = X.shape
        weights = np.ones(n_samples) / n_samples
        estimators = self.estimators
        estimator_weights = np.zeros(self.n_estimators)

        for i in range(self.n_estimators):
            # Fit the base estimator
            estimator = estimators[i]
            estimator.fit(X, y, sample_weight=weights)

            # Calculate error and estimator weight
            y_pred = estimator.predict(X)
            incorrect = (y_pred != y)
            error = np.sum(weights[incorrect])
            estimator_weight = 0.5 * np.log((1 - error) / max(error, 1e-10))
            estimator_weights[i] = estimator_weight

            # Update sample weights
            weights *= np.exp(estimator_weight * (incorrect * 2 - 1))
            weights /= np.sum(weights)

        self.estimator_weights_ = estimator_weights
        return self

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list

        class_votes = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(self.n_estimators):
            pred = self.estimators[i].predict(X)
            for j, c in enumerate(self.classes_):
                class_votes[:, j] += (pred == c) * self.estimator_weights_[i]

        predictions = [self.classes_[np.argmax(votes)] for votes in class_votes]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob: what percentage of the base estimators predict input as class C
        # prob(x)[C] = sum(alpha[j] * (base_model[j].predict(x) == C))
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)

        class_probs = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(self.n_estimators):
            pred = self.estimators[i].predict(X)
            for j, c in enumerate(self.classes_):
                class_probs[:, j] += (pred == c) * self.estimator_weights_[i]

        probs = class_probs / np.sum(class_probs, axis=1, keepdims=True)
        return pd.DataFrame(probs, columns=self.classes_)