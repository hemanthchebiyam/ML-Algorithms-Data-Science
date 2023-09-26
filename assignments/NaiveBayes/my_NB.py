# This code is written by Hemanth Chebiyam.
# Email: hc3746@rit.edu
import pandas as pd
import numpy as np
from collections import Counter

# I have not used the hint file for this assignment.
class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array, or pd.Series, dependent variables, int or str

        # Store the unique classes in self.classes_
        self.classes_ = list(set(list(y)))

        # Calculate the prior probability P(yj) for each class
        self.class_probabilities = {}
        for c in self.classes_:
            # Calculate P(yj)
            self.class_probabilities[c] = (y == c).sum() / len(y)

        # Initialize a dictionary to store conditional probabilities P(xi|yj)
        self.feature_probs = {}

        # Calculate P(xi|yj) for each feature i and class yj
        for c in self.classes_:
            self.feature_probs[c] = {}
            for feature in X.columns:
                feature_values = X[feature].unique()
                self.feature_probs[c][feature] = {}
                for value in feature_values:
                    # Calculate P(xi = t | y = c) using Laplace smoothing
                    # (N(t,c) + alpha) / (N(c) + n(i)*alpha)
                    n_t_c = ((X[feature] == value) & (y == c)).sum()
                    n_c = (y == c).sum()
                    n_i = len(feature_values)
                    probability = (n_t_c + self.alpha) / (n_c + n_i * self.alpha)
                    self.feature_probs[c][feature][value] = probability

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str

        predictions = []
        for index, row in X.iterrows():
            # Initialize dictionaries to store class probabilities
            class_probabilities = {}
            for c in self.classes_:
                # Initialize the probability with P(yj)
                class_probabilities[c] = np.log(self.class_probabilities[c])
                for feature in X.columns:
                    value = row[feature]
                    # Add the log probability of P(xi|yj)
                    if value in self.feature_probs[c][feature]:
                        class_probabilities[c] += np.log(self.feature_probs[c][feature][value])
            # Choose the class with the highest probability as the prediction
            prediction = max(class_probabilities, key=class_probabilities.get)
            predictions.append(prediction)
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str

        probs = []
        for index, row in X.iterrows():
            class_log_probabilities = {}
            for c in self.classes_:
                class_log_probabilities[c] = np.log(self.class_probabilities[c])
                for feature in X.columns:
                    value = row[feature]
                    if value in self.feature_probs[c][feature]:
                        class_log_probabilities[c] += np.log(self.feature_probs[c][feature][value])

            # Use log-sum-exp to normalize the log probabilities
            max_log_prob = max(class_log_probabilities.values())
            log_prob_sum = max_log_prob + np.log(
                sum(np.exp(prob - max_log_prob) for prob in class_log_probabilities.values()))

            class_probabilities = {c: np.exp(prob - log_prob_sum) for c, prob in class_log_probabilities.items()}
            probs.append(class_probabilities)
        return pd.DataFrame(probs, columns=self.classes_)



