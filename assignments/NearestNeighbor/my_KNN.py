import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        self.X = X
        self.y = y

    def dist(self,x):
        # Calculate distances of training data to a single input data point (distances from self.X to x)
        # Output np.array([distances to x])
        if self.metric == "minkowski":
            distances = np.linalg.norm(self.X - x, ord=self.p, axis=1)


        elif self.metric == "euclidean":
            distances = np.linalg.norm(self.X - x, ord=2, axis=1)


        elif self.metric == "manhattan":
            distances = np.linalg.norm(self.X - x, ord=1, axis=1)


        elif self.metric == "cosine":
            x_normalized = x / np.linalg.norm(x)
            X_normalized = self.X / np.linalg.norm(self.X, axis=1, keepdims=True)
            distances = np.dot(X_normalized, x_normalized)


        else:
            raise Exception("Unknown criterion.")
        return distances

    def k_neighbors(self,x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors) e.g. {"Class A":3, "Class B":2}
        distances = self.dist(x)
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        nearest_labels = [self.y[i] for i in nearest_indices]
        return Counter(nearest_labels)

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        probs = []
        try:
            X_feature = X[self.X.columns]
        except:
            raise Exception("Input data mismatch.")

        for x in X_feature.to_numpy():
            neighbors = self.k_neighbors(x)
            # Calculate the probability of data point x belonging to each class
            # e.g. prob = {"2": 1/3, "1": 2/3}
            prob = {label: count / self.n_neighbors for label, count in neighbors.items()}
            probs.append(prob)

        # Fill missing classes with 0 probability
        for prob in probs:
            for label in self.classes_:
                prob[label] = prob.get(label, 0)

        probs = pd.DataFrame(probs, columns=self.classes_)
        print(probs)
        return probs