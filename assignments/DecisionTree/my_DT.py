import pandas as pd
import numpy as np
from collections import Counter

class my_DT:

    def __init__(self, criterion="gini", max_depth=8, min_impurity_decrease=0, min_samples_split=2):
        # criterion = {"gini", "entropy"},
        # Stop training if depth = max_depth. Depth of a binary tree: the max number of edges from the root node to a leaf node
        # Only split node if impurity decrease >= min_impurity_decrease after the split
        #   Weighted impurity decrease: N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
        # Only split node with >= min_samples_split samples
        self.criterion = criterion
        self.max_depth = int(max_depth)
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = int(min_samples_split)

    def impurity(self, labels):
        # Calculate impurity (unweighted)
        # Input is a list (or np.array) of labels
        # Output impurity score
        stats = Counter(labels)
        N = float(len(labels))
        if self.criterion == "gini":
            impure = 1.0
            for label in stats:
                prob = stats[label] / N
                impure -= prob ** 2
        elif self.criterion == "entropy":
            impure = 0.0
            for label in stats:
                prob = stats[label] / N
                impure -= prob * np.log2(prob)
        else:
            raise Exception("Unknown criterion.")
        return impure

    def find_best_split(self, pop, X, labels):
        # Find the best split
        # Inputs:
        #   pop:    indices of data in the node
        #   X:      independent variables of training data
        #   labels: dependent variables of training data
        # Output: tuple(best feature to split, weighted impurity score of best split, splitting point of the feature, [indices of data in left node, indices of data in right node], [weighted impurity score of left node, weighted impurity score of right node])
        best_feature = None
        best_split_value = None
        best_left_mask = None
        best_right_mask = None
        best_left_impurity = None
        best_right_impurity = None
        current_impurity = self.impurity(labels[pop])
        for feature in X.columns:
            values = np.unique(X[feature][pop])
            for value in values:
                left_mask = X[feature][pop] <= value
                right_mask = ~left_mask
                left_impurity = self.impurity(labels[pop[left_mask]])
                right_impurity = self.impurity(labels[pop[right_mask]])
                weighted_impurity = (len(pop[left_mask]) * left_impurity + len(pop[right_mask]) * right_impurity) / len(
                    pop)
                if weighted_impurity < current_impurity and weighted_impurity < current_impurity - self.min_impurity_decrease:
                    best_feature = feature
                    best_split_value = value
                    best_left_mask = left_mask
                    best_right_mask = right_mask
                    best_left_impurity = left_impurity
                    best_right_impurity = right_impurity
                    current_impurity = weighted_impurity
        if best_feature is not None:
            return (best_feature, current_impurity, best_split_value, [pop[best_left_mask], pop[best_right_mask]],
                    [best_left_impurity, best_right_impurity])
        else:
            return None

    def fit(self, X, y):
        self.classes_ = list(set(list(y)))
        labels = np.array(y)
        N = len(y)
        self.tree = {}
        population = {0: np.array(range(N))}
        impurity = {0: self.impurity(labels[population[0]]) * N}
        level = 0
        nodes = [0]
        while level < self.max_depth and nodes:
            next_nodes = []
            for node in nodes:
                current_pop = population[node]
                current_impure = impurity[node]
                if len(current_pop) < self.min_samples_split or current_impure == 0 or level+1 == self.max_depth:
                    self.tree[node] = Counter(labels[current_pop])
                else:
                    best_split = self.find_best_split(current_pop, X, labels)
                    if best_split:
                        self.tree[node] = (best_split[0], best_split[2])
                        next_nodes.extend([node * 2 + 1, node * 2 + 2])
                        population[node * 2 + 1] = best_split[3][0]
                        population[node * 2 + 2] = best_split[3][1]
                        impurity[node * 2 + 1] = best_split[4][0]
                        impurity[node * 2 + 2] = best_split[4][1]
                    else:
                        self.tree[node] = Counter(labels[current_pop])
            nodes = next_nodes
            level += 1

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            node = 0
            while True:
                if type(self.tree[node]) == Counter:
                    label = list(self.tree[node].keys())[np.argmax(self.tree[node].values())]
                    predictions.append(label)
                    break
                else:
                    if X[self.tree[node][0]][i] < self.tree[node][1]:
                        node = node * 2 + 1
                    else:
                        node = node * 2 + 2
        return predictions

    def predict_proba(self, X):
        predictions = []
        for i in range(len(X)):
            node = 0
            while True:
                if type(self.tree[node]) == Counter:
                    leaf_counts = self.tree[node]
                    total_count = sum(leaf_counts.values())
                    prob = {label: count / total_count for label, count in leaf_counts.items()}
                    predictions.append(prob)
                    break
                else:
                    if X[self.tree[node][0]][i] < self.tree[node][1]:
                        node = node * 2 + 1
                    else:
                        node = node * 2 + 2
        probs = pd.DataFrame(predictions, columns=self.classes_)
        return probs