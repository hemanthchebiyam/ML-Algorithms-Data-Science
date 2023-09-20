# This code is written by Hemanth Chebiyam.
# Email: hc3746@rit.edu

import numpy as np
import pandas as pd
from collections import Counter

# I have not used the hint file for this assignment.
class my_evaluation:
    # Binary class or multi-class classification evaluation
    # Each data point can only belong to one class

    def __init__(self, predictions, actuals, pred_proba=None):
        # inputs:
        # predictions: list of predicted classes
        # actuals: list of ground truth
        # pred_proba: pd.DataFrame of prediction probability of belonging to each class
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba) == pd.DataFrame:
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None

    def confusion(self):
        # compute confusion matrix for each class in self.classes_
        # self.confusion = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables
        # write your own code below

        # Initialize an empty dictionary to store confusion matrices
        self.confusion_matrix = {}

        # Iterate through each class in the list of classes
        for target in self.classes_:
            tp = np.sum((self.actuals == target) & (self.predictions == target))  # Calculate True Positives (TP)
            tn = np.sum((self.actuals != target) & (self.predictions != target))  # Calculate True Negatives (TN)
            fp = np.sum((self.actuals != target) & (self.predictions == target))  # Calculate False Positives (FP)
            fn = np.sum((self.actuals == target) & (self.predictions != target))  # Calculate False Negatives (FN)

            # # Store the TP, TN, FP, and FN values in the confusion matrix dictionary
            self.confusion_matrix[target] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

    def precision(self, target=None, average="macro"):
        # compute precision
        # target: target class (str). If not None, then return precision of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average precision
        # output: prec = float
        # note: be careful for divided by 0

        # Check if the confusion matrix has been computed; compute it if not
        if self.confusion_matrix is None:
            self.confusion()

        if target is not None:
            # Calculate precision for a specific target class
            if target not in self.classes_:
                raise ValueError(f"Target class '{target}' not found in classes.")

            tp = self.confusion_matrix[target]["TP"]
            fp = self.confusion_matrix[target]["FP"]

            # Handle division by zero
            if tp + fp == 0:
                return 0.0
            else:
                return tp / (tp + fp)

        else:
            # Calculate average precision
            precisions = []

            for class_label in self.classes_:
                tp = self.confusion_matrix[class_label]["TP"]
                fp = self.confusion_matrix[class_label]["FP"]

                # Handle division by zero
                if tp + fp == 0:
                    precisions.append(0.0)
                else:
                    precisions.append(tp / (tp + fp))

            if average == "macro":
                # Compute the macro-average precision
                return sum(precisions) / len(self.classes_)
            elif average == "micro":
                # Compute the micro-average precision
                total_tp = sum(self.confusion_matrix[class_label]["TP"] for class_label in self.classes_)
                total_fp = sum(self.confusion_matrix[class_label]["FP"] for class_label in self.classes_)

                # Handle division by zero
                if total_tp + total_fp == 0:
                    return 0.0
                else:
                    return total_tp / (total_tp + total_fp)
            elif average == "weighted":
                # Compute the weighted average precision
                class_counts = [len(self.actuals[self.actuals == class_label]) for class_label in self.classes_]
                weighted_precisions = [precisions[i] * class_counts[i] for i in range(len(self.classes_))]
                total_count = sum(class_counts)

                # Handle division by zero
                if total_count == 0:
                    return 0.0
                else:
                    return sum(weighted_precisions) / total_count

    def recall(self, target=None, average="macro"):
        # compute recall
        # target: target class (str). If not None, then return recall of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average recall
        # output: recall = float
        # note: be careful for divided by 0

        # Check if the confusion matrix has been computed; compute it if not
        if self.confusion_matrix is None:
            self.confusion()

        if target is not None:
            # Calculate recall for a specific target class
            if target not in self.classes_:
                raise ValueError(f"Target class '{target}' not found in classes.")

            tp = self.confusion_matrix[target]["TP"]
            fn = self.confusion_matrix[target]["FN"]

            # Handle division by zero
            if tp + fn == 0:
                return 0.0
            else:
                return tp / (tp + fn)

        else:
            # Calculate average recall
            recalls = []

            for class_label in self.classes_:
                tp = self.confusion_matrix[class_label]["TP"]
                fn = self.confusion_matrix[class_label]["FN"]

                # Handle division by zero
                if tp + fn == 0:
                    recalls.append(0.0)
                else:
                    recalls.append(tp / (tp + fn))

            if average == "macro":
                # Compute the macro-average recall
                return sum(recalls) / len(self.classes_)
            elif average == "micro":
                # Compute the micro-average recall
                total_tp = sum(self.confusion_matrix[class_label]["TP"] for class_label in self.classes_)
                total_fn = sum(self.confusion_matrix[class_label]["FN"] for class_label in self.classes_)

                # Handle division by zero
                if total_tp + total_fn == 0:
                    return 0.0
                else:
                    return total_tp / (total_tp + total_fn)
            elif average == "weighted":
                # Compute the weighted average recall
                class_counts = [len(self.actuals[self.actuals == class_label]) for class_label in self.classes_]
                weighted_recalls = [recalls[i] * class_counts[i] for i in range(len(self.classes_))]
                total_count = sum(class_counts)

                # Handle division by zero
                if total_count == 0:
                    return 0.0
                else:
                    return sum(weighted_recalls) / total_count

    def f1(self, target=None, average="macro"):
        # compute f1
        # target: target class (str). If not None, then return f1 of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average f1
        # output: f1 = float
        # note: be careful for divided by 0

        # Check if the confusion matrix has been computed; compute it if not
        if self.confusion_matrix is None:
            self.confusion()

        if target is not None:
            # Calculate F1-score for a specific target class
            if target not in self.classes_:
                raise ValueError(f"Target class '{target}' not found in classes.")

            tp = self.confusion_matrix[target]["TP"]
            fp = self.confusion_matrix[target]["FP"]
            fn = self.confusion_matrix[target]["FN"]

            # Handle division by zero
            if tp + fp == 0 or tp + fn == 0:
                return 0.0
            else:
                # precision = tp / (tp + fp)
                # recall = tp / (tp + fn)
                # f1_score_target = 2 * (precision * recall) / (precision + recall)
                # return f1_score_target
                precision = self.precision(target = target, average=average)
                recall = self.recall(target = target, average=average)
                f1_score_target = 2 * (precision * recall) / (precision + recall)
                return f1_score_target

        else:
            # Calculate average F1-score
            f1_scores = []

            for class_label in self.classes_:
                tp = self.confusion_matrix[class_label]["TP"]
                fp = self.confusion_matrix[class_label]["FP"]
                fn = self.confusion_matrix[class_label]["FN"]

                # Handle division by zero
                if tp + fp == 0 or tp + fn == 0:
                    f1_scores.append(0.0)
                else:
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    f1_scores.append(2 * (precision * recall) / (precision + recall))

            if average == "macro":
                # Compute the macro-average F1-score
                return sum(f1_scores) / len(self.classes_)
            elif average == "micro":
                # Compute the micro-average F1-score
                total_tp = sum(self.confusion_matrix[class_label]["TP"] for class_label in self.classes_)
                total_fp = sum(self.confusion_matrix[class_label]["FP"] for class_label in self.classes_)
                total_fn = sum(self.confusion_matrix[class_label]["FN"] for class_label in self.classes_)

                # Handle division by zero
                if total_tp + total_fp == 0 or total_tp + total_fn == 0:
                    return 0.0
                else:
                    total_precision = total_tp / (total_tp + total_fp)
                    total_recall = total_tp / (total_tp + total_fn)
                    return 2 * (total_precision * total_recall) / (total_precision + total_recall)
            elif average == "weighted":
                # Compute the weighted average F1-score
                class_counts = [len(self.actuals[self.actuals == class_label]) for class_label in self.classes_]
                weighted_f1_scores = [f1_scores[i] * class_counts[i] for i in range(len(self.classes_))]
                total_count = sum(class_counts)

                # Handle division by zero
                if total_count == 0:
                    return 0.0
                else:
                    return sum(weighted_f1_scores) / total_count

    def auc(self, target):
        # compute AUC of ROC curve for the target class
        # return auc = float

        # Check if prediction probabilities are available; return None if not
        if self.pred_proba is None:
            return None

        if target in self.classes_:
            # Extract prediction probabilities for the target class
            target_pred_proba = self.pred_proba[target]

            # Sort the prediction probabilities in descending order and get the index order
            order = np.argsort(target_pred_proba)[::-1]

            # Initialize true positive rate (tpr) and false positive rate (fpr) with initial values
            tpr = [0]
            fpr = [0]

            # Initialize the AUC for the target class
            auc_target = 0

            # Initialize counts for true positives (tp), false positives (fp), false negatives (fn), and true negatives (tn)
            tp = 0
            fp = 0
            fn = Counter(self.actuals)[target]  # Count occurrences of the target class in actuals
            tn = len(self.actuals) - fn  # Calculate the count of non-target class

            # Iterate through the prediction probabilities in the sorted order
            for i in order:
                if self.actuals[i] == target:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1
                    tn -= 1

                # Calculate and append the true positive rate (tpr) and false positive rate (fpr)
                tpr.append(tp / (tp + fn))
                fpr.append(fp / (fp + tn))

                # Update the AUC for the target class using the trapezoidal rule
                auc_target += (fpr[-1] - fpr[-2]) * (tpr[-1] + tpr[-2]) / 2
                # (fpr[-1] - fpr[-2]) calculates the horizontal width of the trapezoid,
                # which corresponds to the change in the false positive rate between the current and previous points on the ROC curve.

                # (tpr[-1] + tpr[-2]) calculates the average vertical height of the trapezoid,
                # which corresponds to the sum of the true positive rates at the current and previous points.

            return auc_target

        else:
            raise Exception("Unknown target class.")


