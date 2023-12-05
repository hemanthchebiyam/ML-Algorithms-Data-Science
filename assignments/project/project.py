# This code is written by Hemanth Chebiyam.
# Email: hc3746@rit.edu

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

class my_model():

    def __init__(self) -> None:
        ohe = OneHotEncoder(drop='first', handle_unknown='ignore')
        vect1 = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False, max_features=5000)

        # Create the column transformer
        self.ct = make_column_transformer(
            (ohe, ['telecommuting', 'has_company_logo', 'has_questions']),
            (vect1, 'text'),
            (StandardScaler(copy=False), ['character_count'])
        )

        self.clf = SVC()

        # Set up the hyperparameter grid for SVC
        param_grid = {
            'svc__C': [0.1, 1, 10],
            'svc__kernel': ['linear', 'rbf'],
            'svc__class_weight': ['balanced'],
            'svc__gamma': ['scale', 'auto', 0.1, 0.01],
            'svc__shrinking': [True, False],
            'svc__decision_function_shape': ['ovr', 'ovo']
        }

        # Create the pipeline with GridSearchCV
        self.pipe = make_pipeline(self.ct, self.clf)

        # Use F1 score as the scoring metric for GridSearchCV
        self.grid_search = GridSearchCV(self.pipe, param_grid, cv=5, scoring=make_scorer(f1_score))



    def fit(self, X, y):
        # Preprocess the data
        X_p = preprocessing(X)

        # Separate the majority and minority classes
        majority_class = X_p[y == 0]
        minority_class = X_p[y == 1]

        # Determine the number of samples to randomly select (e.g., same as the minority class size)
        desired_sample_size = len(minority_class)

        # Randomly sample the majority class to match the desired size
        majority_sampled = resample(majority_class, replace=False, n_samples=desired_sample_size, random_state=38)

        # Combine the minority class and the randomly sampled majority class
        balanced_X = pd.concat([minority_class, majority_sampled])
        balanced_y = np.concatenate([np.ones(len(minority_class)), np.zeros(len(majority_sampled))])
        # Perform hyperparameter tuning
        self.grid_search.fit(balanced_X, balanced_y)

        # Set the best hyperparameters in the pipeline
        best_params = self.grid_search.best_params_
        self.pipe.set_params(**best_params)

        # Fit the model with the best hyperparameters
        self.pipe.fit(X_p, y)

        return

    def predict(self, X):
        X_p = preprocessing(X)

        predictions = self.pipe.predict(X_p)
        return predictions

def preprocessing(X):
    X_copy = X.copy(deep=True)
    X_copy.fillna(' ', inplace=True)
    X_copy['text'] = (X_copy['title'] + ' ' + X_copy['location'] + ' '
                      + X_copy['description'] + ' ' + X_copy['requirements'])

    X_copy.drop(['location'], inplace=True, axis='columns')
    X_copy.drop(['requirements'], inplace=True, axis='columns')
    X_copy.drop(['title'], inplace=True, axis='columns')
    X_copy.drop(['description'], inplace=True, axis='columns')
    X_copy['character_count'] = X_copy['text'].apply(len)

    return X_copy