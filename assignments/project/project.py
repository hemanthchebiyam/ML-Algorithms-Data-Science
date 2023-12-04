# This code is written by Hemanth Chebiyam.
# Email: hc3746@rit.edu

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class my_model():

    def __init__(self) -> None:
        ohe = OneHotEncoder()
        vect1 = TfidfVectorizer(stop_words="english")

        # Create the column transformer
        self.ct = make_column_transformer(
            (ohe, ['telecommuting', 'has_company_logo', 'has_questions']),
            (vect1, 'text'),
            (StandardScaler(), ['character_count'])
        )
        self.clf = SGDClassifier(loss="log_loss", class_weight="balanced",
                          random_state=0, alpha=0.01, penalty='l1')
        # Set up the hyperparameter grid for SGDClassifier
        param_grid = {
            'sgdclassifier__loss': ['hinge', 'log_loss', 'perceptron'],
            'sgdclassifier__class_weight': [None, 'balanced'],
            'sgdclassifier__random_state': [0, 42, 100],
            'sgdclassifier__alpha': [0.0001, 0.01],
            'sgdclassifier__penalty': ['l1', 'l2']
        }

        # Create the pipeline with GridSearchCV
        self.pipe = make_pipeline(self.ct, self.clf)

        # Use F1 score as the scoring metric for GridSearchCV
        self.grid_search = GridSearchCV(self.pipe, param_grid, cv=5, scoring=make_scorer(f1_score))



    def fit(self, X, y):
        # Preprocess the data
        X_p = preprocessing(X)
        # Perform hyperparameter tuning
        self.grid_search.fit(X_p, y)

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
    # X_copy.drop(['telecommuting'], inplace=True, axis='columns')
    # X_copy.drop(['has_company_logo'], inplace=True, axis='columns')
    # X_copy.drop(['has_questions'], inplace=True, axis='columns')
    X_copy['character_count'] = X_copy['text'].apply(len)

    return X_copy