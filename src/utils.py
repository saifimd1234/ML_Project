
"""
utils.py

This module contains utility functions and classes that are used throughout the machine learning project. 
These utilities help with common tasks such as data preprocessing, feature engineering, model evaluation, 
and other repetitive tasks that are needed in multiple parts of the project. By centralizing these 
functions in a single module, we promote code reuse, maintainability, and organization.

Functions and classes in this module may include:
- Data loading and saving functions
- Data transformation and normalization utilities
- Feature extraction and selection methods
- Model evaluation metrics and helper functions
- Logging and debugging tools
- Miscellaneous helper functions

Usage:
    Import the necessary functions or classes from this module into your scripts or notebooks as needed.
    Example:

    Then use the imported functions in your code:
        data = load_data('data.csv')
        normalized_data = normalize_features(data)
"""

import os
import sys
import dill

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves an object to the specified file path using dill.

    Args:
        file_path (str): Path to save the object.
        obj: Object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    """
    Evaluates multiple machine learning models using GridSearchCV.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Testing features.
        y_test: Testing target.
        models: Dictionary of models to evaluate.
        param: Dictionary of hyperparameters for each model.

    Returns:
        dict: A dictionary containing model names as keys and their R2 scores as values.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3) # you can also apply randomSearchCV
            gs.fit(X_train,y_train)  # to select the best parameter

            model.set_params(**gs.best_params_)  # set the best parameter to the model
            model.fit(X_train,y_train)

            # model.fit(X_train,y_train) # train the model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)