
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

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)