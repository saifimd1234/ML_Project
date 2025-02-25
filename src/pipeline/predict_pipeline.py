"""
This module contains the PredictPipeline class, which is responsible for making predictions using a trained machine learning model.

Classes:
    PredictPipeline: A class that encapsulates the logic for loading a trained model, preprocessing input data, making predictions, and postprocessing the results.

Methods:
    __init__(self, model_path): Initializes the PredictPipeline with the path to the trained model.
    load_model(self): Loads the trained model from the specified file path.
    preprocess_input(self, input_data): Preprocesses the input data before making predictions.
    predict(self, input_data): Makes predictions using the preprocessed input data.
    postprocess_output(self, predictions): Postprocesses the predictions before returning them.
"""

import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    """
    A class to handle the prediction pipeline.

    This class loads the pre-trained model and preprocessor, scales the input features,
    and makes predictions using the model.

    Methods:
        predict(features): Predicts the target variable for the given input features.
    """

    def __init__(self):
        pass

    def predict(self, features):
        """
        Predicts the target variable for the given input features.

        Args:
            features (pd.DataFrame): Input features for which predictions are to be made.

        Returns:
            preds (numpy.ndarray): Predicted values for the input features.

        Raises:
            CustomException: If an error occurs during prediction.
        """
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\proprocessor.pkl'
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    A class to handle custom input data.

    This class takes input data, stores it as attributes, and provides a method
    to convert the data into a pandas DataFrame.

    Attributes:
        gender (str): Gender of the student.
        race_ethnicity (str): Race/ethnicity of the student.
        parental_level_of_education (str): Parental level of education.
        lunch (str): Type of lunch the student has.
        test_preparation_course (str): Whether the student took a test preparation course.
        reading_score (int): Reading score of the student.
        writing_score (int): Writing score of the student.

    Methods:
        get_data_as_data_frame(): Converts the input data into a pandas DataFrame.
    """

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        """
        Initializes the CustomData class with input data.

        Args:
            gender (str): Gender of the student.
            race_ethnicity (str): Race/ethnicity of the student.
            parental_level_of_education (str): Parental level of education.
            lunch (str): Type of lunch the student has.
            test_preparation_course (str): Whether the student took a test preparation course.
            reading_score (int): Reading score of the student.
            writing_score (int): Writing score of the student.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Converts the input data into a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the input data.

        Raises:
            CustomException: If an error occurs during DataFrame creation.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)