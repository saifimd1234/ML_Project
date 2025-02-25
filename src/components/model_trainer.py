"""
This module contains the ModelTrainer class, which is responsible for training machine learning models.

Classes:
    ModelTrainer: A class that encapsulates the logic for training machine learning models, including data preprocessing, model selection, training, and evaluation.

Methods:
    __init__(self, model, data): Initializes the ModelTrainer with a specific model and dataset.
    preprocess_data(self): Preprocesses the data before training.
    train_model(self): Trains the machine learning model using the preprocessed data.
    evaluate_model(self): Evaluates the trained model on a validation dataset.
    save_model(self, filepath): Saves the trained model to a specified file path.
"""
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:  # this will give whatever input we require w.r.t model training
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:   # responsible for training the model
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()  # inside self.model_trainer_config we will be getting the above variable path name (trained_model_file_path)


    def initiate_model_trainer(self,train_array,test_array):  # this function will be responsible for training the model. The train and test array comes from data transformation
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
           

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

# save_object is responsible for saving the model.pkl to designated path
            save_object( 
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            

        except Exception as e:
            raise CustomException(e,sys)
