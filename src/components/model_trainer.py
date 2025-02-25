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
    """Configuration class for model training."""
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:   # responsible for training the model
    """Class responsible for training machine learning models."""
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()  # inside self.model_trainer_config we will be getting the above variable path name (trained_model_file_path)


    def initiate_model_trainer(self,train_array,test_array):  # this function will be responsible for training the model. The train and test array comes from data transformation
        """
        Trains multiple machine learning models and selects the best one based on R2 score.

        Args:
            train_array (np.ndarray): Training data array.
            test_array (np.ndarray): Testing data array.

        Returns:
            float: R2 score of the best model.
        """
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
           
           # params can be created over here or it can be in some other config file and we can import it here
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)
            
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            # Alternative:
            # best_model_name = next(key for key, value in model_report.items() if value == best_model_score)
            # Used a generator expression (next) to find the best model name more efficiently.

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient R2 score.")
            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

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
