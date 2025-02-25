"""
Module: data_ingestion

This module is responsible for ingesting raw data from various sources into the system. 
It includes functions and classes that handle data extraction, loading, and initial validation.

Classes:
    DataIngestor: A class that encapsulates various data ingestion techniques.

Functions:
    ingest_data: Extracts and loads data from the specified sources.

Usage:
    This module is used in the initial stage of the data pipeline to bring raw data into the system for further processing and analysis.
"""

import sys
import os
from src.exception import CustomException  # Ensure this import is correct and the CustomException class exists
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig, DataTransformation

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.
    """
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Corrected attribute name
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() 

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            df = pd.read_csv(os.path.join('notebook', 'data', 'stud.csv'))  # Correct relative path to your CSV file
            logging.info("Dataset loaded as dataframe successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data completed successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion: {}".format(e))
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr)) # this will give the r2_score

# The underscore (_) is often used as a variable name in Python to indicate that the value it holds is temporary or insignificant and will not be used later in the code.
    