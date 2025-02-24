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

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.
    """
    train_data_path: str = os.path.join('artifact', "train.csv")  # Corrected attribute name
    test_data_path: str = os.path.join('artifact', "test.csv")
    raw_data_path: str = os.path.join('artifact', "raw_data.csv")

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.ingestion_config = config  # Use the provided config instead of creating a new one

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
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path,
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion: {}".format(e))
            raise CustomException(e, sys)

if __name__ == "__main__":
    config = DataIngestionConfig()
    data_ingestion = DataIngestion(config)
    data_ingestion.initiate_data_ingestion()