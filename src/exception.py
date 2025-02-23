"""
exception.py

This module defines custom exception classes for the machine learning project.
These exceptions are used to handle specific error cases and provide more
informative error messages throughout the project.

Classes:
    CustomException: A base class for other custom exceptions.
    DataValidationError: Raised when data validation fails.
    ModelTrainingError: Raised when an error occurs during model training.
    PredictionError: Raised when an error occurs during prediction.

Usage:
    Use these custom exceptions to handle and raise specific errors in your
    machine learning pipeline, ensuring that error handling is consistent and
    informative.
"""
import os
from datetime import datetime
import sys
import logging
import traceback

# Create a log file name based on the current date and time
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

# Define the path where the log file will be stored
logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)

# Create the directory for logs if it doesn't exist
os.makedirs(logs_path, exist_ok=True)

# Configure logging
LOGS_FILE_PATH = os.path.join(logs_path, LOG_FILE)
logging.basicConfig(
    filename=LOGS_FILE_PATH,  # Set the log file path
    format='[ %(asctime)s ]  %(lineno)d %(name)s - %(levelname)s: %(message)s',  # Define the log message format
    level=logging.INFO,  # Set the logging level to INFO
    filemode='a'  # Append to the log file if it exists
)

def error_message_detail(error, error_detail: sys):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in file: {file_name} at line: {line_number} - {str(error)}"
    return error_message

class CustomException(Exception):
    """
    Base class for custom exceptions in the machine learning project.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        self.traceback = traceback.format_exc()
        logging.error(self.__str__())

    def __str__(self):
        return f"{self.error_message}\nTraceback:\n{self.traceback}"
