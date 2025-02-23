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
import sys
from src.logger import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))

    return error_message

    

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message