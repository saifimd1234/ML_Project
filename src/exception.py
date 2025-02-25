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
import traceback
from src.logger import logging
import logging



def error_message_detail(error, error_detail:sys):
    """
    Extract detailed error information from the traceback.

    Args:
        error: The exception object.
        error_detail (tuple): The traceback information from sys.exc_info().

    Returns:
        str: A formatted error message with file name, line number, and error details.
    """
    exc_type, exc_obj, exc_tb = error_detail  # Correctly unpack the error detail
    
    # Get the filename where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Get the line number where the error occurred
    line_number = exc_tb.tb_lineno
    
    # Create an error message string with the filename, line number, and error message
    error_message = f"Error occurred in file: {file_name} at line: {line_number} - {str(error)}"
    
    # Print the error message to the console
    # print(error_message)
    
    # Return the error message string
    return error_message

class CustomException(Exception):
    """
    Base class for custom exceptions in the machine learning project.
    """
    def __init__(self, error_message, error_detail:sys):
        """
        Initialize the custom exception.

        Args:
            error_message: The exception message.
            error_detail (tuple): The traceback information from sys.exc_info().
        """
        super().__init__(error_message)
        
        # Set the error message attribute by calling the error_message_detail function
        # This function formats the error message with additional details

        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        self.traceback = traceback.format_exc()

    def __str__(self):
        # Return the formatted error message when the exception is converted to a string
        return f"{self.error_message}\nTraceback:\n{self.traceback}"