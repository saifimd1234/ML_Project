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
# any exception that is controlled, sys is used to get the exception, it has the error message
# and the error code
import sys
import logging
import traceback

def error_message_detail(error, error_detail: sys):
    # Extract the traceback object from the error details
    exc_type, exc_obj, exc_tb = sys.exc_info()
    
    # Get the filename where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Get the line number where the error occurred
    line_number = exc_tb.tb_lineno
    
    # Create an error message string with the filename, line number, and error message
    error_message = f"Error occurred in file: {file_name} at line: {line_number} - {str(error)}"
    
    # Print the error message to the console
    print(error_message)
    
    # Return the error message string
    return error_message

class CustomException(Exception):
    """
    Base class for custom exceptions in the machine learning project.
    """
    def __init__(self, error_message, error_detail: sys):
        # Call the base class constructor with the error message
        super().__init__(error_message)
        
        # Set the error message attribute by calling the error_message_detail function
        # This function formats the error message with additional details
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        self.traceback = traceback.format_exc()

    def __str__(self):
        # Return the formatted error message when the exception is converted to a string
        return f"{self.error_message}\nTraceback:\n{self.traceback}"

if __name__ == "__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("Divide by Zero error.")
        raise CustomException(e, sys)