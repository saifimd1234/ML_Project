"""
logger.py

This module provides a logging utility for the machine learning project. It is used to configure and manage logging 
throughout the project, ensuring that all logs are consistently formatted and directed to appropriate outputs 
(e.g., console, file). The logger helps in tracking the flow of execution, debugging, and monitoring the performance 
and behavior of the application.

Key functionalities:
- Setting up a logger with a specific configuration (e.g., log level, format).
- Writing log messages of various severity levels (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).
- Directing log messages to different handlers (e.g., console, file).
- Ensuring that logs are timestamped and contain relevant context information.

Usage:
- Import the logger module in other parts of the project to log messages.
- Use the logger to record important events, errors, and other information that can help in debugging and monitoring 
    the application.
"""
import logging  # Import the logging module to enable logging functionality
import os  # Import the os module to interact with the operating system
from datetime import datetime  # Import datetime to work with date and time

# Create a log file name based on the current date and time
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

# Define the path where the log file will be stored
logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)

# Create the directory for logs if it doesn't exist
os.makedirs(logs_path, exist_ok=True)

# Define the full path for the log file
LOGS_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging system
logging.basicConfig(
    filename=LOGS_FILE_PATH,  # Set the log file path
    format='[ %(asctime)s ]  %(lineno)d %(name)s - %(levelname)s: %(message)s',  # Define the log message format
    level=logging.INFO,  # Set the logging level to INFO
    filemode='a'  # Append to the log file if it exists
)