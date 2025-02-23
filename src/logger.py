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