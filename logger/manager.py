import os
import datetime
import logging
from logging.handlers import TimedRotatingFileHandler

class LoggerManager:
    """
    Class to manage and initialize a logger that writes logs to a file with rotation.
    """
    def __init__(self, log_directory='./logs', log_file_name='segmentation-training.log', logger_name='main', log_level=logging.DEBUG):
        """
        Initializes a new instance of LoggerManager.
        
        Args:
        - log_directory (str): The directory where the log file will be stored.
        - log_file_name (str): The name of the log file.
        - logger_name (str): The name for the logger.
        - log_level (logging.LEVEL): The logging level.
        """
        self.log_directory = log_directory
        self.log_file_name = log_file_name
        self.logger_name = logger_name
        self.log_level = log_level
        self.logger = None
        self.setup_logger()

    def setup_logger(self):
        """
        Sets up and configures the logger.
        """
        # Ensure the log directory exists
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)

        # Get the current date and time
        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Append the current date and time to the log_file_name before the .log extension
        log_file_name_with_datetime = self.log_file_name.replace(".log", f"_{current_datetime}.log")

        log_path = os.path.join(self.log_directory, log_file_name_with_datetime)

        # Create logger and set its level to DEBUG
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(self.log_level)

        # Create a handler that writes log messages to a file, with rotation happening at midnight
        handler = TimedRotatingFileHandler(log_path, when='midnight', backupCount=30)

        # Define the format for the log messages
        log_format = '%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s'
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(handler)

    def get_logger(self):
        """
        Returns the configured logger instance.
        """
        return self.logger