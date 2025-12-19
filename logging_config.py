import logging
import sys

def setup_logger():
    """
    Configures and returns a logger instance.
    - File handler: DEBUG level, detailed format (time, file, line).
    - Console handler: INFO level, concise format (message only).
    """
    logger = logging.getLogger("cat_app")
    logger.setLevel(logging.DEBUG)
    
    # Avoid adding handlers multiple times if setup_logger is called repeatedly
    if logger.hasHandlers():
        return logger

    # File Handler
    # Use mode='a' to avoid clearing the log file when worker processes import this module
    file_handler = logging.FileHandler("app.log", mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

# Create the logger instance to be imported by other modules
logger = setup_logger()
