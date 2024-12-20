import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    """
    Configures a logger with the specified name and log file, along with a console handler.
    
    Parameters:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logs_dir = os.path.dirname(log_file)
    if logs_dir and not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)

    return logger
