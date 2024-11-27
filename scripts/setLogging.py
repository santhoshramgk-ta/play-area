import logging
import os


def setup_logging(log_level=logging.INFO, log_to_console=True, log_path=None):
    handlers = []

    # Ensure log directory exists if a valid log file path is specified
    if log_path and isinstance(log_path, str):  # Ensure log_path is a string
        log_dir = os.path.dirname(log_path)
        if log_dir:  # Create directory only if the path includes a directory
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    if log_to_console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=log_level,
        format=(
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(funcName)s - %(message)s"
        ),
        handlers=handlers,
    )
