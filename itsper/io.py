import logging


def get_logger(name: str)-> logging.Logger:
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set logger to capture info level messages

    # Create a console handler and set level to INFO
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)
    return logger
