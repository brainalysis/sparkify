import logging

logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
# info level logger


def info_log(message: str):
    return logging.info(message)

