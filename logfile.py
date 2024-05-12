import os
import logging
from datetime import datetime

def init_logger(log_file_name ,log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        if not os.path.isdir(log_file_name):
            os.makedirs(log_file_name)
        log_file = os.path.join(log_file_name, log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger

## Change path to your log file here
log_file_name = '/path/to/your_logfile'
LOGGER = init_logger(log_file_name, datetime.now().strftime('%d%b%Y_%H-%M-%S.log'))
