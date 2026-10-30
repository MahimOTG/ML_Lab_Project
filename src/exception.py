import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logger import logging  # noqa: F401


def error_message_details(error, error_detail=None):
    if error_detail is None:
        exc_tb = sys.exc_info()[2]
    elif hasattr(error_detail, 'exc_info'):
        exc_tb = error_detail.exc_info()[2]
    else:
        exc_tb = error_detail[2]
    if exc_tb is None:
        return str(error)
    file_name = exc_tb.tb_frame.f_code.co_filename
    return (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] error message [{error}]"
    )


class CustomException(Exception):
    def __init__(self, error_message, error_detail: tuple = None):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
