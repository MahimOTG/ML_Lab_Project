
"""Reusable helpers for saving and loading C-MAPSS models."""

import os
import pickle
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.exception import CustomException  # noqa: E402
from src.logger import logging  # noqa: E402


def save_object(file_path, obj):
    try:
        folder = os.path.dirname(file_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

        logging.info('Saved object to %s', file_path)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)

        logging.info('Loaded object from %s', file_path)
        return obj
    except Exception as e:
        raise CustomException(e, sys)
