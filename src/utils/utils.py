import os
import sys

import pandas as pd
import numpy as np

# Fix imports to use absolute paths
from src.utils.logger import logging
from src.utils.exception import CustomException

import dill

def save_object(file_path, obj):
    try:
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving object: {e}")
        raise CustomException(e, sys) from e