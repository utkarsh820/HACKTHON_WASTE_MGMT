import sys
from dataclasses import dataclass
import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.feature_selection import SelectKBest
from sklearn.metrics import r2_score, mean_squared_error

# Exceptions
from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.utils import save_object

# Pipeline export dill
import dill

# Data Transformation Configuration

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    
class DataTransformation:
    '''
    This function is responsible for data preprocessing.
    
    '''
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            numerical_features = ["green_technology_adoption", "recycling_infrastructure_rating"]
            categorical_features = ["city/district"]
            
            numerical_pipline = Pipeline([
                ("num_scaler",StandardScaler())
            ])
            categorical_pipeline = Pipeline(steps=[
                ("categorical_encoder", OneHotEncoder(handle_unknown="ignore"))
            ])
            
            preprocessor = ColumnTransformer(
                [
                 ("num_pipeline", numerical_pipline, numerical_features),
                 ("cat_pipeline", categorical_pipeline, categorical_features)
                ]
            )
            logging.info("Data transformation object created successfully.")
            return preprocessor
        except Exception as e:
            logging.error(f"Error occurred while getting data transformer object: {e}")
            raise CustomException(e, sys) from e
        
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            # Print column names to debug
            print("Available columns in train_data:", train_data.columns.tolist())

            logging.info("Data loaded successfully.")

            preprocessor = self.get_data_transformer_object()

            X_train = train_data.drop(columns=["recycling_rate_(%)"])
            y_train = train_data["recycling_rate_(%)"]

            X_test = test_data.drop(columns=["recycling_rate_(%)"])
            y_test = test_data["recycling_rate_(%)"]

            logging.info("Data split into features and target.")

            # Fit and transform the training data
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Data transformation completed.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            return X_train_transformed, y_train, X_test_transformed, y_test
        
        except Exception as e:
            logging.error(f"Error occurred during data transformation: {e}")
            raise CustomException(e, sys) from e