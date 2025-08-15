import os
import sys

from src.utils.exception import CustomException
from src.utils.logger import logging

from src.data.preprocess import DataTransformation, DataTransformationConfig
from src.model_pipeline.train_pipeline import ModelTrainer, ModelTrainerConfig
from src.utils.utils import save_object

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

# --------------Ingestions--------------

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'train.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")
        
        try:
            df=pd.read_csv('data\processed\Waste_Management_with_Extra_Features.csv')
            logging.info("Data loaded successfully")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # saved to csv
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # train_test_split
            logging.info("Initiated Splitting data into train and test sets")
            
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=7)
            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data splitting completed successfully")
            
            return (
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {e}")
            raise CustomException(e, sys)

if __name__=="__main__":
    data_ingestion = DataIngestion()
    raw_data, train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train, y_train, X_test, y_test = data_transformation.initiate_data_transformation(train_data, test_data)
    
    model_trainer = ModelTrainer()
    train_score, test_score, model = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)