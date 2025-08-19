"""
Main entry point for the waste management ML pipeline.
This script runs the entire workflow from data ingestion to model training.
"""
import os
import sys
import mlflow
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.exception import CustomException
from src.utils.logger import logging
from src.data.data_ingestion import DataIngestion
from src.data.preprocess import DataTransformation
from src.model_pipeline.train_pipeline import ModelTrainer
from src.model_pipeline.predict_pipeline import ModelPredictor

def run_training_pipeline():
    """Run the complete training pipeline"""
    try:
        logging.info("=== Starting Complete ML Pipeline ===")
        
        # Step 1: Data Ingestion
        logging.info("Step 1: Data Ingestion")
        data_ingestion = DataIngestion()
        raw_data_path, train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        
        # Step 2: Data Preprocessing
        logging.info("Step 2: Data Preprocessing")
        data_transformation = DataTransformation()
        X_train, y_train, X_test, y_test = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        
        # Step 3: Model Training with MLflow tracking
        logging.info("Step 3: Model Training")
        model_trainer = ModelTrainer()
        result = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
        
        # Step 4: Export model to joblib format (easier to load)
        logging.info("Step 4: Exporting model to joblib format")
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "waste_mgmt.joblib")
        joblib.dump(result["model"], model_path)
        logging.info(f"Model exported to {model_path}")
        
        # Print results
        print("\n" + "="*50)
        print(f"TRAINING COMPLETE!")
        print(f"Model training results:")
        print(f"  - Train R² score: {result['train_score']:.4f}")
        print(f"  - Test R² score: {result['test_score']:.4f}")
        print(f"  - Train RMSE: {result['train_rmse']:.4f}")
        print(f"  - Test RMSE: {result['test_rmse']:.4f}")
        print(f"Model saved to {model_path}")
        print("="*50 + "\n")
        
        return result
        
    except Exception as e:
        logging.error(f"Error in training pipeline: {e}")
        raise CustomException(e, sys) from e

def test_prediction():
    """Test the prediction functionality on a sample from test data"""
    try:
        logging.info("Testing prediction functionality")
        
        # Load a sample from test data
        test_data_path = os.path.join("artifacts", "test.csv") 
        test_df = pd.read_csv(test_data_path)
        
        # Get features (all columns except target)
        X = test_df.drop(columns=["recycling_rate_(%)"], axis=1)
        y_true = test_df["recycling_rate_(%)"].values
        
        # Take just one sample
        sample_X = X.iloc[[0]]
        sample_y_true = y_true[0]
        
        # Check if focused model exists
        focused_model_path = os.path.join("models", "waste_mgmt_focused.joblib")
        using_focused_model = os.path.exists(focused_model_path)
        
        # Load the model
        model_path = focused_model_path if using_focused_model else None
        predictor = ModelPredictor(model_path)
        
        if using_focused_model:
            # For focused model, use raw features (model has internal preprocessing)
            logging.info("Using focused model with internal preprocessing")
            
            # Ensure we have all required features
            required_features = ['city/district', 'recycling_infrastructure_rating', 'green_technology_adoption']
            if all(feature in sample_X.columns for feature in required_features):
                # Only keep required features
                sample_X = sample_X[required_features].copy()
                logging.info(f"Using only specific features: {required_features}")
            
            # Make prediction with raw features
            prediction = predictor.predict(sample_X)[0]
        else:
            # For standard model, use preprocessor
            logging.info("Using standard model with external preprocessing")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            preprocessor = joblib.load(preprocessor_path)
            logging.info(f"Loaded preprocessor from {preprocessor_path}")
            
            # Transform features
            sample_X_transformed = preprocessor.transform(sample_X)
            
            # Make prediction
            prediction = predictor.predict(sample_X_transformed)[0]
        
        # Calculate error metrics
        absolute_error = abs(prediction - sample_y_true)
        rmse = np.sqrt(np.mean((prediction - sample_y_true)**2))
        
        print("\n" + "="*50)
        print("PREDICTION TEST")
        print(f"Sample input features:\n{sample_X.head()}")
        print(f"Predicted recycling rate: {prediction:.2f}%")
        print(f"Actual recycling rate: {sample_y_true:.2f}%")
        print(f"RMSE: {rmse:.4f}")
        print("="*50 + "\n")
        
        return {"prediction": prediction, "actual": sample_y_true}
        
    except Exception as e:
        logging.error(f"Error in prediction test: {e}")
        raise CustomException(e, sys) from e

if __name__ == "__main__":
    mlflow.set_tracking_uri("mlruns/")
    
    # Configure MLflow experiment
    experiment_name = "waste_management_model"
    mlflow.set_experiment(experiment_name)
    
    # Run the entire pipeline
    print("\n=== WASTE MANAGEMENT ML PIPELINE ===\n")
    print("1. Running training pipeline...")
    training_results = run_training_pipeline()
    
    print("2. Testing prediction functionality...")
    prediction_results = test_prediction()
    
    print("Pipeline execution complete!")
