import os
import sys
import joblib
import mlflow
import numpy as np
import mlflow.sklearn
from dataclasses import dataclass
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from src.utils.logger import logging
from src.utils.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "trained_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Starting model training with Ridge Regression")

            # Start MLflow run
            with mlflow.start_run(run_name="final_ridge_prod"):
                
                ridge_model = Ridge(
                    alpha=1.623776739188721,
                    solver='lsqr',
                    fit_intercept=True,
                    random_state=42
                )

                ridge_model.fit(X_train, y_train)
                logging.info("Model training completed")

                # Calculate R² scores
                train_score = ridge_model.score(X_train, y_train)
                test_score = ridge_model.score(X_test, y_test)

                # Calculate RMSE
                y_train_pred = ridge_model.predict(X_train)
                y_test_pred = ridge_model.predict(X_test)
                
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

                logging.info(f"Train R²: {train_score:.4f}, Train RMSE: {train_rmse:.4f}")
                logging.info(f"Test R²: {test_score:.4f}, Test RMSE: {test_rmse:.4f}")

                # Log parameters to MLflow
                mlflow.log_param("alpha", 1.623776739188721)
                mlflow.log_param("solver", "lsqr")
                mlflow.log_param("fit_intercept", True)
                mlflow.log_param("random_state", 7)

                # Log metrics to MLflow
                mlflow.log_metric("train_r2", train_score)
                mlflow.log_metric("test_r2", test_score)
                mlflow.log_metric("train_rmse", train_rmse)
                mlflow.log_metric("test_rmse", test_rmse)

                # Log the model to MLflow
                mlflow.sklearn.log_model(ridge_model, artifact_path="model")

                # Save the model locally as well
                joblib.dump(ridge_model, self.model_trainer_config.trained_model_file_path)
                logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")

            return {
                "train_score": train_score,
                "test_score": test_score,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "model": ridge_model
            }

        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise CustomException(e, sys) from e
