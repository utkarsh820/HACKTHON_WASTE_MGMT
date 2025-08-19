import joblib
import sys
import pandas as pd
from src.utils.exception import CustomException
from src.utils.logger import logging


class ModelPredictor:
    """Loads a trained model and makes predictions with simple explanations."""

    def __init__(self, model_path="artifacts/trained_model.pkl"):
        try:
            self.model = joblib.load(model_path)
            self.model_path = model_path
            logging.info(f"Model loaded from {model_path}")
        except Exception as e:
            raise CustomException(f"Failed to load model at {model_path}", sys) from e

    def predict(self, features):
        try:
            # Check if features is already a numpy array (preprocessed data)
            if not isinstance(features, pd.DataFrame):
                return self.model.predict(features)
            
            # Check if we're using the focused model (which has internal preprocessing)
            if "waste_mgmt_focused" in self.model_path:
                # Ensure we have the specific features needed for the focused model
                required_features = ['city/district', 'recycling_infrastructure_rating', 'green_technology_adoption']
                if all(feature in features.columns for feature in required_features):
                    # Only keep the required features for focused model
                    features = features[required_features].copy()
                    logging.info(f"Using focused model with specific features: {required_features}")
                else:
                    missing = [f for f in required_features if f not in features.columns]
                    raise CustomException(f"Missing required features for focused model: {missing}", sys)
            
            return self.model.predict(features)
        except Exception as e:
            raise CustomException(f"Prediction failed: {str(e)}", sys) from e

    def explain_prediction(self, prediction):
        try:
            explanation = {
                "prediction": prediction,
                "model_type": type(self.model).__name__,
            }
            for param in ("alpha", "solver", "fit_intercept"):
                if hasattr(self.model, param):
                    explanation[param] = getattr(self.model, param)
            if hasattr(self.model, "coef_"):
                coeffs = self.model.coef_
                explanation["top_features"] = sorted(
                    enumerate(coeffs), key=lambda x: abs(x[1]), reverse=True
                )[:5]
            return explanation
        except Exception as e:
            raise CustomException("Explanation failed", sys) from e
