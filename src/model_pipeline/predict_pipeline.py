import joblib
import sys
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
            return self.model.predict(features)
        except Exception as e:
            raise CustomException("Prediction failed", sys) from e

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
