"""
Ship Value Prediction Model - Inference Script
================================================

This script handles model loading and inference for ship value predictions.

Usage:
    from inference import ShipValuePredictor
    
    predictor = ShipValuePredictor(model_path='models/xgb_model_v1.pkl')
    prediction = predictor.predict(ship_data)

Author: Samir
Date: 2026-01-17
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from typing import Union, Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShipValuePredictor:
    """Handles model loading and inference for ship value predictions"""
    
    def __init__(self, model_path: str, config_path: str = 'xgb_model_v1.json'):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the trained model file
            config_path: Path to the model configuration file
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.model = None
        self.config = None
        self.feature_names = None
        self.load_model()
        self.load_config()
        
    def load_model(self) -> None:
        """Load the trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_config(self) -> None:
        """Load model configuration"""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            self.feature_names = self.config.get('features', {}).get('feature_names', [])
            logger.info(f"Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def validate_input(self, X: Union[np.ndarray, pd.DataFrame, Dict, List]) -> pd.DataFrame:
        """
        Validate and convert input data
        """
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        # --- AJOUTE CES DEUX LIGNES CI-DESSOUS ---
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        # -----------------------------------------
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X = pd.DataFrame(X, columns=self.feature_names)
        elif not isinstance(X, pd.DataFrame):
            raise TypeError(f"Unsupported input type: {type(X)}")
        
        # Validation des colonnes
        if self.feature_names and set(self.feature_names) != set(X.columns):
            logger.warning("Input columns don't match expected features")
            # Optionnel : réordonner pour être sûr
            X = X[self.feature_names]
        
        return X
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame, Dict], inverse_transform: bool = True) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input data
            inverse_transform: Whether to inverse-transform log predictions to original scale
            
        Returns:
            np.ndarray: Predictions
        """
        X_validated = self.validate_input(X)
        
        # Make predictions in log scale
        y_pred_log = self.model.predict(X_validated)
        
        # Inverse transform if requested
        if inverse_transform:
            y_pred = np.expm1(y_pred_log)
        else:
            y_pred = y_pred_log
        
        return y_pred
    
    def predict_batch(self, X: Union[np.ndarray, pd.DataFrame, List[Dict]]) -> pd.DataFrame:
        """
        Make batch predictions with detailed output
        
        Args:
            X: Input data
            
        Returns:
            pd.DataFrame: Predictions with metadata
        """
        X_validated = self.validate_input(X)
        predictions = self.predict(X_validated, inverse_transform=True)
        
        results = X_validated.copy()
        results['predicted_value_usd'] = predictions
        results['predicted_value_formatted'] = results['predicted_value_usd'].apply(
            lambda x: f"${x:,.2f}"
        )
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model"""
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model doesn't have feature_importances_ attribute")
            return {}
        
        importances = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if not self.config:
            return {}
        
        return {
            'model_name': self.config.get('model_info', {}).get('name'),
            'version': self.config.get('model_info', {}).get('version'),
            'r2_score': self.config.get('performance_metrics', {}).get('r2_score'),
            'mae': self.config.get('performance_metrics', {}).get('mae'),
            'rmse': self.config.get('performance_metrics', {}).get('rmse')
        }


class ShipValueEstimator:
    """Simplified interface for quick estimations"""
    
    def __init__(self, model_path: str):
        """Initialize the estimator"""
        self.predictor = ShipValuePredictor(model_path)
    
    def estimate_from_specs(self, age: int, dwt: float, grt: float, 
                           power: float, ship_type: int = 5, 
                           is_iacs: int = 1, country: int = 1) -> float:
        """
        Estimate value from ship specifications
        
        Args:
            age: Age of ship in years
            dwt: Deadweight tonnage
            grt: Gross register tonnage
            power: Engine power in kW
            ship_type: Encoded ship type (0-n)
            is_iacs: IACS member (0 or 1)
            country: Encoded country code
            
        Returns:
            float: Estimated value in USD
        """
        data = {
            'AGE': age,
            'DWT': dwt,
            'GRT': grt,
            'Puissance_Moteur': power,
            'TYPE_ENCODED': ship_type,
            'is_IACS': is_iacs,
            'PAYS_ENC': country
        }
        
        prediction = self.predictor.predict(data)[0]
        return prediction
    
    def batch_estimate(self, specs_list: List[Dict]) -> pd.DataFrame:
        """
        Estimate values for multiple ships
        
        Args:
            specs_list: List of ship specification dictionaries
            
        Returns:
            pd.DataFrame: Predictions with metadata
        """
        return self.predictor.predict_batch(specs_list)


# Example usage
if __name__ == '__main__':
    # Initialize predictor
    predictor = ShipValuePredictor(
        model_path='models/xgb_model_v1.pkl',
        config_path='xgb_model_v1.json'
    )
    
    # Single prediction
    ship_data = {
        'AGE': 10,
        'DWT': 5000,
        'GRT': 2500,
        'Puissance_Moteur': 1200,
        'TYPE_ENCODED': 5,
        'is_IACS': 1,
        'PAYS_ENC': 1
    }
    
    prediction = predictor.predict(ship_data)
    print(f"Estimated value: ${prediction[0]:,.2f} USD")
    
    # Batch predictions
    ships = [
        {'AGE': 10, 'DWT': 5000, 'GRT': 2500, 'Puissance_Moteur': 1200, 'TYPE_ENCODED': 5, 'is_IACS': 1, 'PAYS_ENC': 1},
        {'AGE': 15, 'DWT': 8000, 'GRT': 4000, 'Puissance_Moteur': 2000, 'TYPE_ENCODED': 3, 'is_IACS': 1, 'PAYS_ENC': 2},
    ]
    
    results = predictor.predict_batch(ships)
    print("\nBatch predictions:")
    print(results)
    
    # Model info
    print("\nModel information:")
    print(predictor.get_model_info())
    
    # Feature importance
    print("\nFeature importance:")
    importance = predictor.get_feature_importance()
    for feature, score in importance.items():
        print(f"  {feature}: {score:.4f}")
