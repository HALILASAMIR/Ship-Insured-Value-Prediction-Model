"""
Ship Value Prediction Model - Training Script
=============================================

This script trains the XGBoost and Random Forest models for ship insured value prediction.

Usage:
    python training.py

Author: Samir
Date: 2026-01-17
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'data_file': 'data_navire.csv',
    'separator': ';',
    'test_size': 0.2,
    'random_state': 42,
    'output_dir': 'output',
    'models_dir': 'models',
    'visualizations_dir': 'visualizations'
}

# Create output directories
for directory in [CONFIG['output_dir'], CONFIG['models_dir'], CONFIG['visualizations_dir']]:
    Path(directory).mkdir(exist_ok=True)


class DataPreprocessor:
    """Handles data loading, cleaning, and preprocessing"""
    
    def __init__(self, data_file, separator=';'):
        """Initialize the preprocessor"""
        self.data_file = data_file
        self.separator = separator
        self.df = None
        self.iacs_members = ['ABS', 'BV', 'DNV', 'LR', 'NK', 'RINA', 'KR', 'CCS', 'RS', 'CRS', 'PRS']
        
    def load_data(self):
        """Load the dataset"""
        logger.info(f"Loading data from {self.data_file}")
        self.df = pd.read_csv(self.data_file, sep=self.separator)
        self.df.columns = self.df.columns.str.strip()
        logger.info(f"Data shape: {self.df.shape}")
        return self.df
    
    def clean_target(self):
        """Clean the target variable (INSURED VALUE)"""
        logger.info("Cleaning target variable...")
        self.df['INSURED VALUE'] = self.df['INSURED VALUE'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        self.df['INSURED VALUE'] = pd.to_numeric(self.df['INSURED VALUE'], errors='coerce')
        return self.df
    
    def create_features(self):
        """Create new features"""
        logger.info("Creating new features...")
        # Age of ships
        self.df['AGE'] = 2026 - self.df['BUILT']
        
        # Clean TYPE column
        self.df['TYPE'] = self.df['TYPE'].str.upper().str.strip()
        
        # Mapping similar types
        mappage = {
            'TANKER': 'OIL TANKER',
            'OIL PRODUCTS TANKER': 'OIL TANKER',
            'OIL PRODUCTS TAN': 'OIL TANKER',
            'OFFSHORE SUPPI': 'OFFSHORE SUPPLY',
            'OFFSHORE SUPPL': 'OFFSHORE SUPPLY',
            'MPP BULK': 'BULK CARRIER',
            'BBU': 'BULK CARRIER',
            'GENERAL CARGO SH': 'GENERAL CARGO',
            'GENERAL CARGO VESSEL': 'GENERAL CARGO',
            'CONTAINER SHIP': 'CONTAINER',
            'CONTAINER VESSEL': 'CONTAINER',
            'UCC': 'CONTAINER',
            'dredge': 'DREDGER',
            'lpg tanker': 'LPG TANKER',
            'lpg': 'LPG TANKER',
        }
        self.df['TYPE'] = self.df['TYPE'].replace(mappage)
        
        return self.df
    
    def encode_categorical(self):
        """Encode categorical variables"""
        logger.info("Encoding categorical variables...")
        
        # Encode TYPE
        le_type = LabelEncoder()
        self.df['TYPE_ENCODED'] = le_type.fit_transform(self.df['TYPE'])
        
        # Clean and encode CLASSIFICATION
        self.df['CLASS'] = self.df['CLASS'].astype(str).str.strip().str.upper()
        mappage_class = {
            'NKK': 'NK',
            'CSC': 'CCS',
            'RMRS': 'RS',
            'TURK LLOYD': 'TURKISH LLOYD',
        }
        self.df['CLASS'] = self.df['CLASS'].replace(mappage_class)
        
        # Create IACS binary
        self.df['is_IACS'] = self.df['CLASS'].isin(self.iacs_members).astype(int)
        
        # Clean and encode BUILDER
        self.df['Builder'] = self.df['Builder'].str.strip().str.upper()
        self.df['Builder'] = self.df['Builder'].replace({'ÉTATS-UNIS': 'USA'})
        
        le_country = LabelEncoder()
        self.df['PAYS_ENC'] = le_country.fit_transform(self.df['Builder'])
        
        return self.df
    
    def rename_columns(self):
        """Rename columns to French"""
        logger.info("Renaming columns...")
        traductions = {
            'IMO': 'IMO',
            'TYPE': 'Type_Navire',
            'BUILT': 'Annee_Construction',
            'GRT': 'GRT',
            'DWT': 'DWT',
            'Engine power': 'Puissance_Moteur',
            'Builder': 'Constructeur',
            'CLASS': 'Societe_Classification',
            'FLAG': 'Pavillon',
            'INSURED VALUE': 'Valeur_Assuree'
        }
        self.df = self.df.rename(columns=traductions)
        self.df = self.df.drop(columns=['CAT CLASS'], errors='ignore')
        return self.df
    
    def create_ratio(self):
        """Create value-to-tonnage ratio"""
        logger.info("Creating ratio features...")
        self.df['Ratio_Valeur_DWT'] = self.df['Valeur_Assuree'] / self.df['DWT']
        return self.df
    
    def remove_outliers(self):
        """Remove outliers using IQR method"""
        logger.info("Removing outliers...")
        
        def filter_outliers_by_type(group):
            Q1 = group['Ratio_Valeur_DWT'].quantile(0.25)
            Q3 = group['Ratio_Valeur_DWT'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return group[(group['Ratio_Valeur_DWT'] >= lower_bound) & (group['Ratio_Valeur_DWT'] <= upper_bound)]
        
        df_clean = self.df.groupby('Type_Navire', group_keys=False).apply(filter_outliers_by_type)
        df_clean = df_clean[df_clean['Ratio_Valeur_DWT'] < 10000]
        
        logger.info(f"Rows before cleaning: {len(self.df)}, after: {len(df_clean)}")
        self.df = df_clean
        return self.df
    
    def prepare_data(self):
        """Run all preprocessing steps"""
        self.load_data()
        self.clean_target()
        self.create_features()
        self.encode_categorical()
        self.rename_columns()
        self.create_ratio()
        self.remove_outliers()
        logger.info("Data preprocessing completed!")
        return self.df


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, X, y, test_size=0.2, random_state=42):
        """Initialize the trainer"""
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def split_data(self):
        """Split data into train/test sets"""
        logger.info("Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        logger.info(f"Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_xgboost(self):
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            objective='reg:squarederror',
            random_state=self.random_state,
            verbosity=0
        )
        model.fit(self.X_train, self.y_train)
        self.models['xgboost'] = model
        logger.info("XGBoost training completed!")
        return model
    
    def train_random_forest(self):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = model
        logger.info("Random Forest training completed!")
        return model
    
    def evaluate_model(self, model, name):
        """Evaluate a trained model"""
        logger.info(f"Evaluating {name}...")
        
        y_pred_log = model.predict(self.X_test)
        y_pred_real = np.expm1(y_pred_log)
        y_test_real = np.expm1(self.y_test)
        
        r2 = r2_score(self.y_test, y_pred_log)
        mae = mean_absolute_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        
        results = {
            'model': name,
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'predictions': y_pred_real,
            'actual': y_test_real
        }
        self.results[name] = results
        
        logger.info(f"{name} Results:")
        logger.info(f"  R² Score: {r2:.4f} ({r2*100:.2f}%)")
        logger.info(f"  MAE: ${mae:,.2f}")
        logger.info(f"  RMSE: ${rmse:,.2f}")
        
        return results
    
    def cross_validate(self, model, name, cv=5):
        """Perform cross-validation"""
        logger.info(f"Cross-validating {name}...")
        cv_scores = cross_val_score(model, self.X, self.y, cv=cv, scoring='r2')
        logger.info(f"{name} CV Scores: {cv_scores}")
        logger.info(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        return cv_scores


class ModelVisualizer:
    """Handles visualization of results"""
    
    @staticmethod
    def plot_feature_importance(model, feature_names, output_path):
        """Plot feature importance"""
        logger.info("Creating feature importance plot...")
        plt.figure(figsize=(10, 6))
        importances = pd.Series(model.feature_importances_, index=feature_names)
        importances.sort_values(ascending=True).plot(kind='barh', color='skyblue', edgecolor='navy')
        plt.title('Feature Importance for Ship Value Prediction', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved to {output_path}")
    
    @staticmethod
    def plot_residuals(y_actual, y_pred, output_path):
        """Plot residuals"""
        logger.info("Creating residuals plot...")
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        residuals = y_actual - np.log1p(y_pred)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--', linewidth=2)
        plt.title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Values (Log)')
        plt.ylabel('Residuals')
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        plt.title('Distribution of Residuals', fontsize=12, fontweight='bold')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved to {output_path}")
    
    @staticmethod
    def plot_comparison(results_dict, output_path):
        """Plot model comparison"""
        logger.info("Creating comparison plot...")
        
        models = list(results_dict.keys())
        r2_scores = [results_dict[m]['r2_score'] * 100 for m in models]
        mae_values = [results_dict[m]['mae'] for m in models]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_ylabel('R² Score (%)', color=color, fontsize=12)
        bars = ax1.bar(models, r2_scores, color=color, alpha=0.6, width=0.4)
        ax1.tick_params(axis='y', labelcolor=color)
        
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('MAE (USD)', color=color, fontsize=12)
        line = ax2.plot(models, mae_values, color=color, marker='o', linewidth=3, markersize=10)
        ax2.tick_params(axis='y', labelcolor=color)
        
        for i, (model, mae) in enumerate(zip(models, mae_values)):
            ax2.text(i, mae, f'${mae:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.title('Model Comparison: XGBoost vs Random Forest', fontsize=14, fontweight='bold')
        fig.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved to {output_path}")


def save_model(model, model_name, output_dir):
    """Save trained model"""
    path = Path(output_dir) / f"{model_name}.pkl"
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
    return path


def save_results(results, output_file):
    """Save results to JSON"""
    results_to_save = {}
    for model_name, result in results.items():
        results_to_save[model_name] = {
            'r2_score': float(result['r2_score']),
            'mae': float(result['mae']),
            'rmse': float(result['rmse'])
        }
    
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=4)
    logger.info(f"Results saved to {output_file}")


def main():
    """Main training pipeline"""
    logger.info("Starting training pipeline...")
    
    # Data preprocessing
    preprocessor = DataPreprocessor(CONFIG['data_file'], separator=CONFIG['separator'])
    df = preprocessor.prepare_data()
    
    # Prepare features and target
    features = ['AGE', 'DWT', 'GRT', 'Puissance_Moteur', 'TYPE_ENCODED', 'is_IACS', 'PAYS_ENC']
    X = df[features]
    y = np.log1p(df['Valeur_Assuree'])
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    # Model training
    trainer = ModelTrainer(X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'])
    trainer.split_data()
    
    # Train models
    xgb_model = trainer.train_xgboost()
    rf_model = trainer.train_random_forest()
    
    # Evaluate models
    trainer.evaluate_model(xgb_model, 'XGBoost')
    trainer.evaluate_model(rf_model, 'Random Forest')
    
    # Cross-validation
    trainer.cross_validate(xgb_model, 'XGBoost')
    trainer.cross_validate(rf_model, 'Random Forest')
    
    # Save models
    save_model(xgb_model, 'xgb_model_v1', CONFIG['models_dir'])
    save_model(rf_model, 'rf_model_v1', CONFIG['models_dir'])
    
    # Save results
    save_results(trainer.results, Path(CONFIG['output_dir']) / 'xgb_model_v1.json')
    
    # Visualizations
    viz = ModelVisualizer()
    viz.plot_feature_importance(xgb_model, features, Path(CONFIG['visualizations_dir']) / 'feature_importance_xgb.png')
    viz.plot_feature_importance(rf_model, features, Path(CONFIG['visualizations_dir']) / 'feature_importance_rf.png')
    viz.plot_comparison(trainer.results, Path(CONFIG['visualizations_dir']) / 'model_comparison.png')
    
    logger.info("Training pipeline completed!")


if __name__ == '__main__':
    main()
