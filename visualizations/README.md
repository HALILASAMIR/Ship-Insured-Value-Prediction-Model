# Ship Value Prediction Project - Visualizations

This directory contains all generated visualizations and plots from the model training and evaluation process.

## Contents

### Model Performance
- `feature_importance_xgb.png` - Feature importance for XGBoost model
- `feature_importance_rf.png` - Feature importance for Random Forest model
- `model_comparison.png` - Side-by-side comparison of XGBoost vs Random Forest

### Data Analysis
- `distribution_by_classification.png` - Distribution of insured values by classification society
- `distribution_by_ship_type.png` - Distribution of insured values by ship type
- `dwt_vs_value.png` - Relationship between DWT and insured value
- `correlation_matrix.png` - Heatmap of feature correlations
- `outliers_identification.png` - Outlier detection visualization

### Model Evaluation
- `residuals_analysis.png` - Residuals plot and analysis
- `prediction_accuracy.png` - Predicted vs actual values
- `cross_validation_scores.png` - Cross-validation results visualization

### Exploratory Data Analysis (EDA)
- `data_overview.png` - Summary statistics visualization
- `missing_values.png` - Missing data analysis
- `ship_age_distribution.png` - Distribution of ship ages
- `tonnage_distribution.png` - Distribution of DWT and GRT

## How to Regenerate Visualizations

To regenerate all visualizations, run:

```bash
python training.py
```

This will create all plots and save them to this directory.

## Viewing the Plots

The visualizations are saved in PNG format at 300 DPI for high quality. You can:
1. View them in any image viewer
2. Insert them into reports or presentations
3. Reference them in documentation

All plots include:
- Clear titles and labels
- Grid lines for readability
- Legend where applicable
- Professional styling and colors
