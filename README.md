# Bangalore House Price Prediction

## Overview
Predicted house prices in Bangalore using machine learning.
Dataset: Bengaluru House Price Data (Kaggle) — 13,000+ rows, 9 features.

## Results
| Metric | Score |
|--------|-------|
| R² Score | 0.88 |
| RMSE | 28 Lakhs |
| CV R² | 0.83 |

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

## ML Pipeline
1. Data Cleaning — handled sqft ranges, BHK inconsistencies
2. Feature Engineering — price_per_sqft, outlier removal per location
3. Encoding — One-hot encoded 240+ locations
4. Model Comparison — Linear, Ridge, Lasso, Random Forest, XGBoost
5. Hyperparameter Tuning — GridSearchCV (270 fits, 5-fold CV)
6. Best Model — XGBoost (R² = 0.88)

## How to Run
1. Clone this repo
2. Install dependencies: pip install -r requirements.txt
3. Open bangalore_house_price.ipynb in Jupyter
