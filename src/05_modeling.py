# ----------------------------------
# Step 7: Train/ Test Split
# ----------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data_processed/feature_engineered_real_estate.csv")

# Define X and y
X = df.drop(columns=["PriceRatio"])
y = df["PriceRatio"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check shapes
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nTarget variable (train):")
print(y_train.describe())


# ---------------------------------------------------
# Step 8: Baseline regression models( Ridge & Lasso)
# ---------------------------------------------------

from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Initialize models
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.01)

# Fit models
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# Predictions
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

# Metrics
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

print("Ridge RMSE:", rmse_ridge)
print("Ridge MAE:", mae_ridge)

print("Lasso RMSE:", rmse_lasso)
print("Lasso MAE:", mae_lasso)