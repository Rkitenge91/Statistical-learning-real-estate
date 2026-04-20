# ----------------------------------
# Step 7: Train/ Test Split
# ----------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data_processed/feature_engineered_real_estate.csv")

# Define X and y
X = df.drop(columns=["PriceRatio", "TotalAppraisedValue"])
y = df["PriceRatio"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6243)

# Check shapes
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nTarget variable (train):")
print(y_train.describe())


# ---------------------------------------------------
# Step 8: Baseline regression models( Ridge & Lasso)
# ---------------------------------------------------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Initialize models
ridge = Pipeline([("scale", StandardScaler()),("model", Ridge(alpha=1.0))])
lasso = Pipeline([("scale", StandardScaler()),("model", Lasso(alpha=0.01, max_iter=20000))])
# Fit models
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# Predictions
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

pred_df = pd.DataFrame({
    "Actual": y_test,
    "RidgeClosedForm": y_pred_ridge,
    "Lasso": y_pred_lasso,
})

pred_df.to_csv("data_processed/model_predictions.csv", index=False)

# Metrics
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

print("Ridge RMSE:", rmse_ridge)
print("Ridge MAE:", mae_ridge)

print("Lasso RMSE:", rmse_lasso)
print("Lasso MAE:", mae_lasso)



# ---------------------------------------------
# Step 9: Cross-Validation for Baseline Models
# ---------------------------------------------

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
import numpy as np

# 5-fold cross-validation
kf5 = KFold(n_splits=5, shuffle=True, random_state=6243)

# Ridge and Lasso pipelines
ridge_cv_model = Pipeline([("scale", StandardScaler()),("model", Ridge(alpha=1.0))])

lasso_cv_model = Pipeline([("scale", StandardScaler()),("model", Lasso(alpha=0.01, max_iter=20000))])

# Cross-validated MSE
ridge_cv_mse = -cross_val_score(ridge_cv_model, X_train, y_train,cv=kf5,scoring="neg_mean_squared_error").mean()

lasso_cv_mse = -cross_val_score(lasso_cv_model, X_train, y_train,cv=kf5,scoring="neg_mean_squared_error").mean()

# Convert to RMSE
ridge_cv_rmse = np.sqrt(ridge_cv_mse)
lasso_cv_rmse = np.sqrt(lasso_cv_mse)

print("Ridge 5-fold CV MSE:", ridge_cv_mse)
print("Ridge 5-fold CV RMSE:", ridge_cv_rmse)

print("Lasso 5-fold CV MSE:", lasso_cv_mse)
print("Lasso 5-fold CV RMSE:", lasso_cv_rmse)
