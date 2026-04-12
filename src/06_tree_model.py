import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

# read in
df = pd.read_csv("data_processed/feature_engineered_real_estate.csv")

# data formatting
X = df.drop(columns=["PriceRatio", "TotalAppraisedValue"]).copy()
cols = X.columns
X = np.array(X)
y = df["PriceRatio"].copy()
y = np.array(y)
# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6243)

# fitting the forest and predicting values
estimators = 100
randForest = RandomForestRegressor(n_estimators=estimators, random_state=6243, oob_score=True)
randForest.fit(X_train, y_train)
rf_pred = randForest.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
print("Random Forest MSE:", rf_mse)
print("Random Forest RMSE:", rf_rmse)

# calculating and displaying feature importance
feat_imp = randForest.feature_importances_
feature_imp_df = pd.DataFrame(feat_imp, index=cols, columns=["Importance"])
feature_imp_df = feature_imp_df.sort_values(by="Importance", ascending=True)
feature_imp_df.plot(kind="barh", legend=False)
plt.savefig("figures/random_forest_feature_importance.png")
plt.show()

#storing data
pred_df = pd.read_csv("data_processed/model_predictions.csv")
pred_df["RandomForest"] = rf_pred
pred_df.to_csv("data_processed/model_predictions.csv", index=False)