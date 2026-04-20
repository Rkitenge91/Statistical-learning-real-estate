import pandas as pd
from sklearn.metrics import root_mean_squared_error

pred_df = pd.read_csv("data_processed/model_predictions.csv")
print(pred_df.head())
print(pred_df.columns)

y = pred_df["Actual"].copy()

randf_rmse = root_mean_squared_error(y, pred_df["RandomForest"])
ridge_manual_rmse = root_mean_squared_error(y, pred_df["RidgeClosedForm"])
ridge_builtin_rmse = root_mean_squared_error(y, pred_df["RidgeBuiltin"])
ridge_gd_rmse = root_mean_squared_error(y, pred_df["RidgeGD"])

rmse_list = [randf_rmse,ridge_manual_rmse, ridge_builtin_rmse, ridge_gd_rmse]
rmse_df = pd.DataFrame(rmse_list, index=["RandomForestRMSE", "RidgeClosedFormRMSE", "RidgeBuiltinRMSE", "RidgeGradientDescentRMSE"],
columns=["RMSE"]
)
rmse_df.to_csv("reports/model_comparison_rmse.csv", index_label="Model")
print(rmse_df)