import pandas as pd
from sklearn.metrics import root_mean_squared_error

pred_df = pd.read_csv("model_predictions.csv")
print(pred_df.head())

randf_rmse = root_mean_squared_error(pred_df["yActual"], pred_df["RandomForest"])
ridge_manual_rmse = root_mean_squared_error(pred_df["yActual"], pred_df["RidgeManual"])
ridge_builtin_rmse = root_mean_squared_error(pred_df["yActual"], pred_df["RidgeBuiltin"])

rmse_list = [randf_rmse, ridge_manual_rmse, ridge_builtin_rmse]
rmse_df = pd.DataFrame(rmse_list, index=["RandomForestRMSE", "RidgeManualRMSE", "RidgeBuiltinRMSE"])
print(rmse_df.head())


