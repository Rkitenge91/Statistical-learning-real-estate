import pandas as pd
import dataframe_image as dfi
from sklearn.metrics import root_mean_squared_error

pred_df = pd.read_csv("data_processed/model_predictions.csv")
#print(pred_df.head())

y = pred_df["yActual"]
randf_rmse = root_mean_squared_error(y, pred_df["RandomForest"])
ridge_manual_rmse = root_mean_squared_error(y, pred_df["RidgeClosedForm"])
ridge_builtin_rmse = root_mean_squared_error(y, pred_df["RidgeBuiltin"])
ridge_gd_rmse = root_mean_squared_error(y, pred_df["RidgeGD"])

rmse_list = [randf_rmse, ridge_manual_rmse, ridge_builtin_rmse, ridge_gd_rmse]
rmse_df = pd.DataFrame(rmse_list, index=["RandomForest", "RidgeClosedForm", "RidgeBuiltin", "RidgeGradientDescent"],
columns=["RMSE"]
)
rmse_df.to_csv("reports/model_comparison_rmse.csv", index_label="Model")
dfi.export(rmse_df, "figures/model_comparison_rmse.png")
print(rmse_df)