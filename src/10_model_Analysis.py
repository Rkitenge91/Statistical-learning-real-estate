import pandas as pd
import matplotlib.pyplot as plt

# load predictions
pred_df = pd.read_csv("data_processed/model_predictions.csv")

# predicted vs actual (use best model → Ridge)
plt.scatter(pred_df["yActual"], pred_df["RidgeClosedForm"], alpha=0.5)
plt.xlabel("Actual Price Ratio")
plt.ylabel("Predicted Price Ratio")
plt.title("Predicted vs Actual (Ridge)")
plt.plot([0.5,1.5], [0.5,1.5], color="red")  # perfect line
plt.savefig("figures/pred_vs_actual_ridge.png")
plt.show()



# residuals (Ridge)
residuals = pred_df["yActual"] - pred_df["RidgeClosedForm"]

plt.scatter(pred_df["RidgeClosedForm"], residuals, alpha=0.5)
plt.axhline(0)  # horizontal line at 0
plt.xlabel("Predicted Price Ratio")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot (Ridge)")
plt.savefig("figures/residuals_ridge.png")
plt.show()