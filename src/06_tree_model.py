import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

df = pd.read_csv("data_processed/feature_engineered_real_estate.csv")

X = df.drop(columns=["PriceRatio", "TotalAppraisedValue"]).copy()
cols = X.columns
X = np.array(X)
y = df["PriceRatio"].copy()
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6243)

estimators = 100
randForest = RandomForestRegressor(n_estimators=estimators, random_state=6243, oob_score=True)
randForest.fit(X_train, y_train)
rf_pred = randForest.predict(X_test)

feat_imp = randForest.feature_importances_
feature_imp_df = pd.DataFrame(feat_imp, index=cols)
feature_imp_df.plot(kind="barh")
plt.show()

pred_dict = {"yActual": y_test, "RandomForest": rf_pred}
pred_df = pd.DataFrame(pred_dict)
pred_df.to_csv("model_predictions.csv", index=False)