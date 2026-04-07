import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data_processed/feature_engineered_real_estate.csv")

X = df.drop(columns=["PriceRatio"])
X = np.array(X)
y = df["PriceRatio"]
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6243)

estimators = 100
randForest = RandomForestRegressor(n_estimators=100, random_state=6243, oob_score=True)
randForest.fit(X_train, y_train)
rf_pred = randForest.predict(X_test)