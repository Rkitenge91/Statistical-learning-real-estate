#Manual Ridge Regression

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

#
standardize = StandardScaler()
#alpha = 0.5
df_name = "data_processed/feature_engineered_real_estate.csv"
target_name = "PriceRatio"

alphas = [0.1, 1, 10, 100, 200, 500, 1000]

#
def data_process(name, target):
    df = pd.read_csv(name)
    y = df[target].copy()
    X = df.drop(columns=[target, "TotalAppraisedValue"]).copy()
    return X, y

#
def add_intercept(x):
    df_x = pd.DataFrame(x)
    df_x["intercept"] = 1
    return np.array(df_x)

#
def ridge_fit(x, y, alpha):
    penalty = alpha * np.identity(x.shape[1])
    penalty[0][0] = 0
    w = np.linalg.inv(x.T @ x + penalty) @ x.T @ y
    return np.array(w)

#
def ridge_predict(x, w):
    return x @ w

#
def MSE(y_pred, y_actual):
    n = y_pred.size
    y_pred = y_pred.tolist()
    y_actual = y_actual.tolist()
    sse = 0
    for i in range(n):
        sse += (y_actual[i] - y_pred[i])**2
    return sse/n


#
X, y = data_process(df_name, target_name)
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6243)
#
X_train_standard = standardize.fit_transform(X_train)
X_test_standard = standardize.transform(X_test)
#
X_train_stand_int = add_intercept(X_train_standard)
X_test_stand_int = add_intercept(X_test_standard)

#
mse_list = []
for a in alphas:
    ridge_w = ridge_fit(X_train_stand_int, y_train, a)
    ridge_pred = ridge_predict(X_test_stand_int, ridge_w)
    mse_list.append(MSE(ridge_pred, y_test))
#print(mse_list)

#
best_alpha = 100
ridge_w_best = ridge_fit(X_train_stand_int, y_train, best_alpha)
ridge_pred_best = ridge_predict(X_test_stand_int, ridge_w_best)
ridge_mse_best = MSE(ridge_pred_best, y_test)



# Sanity Check
#
ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train_standard, y_train)
#
ridge_w_builtin = np.hstack([ridge.coef_, ridge.intercept_])
ridge_pred_builtin = ridge.predict(X_test_standard)
ridge_builtin_mse = MSE(ridge_pred_builtin, y_test)

print(ridge_mse_best)
print(ridge_builtin_mse)


pred_df = pd.read_csv("model_predictions.csv")
pred_df["RidgeManual"] = ridge_pred_best
pred_df["RidgeBuiltin"] = ridge_pred_builtin
pred_df.to_csv("model_predictions.csv", index=False)