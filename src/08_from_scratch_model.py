#Manual Ridge Regression

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

# setting values for later use
standardize = StandardScaler()
df_name = "data_processed/feature_engineered_real_estate.csv"
target_name = "PriceRatio"



# read in data and separate based on supplied criteria
def data_process(name, target):
    df = pd.read_csv(name)
    y = df[target].copy()
    X = df.drop(columns=[target, "TotalAppraisedValue"]).copy()
    return X, y
# adding an intercept column
def add_intercept(x):
    df_x = pd.DataFrame(x)
    df_x["intercept"] = 1
    return np.array(df_x)
# mean squared error calculation
def MSE(y_pred, y_actual):
    y_pred = np.array(y_pred)
    y_actual = np.array(y_actual)
    n = len(y_pred)
    sse = 0
    for i in range(n):
        sse += (y_actual[i] - y_pred[i])**2
    return sse/n


# reading in and separating data into features and target
X, y = data_process(df_name, target_name)
# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6243)
# standardize both train and test to train values
X_train_standard = standardize.fit_transform(X_train)
X_test_standard = standardize.transform(X_test)
# adding an intercept column
X_train_stand_int = add_intercept(X_train_standard)
X_test_stand_int = add_intercept(X_test_standard)


# closed-form ridge fitting
def ridge_fit(x, y, alpha):
    penalty = alpha * np.identity(x.shape[1])
    penalty[-1][-1] = 0
    w = np.linalg.inv(x.T @ x + penalty) @ x.T @ y
    return np.array(w)
# predicting from calcuated weights
def ridge_predict(x, w):
    return x @ w



# tuning the alpha parameter
alphas = [0.1, 1, 10, 50, 70, 100, 200, 500, 1000]
mse_list = []
for a in alphas:
    ridge_w = ridge_fit(X_train_stand_int, y_train, a)
    ridge_pred = ridge_predict(X_test_stand_int, ridge_w)
    mse_list.append(MSE(ridge_pred, y_test))

#print(mse_list)

best_alpha = alphas[np.argmin(mse_list)]
# fitting the closed-form model with the optimal alpha value
w_cf = ridge_fit(X_train_stand_int, y_train, best_alpha)
pred_cf = ridge_predict(X_test_stand_int, w_cf)
mse_cf = MSE(pred_cf, y_test)



# Sanity Check
# fitting an sklearn ridge model with same alpha
ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train_standard, y_train)
# computing evaluation data to compare to closed-form and gd
w_skl = np.hstack([ridge.coef_, ridge.intercept_])
pred_skl = ridge.predict(X_test_standard)
mse_skl = MSE(pred_skl, y_test)

# gradient descent for ridge regression
def gd(Phi, y, w_gd, alpha, eta=0.005, epochs=2000):
    n = Phi.shape[0]
    y = np.array(y)
    for i in range(epochs):
        penalty = 2 * alpha * w_gd
        penalty[-1] = 0
        df = (2/n)*Phi.T @ (Phi @ w_gd - y) + penalty
        w_gd = w_gd - eta*df
    return w_gd

# setting original weights
w_init = np.ones(X_train_stand_int.shape[1])
# tuning to find best step size
gd_mse_list = []
etas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
for eta in etas:
    w = gd(X_train_stand_int, y_train, w_init, best_alpha, eta=eta)
    gd_pred = X_test_stand_int @ w
    mse = MSE(gd_pred, y_test)
    gd_mse_list.append(mse)

#print(gd_mse_list)

best_eta = etas[np.argmin(gd_mse_list)]
# computing outputs for best step size
w_gd = gd(X_train_stand_int, y_train, w_init, best_alpha, eta=best_eta)
pred_gd = X_test_stand_int @ w_gd
mse_gd = MSE(gd_pred, y_test)

# comparing MSE
print("Gradient Descent MSE:", mse_gd)
print("Closed Form MSE", mse_cf)
print("Builtin Model MSE", mse_skl)

# storing data
pred_df = pd.read_csv("data_processed/model_predictions.csv")
pred_df["RidgeClosedForm"] = pred_cf
pred_df["RidgeBuiltin"] = pred_skl
pred_df["RidgeGD"] = pred_gd
pred_df.to_csv("data_processed/model_predictions.csv", index=False)