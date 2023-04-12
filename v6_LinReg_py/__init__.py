from sklearn.linear_model import SGDRegressor
import numpy as np
from vantage6.tools.util import info
import math




def master():
    pass

def RPC_fit_round(data, coefs, intercepts, cols, lr, seed):

    #TODO: cross-val?
    #TODO: calc metaboage/health through PHT
    all_cols =  ["metabo_age", "Age", "sex", "Lag_time", "Diabetes_Mellitus", "BMI", "education_category"]
    data = data[all_cols].dropna()

    blood_dates = data["date_blood"].values
    mri_dates = data["date_mri"].values

    blood_dates_years = np.array([int(blood_date[:4]) for blood_date in blood_dates])
    mri_dates_years = np.array([int(mri_date[:4]) for mri_date in mri_dates])

    Age_met = blood_dates_years - data["Birth_year"].values
    Age_brain = mri_dates_years - data["Birth_year"].values

    data["Lag_time"] = Age_met - Age_brain
    data["Age"] = np.mean(np.vstack((Age_met, Age_brain)), axis = 0)


    if "Sens_1" in cols:
        data = data.loc[data['Lag_time'] <= 1]
        cols.remove("Sens_1")
    if "Sens_2" in cols:
        data = data.loc[data["Lag_time"] <= 2]
        cols.remove("Sens_2")

    rng = np.random.default_rng(seed = seed)
    
    X = data[cols].values.reshape(-1, 1)
    y = data["brainAge_sim"].values.reshape(-1, 1)
    train_inds = rng.choice(len(X), math.floor(len(X)* 0.8), replace=False)
    train_inds = np.sort(train_inds)
    train_mask = np.zeros((len(X)), dtype=bool)
    train_mask[train_inds] = True
    X_train = X[train_mask]
    X_test = X[~train_mask]
    
    y_train = y[train_mask]
    y_test = y[~train_mask]
    #info("train inds: " + str(train_inds))
    #info("train shape: " + str(X_train.shape))
    #info("test shape: " + str(X_test.shape))
    #info("X full shape: " + str(X.shape))
    

    #info("data loaded")

    model = SGDRegressor(loss="squared_error", penalty=None, max_iter = 1, eta0=lr)
    
    model.coef_ = np.copy(coefs)
    model.intercept_ = np.copy(intercepts)
    
    #info("training data shape: " + str(y_train.shape))

    model.partial_fit(X_train, y_train)

    #info("model fitted")


    loss = np.mean((model.predict(X_test) - y_test) **2)

    #TODO: how do we make sure which coefs correspond to which covariates?
    return {
        "param": (model.coef_, model.intercept_),
        "loss": loss,
        "size": y_test.shape[0]
    }
    

