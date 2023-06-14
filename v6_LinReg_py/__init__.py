from sklearn.linear_model import SGDRegressor
import numpy as np
from vantage6.tools.util import info
import math
import os
import psycopg2
import pandas as pd

def master():
    pass

def RPC_fit_round(data, coefs, intercepts, data_cols, extra_cols, lr, seed, PG_URI = None, all_cols = [None]):


    #TODO: calc metaboage/health through PHT
    
    data_df = pd.DataFrame()

    # in case we want to use different columns later on, no need to change image this way
    if None in all_cols:
        all_cols =  ["metabo_age", "brain_age", "date_blood", "date_mri", "birth_year", "sex", "dm", "bmi", "education_category"]
    
    # could also use this to set URI
    if PG_URI == None:
        PG_URI = 'postgresql://{}:{}@{}:{}/{}'.format(
                os.getenv("DB_USER"),
                os.getenv("DB_PASSWORD"),
                os.getenv("DB_HOST"),
                os.getenv("DB_PORT"))
    
    # get data from postgres DB
    try:
        connection = psycopg2.connect(PG_URI)
        cursor = connection.cursor()

        #incrementally build dataframe
        for col in all_cols:
            data_pg = cursor.execute("SELECT {} FROM ncdc".format(col))
            data_df[col] = data_pg.fetchall()
    except psycopg2.Error as e:
        print(e)


    data = data_df[all_cols].dropna()

    blood_dates = data["date_blood"].values
    mri_dates = data["date_mri"].values

    blood_dates_years = np.array([int(blood_date[:4]) for blood_date in blood_dates])
    mri_dates_years = np.array([int(mri_date[:4]) for mri_date in mri_dates])

    Age_met = blood_dates_years - data["birth_year"].values
    Age_brain = mri_dates_years - data["birth_year"].values

    if "Lag_time" in extra_cols:
        data["Lag_time"] = Age_met - Age_brain
        data_cols.append("Lag_time")
    if "Age" in extra_cols:
        data["Age"] = np.mean(np.vstack((Age_met, Age_brain)), axis = 0)
        data_cols.append("Age")


    if "Sens_1" in extra_cols:
        data = data.loc[abs(data['Lag_time']) <= 1]
        #cols.remove("Sens_1")
    if "Sens_2" in extra_cols:
        data = data.loc[abs(data["Lag_time"]) <= 2]
        #cols.remove("Sens_2")

    rng = np.random.default_rng(seed = seed)
    
    X = data[data_cols].values.reshape(-1, 1)
    y = data["brain_age"].values.reshape(-1, 1)
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
    

