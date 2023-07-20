from sklearn.linear_model import SGDRegressor
import numpy as np
from vantage6.tools.util import info
import math
import os
import psycopg2
import pandas as pd
from datetime import datetime

def master():
    pass

def RPC_fit_round(db_client, coefs, intercepts, data_cols, extra_cols, lr, seed, PG_URI = None, all_cols = [None]):


    info("starting fit_round")
    #TODO: calc metaboage/health through PHT
    
    data_df = pd.DataFrame()

    # in case we want to use different columns later on, no need to change image this way
    if None in all_cols:
        all_cols =  ["id", "metabo_age", "brain_age", "date_metabolomics", "date_mri", "birth_year", "sex", "dm", "bmi", "education_category"]
    
    # could also use this to set URI
    if PG_URI == None:
        PG_URI = 'postgresql://{}:{}@{}:{}'.format(
            os.getenv("PGUSER"),
            os.getenv("PGPASSWORD"),
            os.getenv("PGHOST"),
            os.getenv("PGPORT"))

       # PG_URI = 'postgresql://{}:{}@{}:{}'.format(
       #         os.getenv("DB_USER"),
       ##         os.getenv("DB_PASSWORD"),
       #         os.getenv("DB_HOST"),
       #         os.getenv("DB_PORT"))
   
    info("connecting to PG DB:" + str(PG_URI))
    # get data from postgres DB
    try:
        connection = psycopg2.connect(PG_URI)
        cursor = connection.cursor()
        info("connected to PG database. building dataframe..")

        #incrementally build dataframe
        for col in all_cols:
            data_pg = cursor.execute("SELECT {} FROM ncdc".format(col))
            column_data = cursor.fetchall()

            #fetchall returns a tuple for some reason, so we have to do this
            data_df[col] = [column_val[0] for column_val in column_data]
    except psycopg2.Error as e:
        return e


    merge_cols = all_cols.copy()
    merge_cols.remove("id")
    data_df = data_df.groupby(["id"]).agg({col : 'first' for col in merge_cols}).reset_index()

    data = data_df[all_cols].dropna()
    info("dataframe built")



    blood_dates = np.array([date.strftime("%Y") for date in data["date_metabolomics"].values]).astype(int)
    mri_dates = np.array([date.strftime("%Y") for date in data["date_mri"].values]).astype(int)

    Age_met = blood_dates - data["birth_year"].values
    Age_brain = mri_dates- data["birth_year"].values

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

    
    info("calculated all covariates")


    rng = np.random.default_rng(seed = seed)
    
    X = data[data_cols].values
    y = data["brain_age"].values.reshape(-1, 1)

    train_inds = rng.choice(len(X), math.floor(len(X)* 0.8), replace=False)
    train_inds = np.sort(train_inds)
    train_mask = np.zeros((len(X)), dtype=bool)
    train_mask[train_inds] = True
    X_train = X[train_mask]
    X_test = X[~train_mask]
    
    y_train = y[train_mask]
    y_test = y[~train_mask]

    model = SGDRegressor(loss="squared_error", penalty=None, max_iter = 1, eta0=lr)
    
    model.coef_ = np.copy(coefs)
    model.intercept_ = np.copy(intercepts)
    
    model.partial_fit(X_train.astype(float), y_train[:,0].astype(int))

    info("model fitted")


    loss = np.mean((model.predict(X_test) - y_test.astype(int)) **2)

    return {
        "param": (model.coef_, model.intercept_),
        "data_cols" : data_cols,
        "loss": loss,
        "size": y_train.shape[0]
    }
    
