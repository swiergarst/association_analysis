from sklearn.linear_model import SGDRegressor
import numpy as np
from vantage6.tools.util import info
import math
import os
import psycopg2
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def master():
    pass

def construct_data(all_cols, data_cols, extra_cols, PG_URI = None, ):
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


    info(str(data_df['dm'].values))
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

    return data


def RPC_fit_round(db_client, coefs, intercepts, data_cols, extra_cols, lr, seed, PG_URI = None, all_cols = [None], bin_width = 2):


    info("starting fit_round")
    #TODO: calc metaboage/health through PHT
    
    data_df = pd.DataFrame()

    # in case we want to use different columns later on, no need to change image this way
    if None in all_cols:
        all_cols =  ["id", "metabo_age", "brain_age", "date_metabolomics", "date_mri", "birth_year", "sex", "dm", "bmi", "education_category"]
    

    
    #info("calculated all covariates")




    data = construct_data(all_cols, data_cols, extra_cols, PG_URI = PG_URI)
    
    X = data[data_cols].values.astype(float)
    
    if "mh" in extra_cols:
        y = data['metabo_health'].values.reshape(-1, 1,).astype(float)
    else:
        y = data["metabo_age"].values.reshape(-1, 1).astype(float)
    

    rng = np.random.default_rng(seed = seed)
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
    
    global_pred = model.predict(X)

    model.partial_fit(X_train, y_train[:,0])

    info("model fitted")
    global_loss = np.mean((global_pred - y_test) **2)
    loss = np.mean((model.predict(X_test) - y_test) **2)

    info("creating boxplot values")
    #metabo_pred = model.predict(X)
    # sort_idx = np.argsort(y, axis = 0)
    res = global_pred - y[:,0]
    
    # info("sorting residual and y")
    # sorted_res = res[sort_idx]
    # sorted_y = y[sort_idx]


    info("binning residual")
    # for b in range(n_bins - 1):
    #     info("binning values " + str(b*vals_per_bin) +  " to " + str((b+1) * vals_per_bin))
    #     bin_vals = list(sorted_res[b * vals_per_bin : (b+1) * vals_per_bin,0])
    #     binned_res.append(bin_vals)
    #     bin_ranges[0,b] = sorted_y[b*vals_per_bin]
    #     bin_ranges[1,b] = sorted_y[(b+1) * vals_per_bin]

    # binned_res.append(list(sorted_res[(n_bins-1) * vals_per_bin:, 0]))
    # bin_ranges[0,-1] = bin_ranges[1, -2]
    # bin_ranges[1, -1] = sorted_y[-1]

    min_age = min(data['brain_age'].values)
    max_age = max(data['brain_age'].values)
        

    n_bins = math.ceil((max_age - min_age) / bin_width)

    binned_res = []
    bin_start = np.zeros(n_bins)
    
    bin_start[0] = min_age
    for b in range(n_bins - 1):

        bin_start[b + 1] = min_age + bin_width * (b + 1)
        #info(str(bin_start[b+1]))
        bin_idx = np.where(np.logical_and((y >= bin_start[b]), (y < bin_start[b+1])))[0]

        if len(bin_idx) < 2:
            #info("too little values for boxplot, skipping")
            binned_res.append([])
        else:
            bin_vals = list(res[bin_idx])
            binned_res.append(bin_vals)
    
    last_bin_idx = np.where(y> bin_start[-1])[0]
    if len(last_bin_idx) < 2:
        #info("too little values for boxplot, skipping")
        binned_res.append([])
    else:
        binned_res.append(list(res[last_bin_idx]))

    info("making boxplot")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.boxplot(binned_res)


    return {
        "param": (model.coef_, model.intercept_),
        "data_cols" : data_cols,
        "loss": global_loss,
        "size": y_train.shape[0],
        "resplot": {
            "fig" : fig,
            "ax" : ax,
            "bp" : bp,
            "ranges" : bin_start
        }
    }
    
def RPC_get_avg(db_client, PG_URI = None, col_name = "brain_age"):

    if PG_URI == None:
        PG_URI = 'postgresql://{}:{}@{}:{}'.format(
            os.getenv("PGUSER"),
            os.getenv("PGPASSWORD"),
            os.getenv("PGHOST"),
            os.getenv("PGPORT"))

    try:
        connection = psycopg2.connect(PG_URI)
        cursor = connection.cursor()
        tmp = cursor.execute("SELECT {} FROM ncdc".format(col_name))
        vals = cursor.fetchall()


    except psycopg2.Error as e:
        return e
    
    return{
        "mean" : np.mean(vals),
        "size" : vals.shape[0]
    }
    

def RPC_calc_se(db_client, global_mean, global_coefs, global_inter, data_cols, extra_cols, all_cols = [None], PG_URI = None):


    if None in all_cols:
        all_cols =  ["id", "metabo_age", "brain_age", "date_metabolomics", "date_mri", "birth_year", "sex", "dm", "bmi", "education_category"]
    

    data = construct_data(all_cols, data_cols, extra_cols, PG_URI = PG_URI)
    X = data["brain_age"].values.astype(float)
    y = data["metabo_age"].values.reshape(-1, 1).astype(float)
    

    model = SGDRegressor(loss="squared_error", penalty=None, max_iter = 1, eta0=lr)
    
    model.coef_ = np.copy(global_coefs)
    model.intercept_ = np.copy(global_inter)
    
    global_pred = model.predict(X)

    top = (y - global_pred)**2
    bot = (X - global_mean)**2
