from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from vantage6.tools.util import info
import math
import os
import psycopg2
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt



ALL_COLS = ["id", "metabo_age", "brain_age", "date_metabolomics", "date_mri", "birth_year", "sex", "dm", "bmi", "education_category_3"]
CAT_COLS = ['education_category_3', 'sex', "dm"]

def master():
    pass

def construct_data(all_cols, data_cols, extra_cols, normalize = 'none', PG_URI = None, cat_cols = CAT_COLS, global_mean = 0, global_std = 1, use_deltas = False):
    data_df = pd.DataFrame()


    # could also use this to set URI
    if PG_URI == None:
        PG_URI = 'postgresql://{}:{}@{}:{}'.format(
            os.getenv("PGUSER"),
            os.getenv("PGPASSWORD"),
            os.getenv("PGHOST"),
            os.getenv("PGPORT"))

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

    # merge rows from same patients (but different visits)

    merge_cols = all_cols.copy()
    merge_cols.remove("id")
    data_df = data_df.groupby(["id"]).agg({col : 'first' for col in merge_cols}).reset_index()

    data = data_df[all_cols].dropna()
    info("base dataframe built. adding columns based on covariates..")

    if ("Lag_time" in extra_cols) or ("Age" in extra_cols) or (use_deltas == True):
        #calculate age at blood draw and mri scan
        blood_dates = np.array([date.strftime("%Y") for date in data["date_metabolomics"].values]).astype(int)
        mri_dates = np.array([date.strftime("%Y") for date in data["date_mri"].values]).astype(int)

        Age_met = blood_dates - data["birth_year"].values
        Age_brain = mri_dates- data["birth_year"].values
        if "Lag_time" in extra_cols:
            data["Lag_time"] = Age_met - Age_brain
            data_cols.append("Lag_time")
        if ("Age" in extra_cols) or (use_deltas == True):
            data["Age"] = np.mean(np.vstack((Age_met, Age_brain)), axis = 0)
            data_cols.append("Age")

    if "Sens_1" in extra_cols:
        data = data.loc[abs(data['Lag_time']) <= 1].reset_index(drop=True)
    elif "Sens_2" in extra_cols:
        data = data.loc[abs(data["Lag_time"]) <= 2].reset_index(drop=True)


    if use_deltas:
        data['metabo_age'] = data['metabo_age'].values.astype(float) - data['Age'].values.astype(float)
        data['brain_age'] = data['brain_age'].values.astype(float) - data['Age'].values.astype(float)

    if normalize != "none":

        norm_cols = [col for col in data_cols if col not in cat_cols]
        norm_cols.append('metabo_age')
        std = [None] # we need this to avoid issues with the if-statement later on
        mean = None
        if normalize == "local":
            mean = data[norm_cols].astype(float).mean()
            std = data[norm_cols].astype(float).std()
            
            #data[norm_cols] = tmp2.values
            #data[norm_cols] = (data[norm_cols].astype(float) - data[norm_cols].astype(float).mean())/ data[norm_cols].astype(float).std()
            #info("normalizing done")
        elif normalize == "global":
            mean = global_mean
            std = global_std

        if 0 in std:
            info("removing 0's from std")
            std[std==0] = 1

        info(str(norm_cols))
        info(str(mean.shape))
        data[norm_cols] = (data[norm_cols].astype(float) - mean) / std
        #data[norm_cols] = (data[norm_cols].astype(float) - data[norm_cols].astype(float).mean())/ data[norm_cols].astype(float).std()
        info("normalizing done")  

    if "education_category_3" in data_cols:
        enc = OneHotEncoder(categories = [[0, 1, 2]], sparse=False)
        mapped_names = ["ec_1", "ec_2", "ec_3"]
        mapped_arr = enc.fit_transform(data['education_category_3'].values.reshape(-1, 1))
        data[mapped_names] = mapped_arr
        data_cols.extend(mapped_names)
        data_cols.remove("education_category_3")


    return data, data_cols


def RPC_fit_round(db_client, coefs, intercepts, data_cols, extra_cols, lr, seed, PG_URI = None, all_cols = ALL_COLS, cat_cols = CAT_COLS, bin_width = 2, normalize = 'none', global_mean = None, global_std = None, use_deltas = False):


    info("starting fit_round")
    #TODO: calc metaboage/health through PHT

    
    data, data_cols = construct_data(all_cols, data_cols, extra_cols, PG_URI = PG_URI, cat_cols = cat_cols, normalize=normalize, global_mean = global_mean, global_std = global_std, use_deltas = use_deltas)
    
    X = data[data_cols].values.astype(float)



    if "mh" in extra_cols:
        y = data['metabo_health'].values.reshape(-1, 1,).astype(float)
    else:
        y = data["metabo_age"].values.reshape(-1, 1).astype(float)
    
    # create a test train split based on seeded rng
    rng = np.random.default_rng(seed = seed)
    train_inds = rng.choice(len(X), math.floor(len(X)* 0.8), replace=False)
    #info(str(train_inds))
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
    global_mse = np.mean((global_pred - y_test) **2)
    global_mae = np.mean(abs(global_pred - y_test))
    loss = np.mean((model.predict(X_test) - y_test) **2)

    info("binning residual")
    res = global_pred - y[:,0]
    
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
        "mse": global_mse,
        "mae" :global_mae,
        "size": y_train.shape[0],
        "resplot": {
            "fig" : fig,
            "ax" : ax,
            "bp" : bp,
            "ranges" : bin_start
        }
    }
   
def RPC_get_avg(db_client, data_cols, extra_cols, all_cols = ALL_COLS, PG_URI = None, use_deltas = False):


    data, data_cols = construct_data(all_cols, data_cols, extra_cols, PG_URI = PG_URI, normalize = 'none', use_deltas=use_deltas)
    values = data[data_cols].values.astype(float)
    info(str(data_cols))
    return{
        "mean" : np.mean(values, axis = 0),
        #"cols" : data_cols,
        "size" : values.shape[0]
    }

def RPC_get_std(db_client, global_mean, data_cols, extra_cols, PG_URI = None, all_cols = ALL_COLS, use_deltas = False):
    data, data_cols = construct_data(all_cols, data_cols, extra_cols, PG_URI = PG_URI, normalize = 'none', use_deltas = use_deltas)
    values = data[data_cols].values.astype(float)
    #info(str(values.shape) + "," +  str(global_mean.shape))
    std_part = np.sum(np.square(values - global_mean), axis = 0)

    return {
        "std_part" : std_part,
        #"cols" :data_cols
    }
    

def RPC_calc_se(db_client, global_mean, global_coefs, global_inter, data_cols, extra_cols, all_cols = ALL_COLS, PG_URI = None):


    data, data_cols = construct_data(all_cols, data_cols, extra_cols, PG_URI = PG_URI)
    ba = data["brain_age"].values.astype(float)

    X = data[data_cols].values.astype(float)
    y = data["metabo_age"].values.reshape(-1, 1).astype(float)
    

    model = SGDRegressor(loss="squared_error", penalty=None, max_iter = 1, eta0=0)
    
    model.coef_ = np.copy(global_coefs)
    model.intercept_ = np.copy(global_inter)
    
    global_pred = model.predict(X)

    top = (np.subtract(y[:,0] , global_pred))**2
    bot = (ba - global_mean)**2

    return(
        {
            "top" : top,
            "bot" : bot,
            "size" : y.shape[0]
        }
    )
