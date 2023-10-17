
import psycopg2
from vantage6.tools.util import info
#from constants import *
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import math
import sys
import os
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '../v6_LinReg_py'))

from .constants import *
def build_dataframe(cols, PG_URI = None):

    info(f'building dataframe with columns: {cols}')
    data_df = pd.DataFrame()

    if PG_URI == None:
            PG_URI = 'postgresql://{}:{}@{}:{}'.format(
                os.getenv("PGUSER"),
                os.getenv("PGPASSWORD"),
                os.getenv("PGHOST"),
                os.getenv("PGPORT"))
    info("connecting to PG DB:" + str(PG_URI))
    try:
        connection = psycopg2.connect(PG_URI)
        cursor = connection.cursor()
        info("connected to PG database. building dataframe..")

        #incrementally build dataframe
        for col in cols:
            data_pg = cursor.execute("SELECT {} FROM ncdc".format(col))
            column_data = cursor.fetchall()

            #fetchall returns a tuple for some reason, so we have to do this
            data_df[col] = [column_val[0] for column_val in column_data]
        
        # also add id to the initial dataframe, so we can sort using that
        data_pg = cursor.execute("SELECT {} FROM ncdc".format(ID))
        column_data = cursor.fetchall()
        data_df[ID] = [column_val[0] for column_val in column_data]

    except psycopg2.Error as e:
        info(f"psycopg2 error: {e}")
        return e
    
    # merge rows from same patients (but different visits)
    data_df = data_df.groupby([ID]).agg({col : 'first' for col in cols}).reset_index()

    data = data_df[cols].dropna()
    info("base dataframe built.")

    return data

def complete_dataframe(df, cols_to_add, use_deltas = False):
    info("adding columns based on covariates")

    if (LAG_TIME in cols_to_add) or (AGE in cols_to_add) or (use_deltas == True):
        #calculate age at blood draw and mri scan
        cols_for_tmp_df = [DATE_METABOLOMICS, DATE_MRI, BIRTH_YEAR]
        tmp_df = build_dataframe(cols_for_tmp_df)
        blood_dates = np.array([date.strftime("%Y") for date in tmp_df[DATE_METABOLOMICS].values]).astype(int)
        mri_dates = np.array([date.strftime("%Y") for date in tmp_df[DATE_MRI].values]).astype(int)

        Age_met = blood_dates - tmp_df[BIRTH_YEAR].values
        Age_brain = mri_dates- tmp_df[BIRTH_YEAR].values
        if LAG_TIME in cols_to_add:
            df[LAG_TIME] = Age_met - Age_brain
        if (AGE in cols_to_add):
            df[AGE] = np.mean(np.vstack((Age_met, Age_brain)), axis = 0)
        if (use_deltas == True):
            age = np.mean(np.vstack((Age_met, Age_brain)), axis = 0)
            df[METABO_AGE] = df[METABO_AGE].values.astype(float) - age
            df[BRAIN_AGE] = df[BRAIN_AGE].values.astype(float) - age


    if EDUCATION_CATEGORY in cols_to_add:
        tmp_df = build_dataframe([EDUCATION_CATEGORY])
        enc = OneHotEncoder(categories = [[0, 1, 2]], sparse=False)
        mapped_names = [EC1, EC2, EC3]
        info("creating mapped array")
        mapped_arr = enc.fit_transform(tmp_df[EDUCATION_CATEGORY].values.reshape(-1, 1))
        info("created mapped array")
        df[mapped_names] = mapped_arr

    if SENSITIVITY_1 in cols_to_add:
        df = df.loc[abs(df[LAG_TIME]) <= 1].reset_index(drop=True)
    elif SENSITIVITY_2 in cols_to_add:
        df = df.loc[abs(df[LAG_TIME]) <= 2].reset_index(drop=True)

    info("dataframe finished")
    return df


def create_test_train_split(X, y, seed):
    # create a test train split based on seeded rng
    rng = np.random.default_rng(seed = seed)
    
    n_samples = X.shape[0]
    train_inds = rng.choice(n_samples, math.floor(n_samples* 0.8), replace=False)

    train_inds = np.sort(train_inds)
    train_mask = np.zeros((n_samples), dtype=bool)
    train_mask[train_inds] = True
    X_train = X[train_mask]
    X_test = X[~train_mask]
    
    y_train = y[train_mask]
    y_test = y[~train_mask]

    return X_train, X_test, y_train, y_test


def make_boxplot(ba, res, bin_width):
    info("binning residual")
    
    
    min_age = min(ba.values)
    max_age = max(ba.values)
      

    n_bins = math.ceil((max_age - min_age) / bin_width)

    binned_res = []
    bin_start = np.zeros(n_bins)
    
    bin_start[0] = min_age
    for b in range(n_bins - 1):

        bin_start[b + 1] = min_age + bin_width * (b + 1)
        #info(str(bin_start[b+1]))
        bin_idx = np.where(np.logical_and((ba >= bin_start[b]), (ba < bin_start[b+1])))[0]

        if len(bin_idx) < 2:
            #info("too little values for boxplot, skipping")
            binned_res.append([])
        else:
            bin_vals = list(res[bin_idx])
            binned_res.append(bin_vals)
    
    last_bin_idx = np.where(ba> bin_start[-1])[0]
    if len(last_bin_idx) < 2:
        #info("too little values for boxplot, skipping")
        binned_res.append([])
    else:
        binned_res.append(list(res[last_bin_idx]))

    info("making boxplot")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.boxplot(binned_res)

    return fig, ax, bp, bin_start


#determine which columns to normalize based on the data columns and normalization settings 
def det_norm_cols(data_settings):
    norm_cat = data_settings[NORM_CAT]
    data_cols = data_settings[DATA_COLS]
    if ~norm_cat:
        norm_cols = [col for col in data_cols if col not in CAT_COLS_VALUES]
    else:
        norm_cols = data_cols
    norm_cols.append(data_settings[TARGET])
    return norm_cols


def normalise(data, data_settings):
    norm_cols = det_norm_cols(data_settings)
    if data_settings[NORMALIZE] == "global":
        data = (data[norm_cols] - data_settings[GLOBAL_MEAN][norm_cols].squeeze()) / data_settings[GLOBAL_STD][norm_cols].squeeze()
    elif data_settings[NORMALIZE] == "local":
        data = (data[norm_cols] - data[norm_cols].mean())/data[norm_cols].std(ddof=0)
    elif data_settings[NORMALIZE] == "none" :
        data = data
    else:
        raise(ValueError(f"unknown value for 'normalize': {data_settings[NORMALIZE]}"))
    
    return data