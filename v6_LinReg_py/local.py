
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

from .local_constants import *
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
            # info(f"adding {col}")
            data_pg = cursor.execute("SELECT {} FROM ncdc".format(col))
            column_data = cursor.fetchall()

            #fetchall returns a tuple for some reason, so we have to do this
            data_df[col] = [column_val[0] for column_val in column_data]
        
        info("done adding columns")
        # # also add id to the initial dataframe, so we can sort using that
        # data_pg = cursor.execute("SELECT {} FROM ncdc".format(ID))
        # column_data = cursor.fetchall()
        # data_df[ID] = [column_val[0] for column_val in column_data]

    except psycopg2.Error as e:
        info(f"psycopg2 error: {e}")
        return e
    

    return data_df


def complete_dataframe(data_settings):
    
    # pull the defines out of data_settings
    model_cols = data_settings[MODEL_COLS]
    use_deltas = data_settings[USE_DELTAS]
    lag_time = data_settings[DEFINES][LAG_TIME_COL]
    age = data_settings[DEFINES][AGE_COL]
    date_metabolomics = data_settings[DEFINES][DATE_METABOLOMICS_COL]
    date_mri = data_settings[DEFINES][DATE_MRI_COL]
    birth_year = data_settings[DEFINES][BIRTH_YEAR_COL]
    metabo_age = data_settings[DEFINES][METABO_AGE_COL]
    brain_age = data_settings[DEFINES][BRAIN_AGE_COL]
    education_category = data_settings[DEFINES][EDUCATION_CATEGORY_COL]
    ec_list = data_settings[DEFINES][EDUCATION_CATEGORIES_LIST]
    id = data_settings[DEFINES][ID_COL]
    
    
    base_df = build_dataframe(data_settings[ALL_COLS])

    info("merging based on id")
    # merge rows from same patients (but different visits)
    data_df = base_df.groupby([id]).agg({col : 'first' for col in data_settings[ALL_COLS] if col != id}).reset_index()

    info("base dataframe built.")
    data_df = data_df.dropna()


    df = data_df[data_settings[DATA_COLS]].copy()
    # info(str(df[date_metabolomics].values))

    info("adding columns based on covariates")
    #cols_to_add = data_settings[SYNTH_COLS]

    if (lag_time in model_cols) or (age in model_cols) or (use_deltas == True):
        #calculate age at blood draw and mri scan
        #cols_for_tmp_df = [DATE_METABOLOMICS, DATE_MRI, BIRTH_YEAR]
        #tmp_df = build_dataframe(cols_for_tmp_df)
        blood_dates = np.array([date.strftime("%Y") for date in df[date_metabolomics].values]).astype(int)
        mri_dates = np.array([date.strftime("%Y") for date in df[date_mri].values]).astype(int)

        Age_met = blood_dates - df[birth_year].values
        Age_brain = mri_dates- df[birth_year].values
        if lag_time in model_cols:
            info("adding lag time")
            df[lag_time] = Age_met - Age_brain
        if (age in model_cols):
            info("adding age")
            df[age] = np.mean(np.vstack((Age_met, Age_brain)), axis = 0)
        if (use_deltas == True):
            info("calculating deltas")
            age = np.mean(np.vstack((Age_met, Age_brain)), axis = 0)
            df[metabo_age] = df[metabo_age].values.astype(float) - age
            df[brain_age] = df[brain_age].values.astype(float) - age

    if ec_list[0] in model_cols:
        info("adding education category")
        # tmp_df = build_dataframe([EDUCATION_CATEGORY])
        enc = OneHotEncoder(categories = [[0, 1, 2]], sparse=False)
        mapped_names = ec_list
        mapped_arr = enc.fit_transform(df[education_category].values.reshape(-1, 1))
        mapped_df = pd.DataFrame(data = mapped_arr, columns = mapped_names)

        df = pd.concat([df.reset_index(drop=True), mapped_df.reset_index(drop=True)], axis = 1, ignore_index=False)
        # data_settings[DATA_COLS].extend(mapped_names)
    if data_settings[SENS] == 1:
        info("selecting lag time < 1 year")
        df = df.loc[abs(df[lag_time]) <= 1].reset_index(drop=True)
    elif data_settings[SENS] == 2:
        info("selecting lag time < 2 years")
        df = df.loc[abs(df[lag_time]) <= 2].reset_index(drop=True)

    # info("bla")
    info(f'df size: {df.shape}')
    info(f'df columns: {df.columns}')
    # data = data_df[cols].dropna()
    data_df = df[data_settings[MODEL_COLS]]
    info("dataframe finished")
    return data_df


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
    model_cols = data_settings[MODEL_COLS]

    if norm_cat == False:
        norm_cols = [col for col in model_cols if col not in data_settings[CAT_COLS]]
    else:
        norm_cols = model_cols

    # bit of a hack, but w/e
    for col in data_settings[OPTION_COLS]:
        if col in norm_cols:
            norm_cols.remove(col)

    #norm_cols.append(data_settings[TARGET])
    return norm_cols


def normalise(data, data_settings):
    info("normalising")
    norm_cols = det_norm_cols(data_settings)
    if data_settings[NORMALIZE] == "global":
        data[norm_cols] = (data[norm_cols] - data_settings[GLOBAL_MEAN][norm_cols].squeeze()) / data_settings[GLOBAL_STD][norm_cols].squeeze()
    elif data_settings[NORMALIZE] == "local":
        std = data[norm_cols].std(ddof=0).squeeze().T
        std = std.replace(to_replace = 0, value = 1)
        data[norm_cols] = (data[norm_cols] - data[norm_cols].mean())/std
    elif data_settings[NORMALIZE] == "none" :
        data = data
    else:
        raise(ValueError(f"unknown value for 'normalize': {data_settings[NORMALIZE]}"))
    info("normalising complete")
    return data