import psycopg2
from vantage6.tools.util import info
#from constants import *
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDRegressor
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../v6_LinReg_py'))
import math
from .constants import *
from .local import build_dataframe, complete_dataframe, create_test_train_split, make_boxplot, normalise


def RPC_train_round(db_client, data_settings, classif_settings):
    info("starting train_round")
    data_tmp = build_dataframe(data_settings[DIRECT_COLS])
    data = complete_dataframe(data_tmp, data_settings)
    info(str(data.columns))
    data = normalise(data.astype(float), data_settings)

    y_full = data[data_settings[TARGET]]
    X_full = data.drop(columns = data_settings[TARGET])

    info("creating test/train split")
    X_train, X_test, y_train, y_test = create_test_train_split(X_full, y_full, classif_settings[SEED])

    model = SGDRegressor(loss="squared_error", penalty=None, max_iter = 1, eta0=classif_settings[LR], fit_intercept=False)
    model.coef_ = classif_settings[COEF].values[0,:]
    model.feature_names_in_ = classif_settings[COEF].columns
    model.intercept_ = [0]

    # info(str(model.feature_names_in_))
    # info(str(X_train.columns))

    info("calculating global predictions/errors")
    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    full_pred = model.predict(X_full)


    train_mae = np.mean(abs(train_pred - y_train))
    test_mae = np.mean(abs(test_pred - y_test))
    test_loss = np.mean((model.predict(X_test) - y_test) **2)

    info("fitting model")
    info(str(X_train))
    info(str(y_train))
    model.partial_fit(X_train, y_train.values)
    info("model fitted")
    return_params = pd.DataFrame(data = [model.coef_], columns = model.feature_names_in_)

    info("creating boxplots")
    residual = full_pred - y_full.values
    fig, ax, bp, bin_start = make_boxplot(X_full[BRAIN_AGE], residual, data_settings[BIN_WIDTH_BOXPLOT])
    return {
        LOCAL_COEF: return_params,
        TRAIN_MAE : train_mae,
        TEST_MAE : test_mae,
        TEST_LOSS : test_loss,
        LOCAL_TRAIN_SIZE : y_train.shape[0],
        BP: {
            "fig" : fig,
            "ax" : ax,
            "bp" : bp,
            "bin_start" : bin_start
        }
    }


def RPC_get_avg(db_client, data_settings):
    data_tmp = build_dataframe(data_settings[DIRECT_COLS])

    data = complete_dataframe(data_tmp, data_settings)

    return {"averages" : data.mean().to_frame().T.astype(float),
            "size" : data.shape[0]
            }


def RPC_get_std(db_client, data_settings: dict, global_mean: pd.DataFrame):
    data_tmp = build_dataframe(data_settings[DIRECT_COLS])

    data = complete_dataframe(data_tmp, data_settings)

    #info(str(data.columns))
    partial_std = ((data.astype(float) - global_mean.astype(float).squeeze())**2).sum(axis = 0)

    return {
        "std" : partial_std.to_frame().T,
        #"std" : tmp3.to_frame().T,
        "size" : data.shape[0]
    }

def RPC_calc_se(db_client, data_settings: dict, classif_settings: dict):
    data_tmp = build_dataframe(data_settings[DIRECT_COLS])

    data = complete_dataframe(data_tmp, data_settings)

    data = normalise(data.astype(float), data_settings)
    X_full = data.drop(columns = data_settings[TARGET])
    y_full = data[data_settings[TARGET]]


    # if data_settings[NORMALIZE] == "global":
    #     mean = data_settings[GLOBAL_MEAN]
    # else:
    #     mean = X_full.mean()

    # mean = mean.drop(columns = data_settings[TARGET])
    model = SGDRegressor(loss="squared_error", penalty=None, max_iter = 1, eta0=0, fit_intercept=False)
    
    model.coef_ = np.copy(classif_settings[COEF])
    model.intercept_ = [0]
    global_pred = model.predict(X_full)

    top = (np.subtract(y_full.values , global_pred))**2
    # bot = (X_full - mean.squeeze())**2

    cov = X_full.T @ X_full
    
    return {
            TOP : top,
            BOT : cov,
            #"data_cols" : X_full.columns,
            SIZE : y_full.shape[0]
    }

def RPC_calc_ABC(db_client, data_settings: dict): 
    data_tmp = build_dataframe(data_settings[DIRECT_COLS])
    data = complete_dataframe(data_tmp, data_settings)
    data = normalise(data.astype(float), data_settings)

    X = data.drop(columns = data_settings[TARGET]).values
    y = data[data_settings[TARGET]].values

    A = X.T @ X
    B = X.T @ y
    C = y.T @ y

    return {
        "A" : A,
        "B" : B,
        "C" : C, 
        SIZE : X.shape[0]
    }