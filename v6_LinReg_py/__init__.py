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
    data = complete_dataframe(data_tmp, data_settings[SYNTH_COLS])

    data = normalise(data, data_settings)
    X_full = data.drop(data_settings[TARGET])
    y_full = data[data_settings[TARGET]]

    X_train, X_test, y_train, y_test = create_test_train_split(X_full, y_full, classif_settings[SEED])

    model = SGDRegressor(loss="squared_error", penalty=None, max_iter = 1, eta0=classif_settings[LR], fit_intercept=False)
    model.coef_ = classif_settings[COEF]

    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    full_pred = model.predict(X_train)


    train_mae = np.mean(abs(train_pred - y_train))
    test_mae = np.mean(abs(test_pred - y_test))
    test_loss = np.mean((model.predict(X_test) - y_test) **2)

    model.partial_fit(X_train, y_train[:,0])
    return_params = pd.DataFrame(data = model.coef_, columns = model.feature_names_in_)

    residual = full_pred - y_full[:,0]
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

    data = complete_dataframe(data_tmp, data_settings[SYNTH_COLS], data_settings[USE_DELTAS])

    return {"averages" : data.mean().to_frame().T,
            "size" : data.shape[0]
            }


def RPC_get_std(db_client, data_settings: dict, global_mean: pd.DataFrame):
    data_tmp = build_dataframe(data_settings[DIRECT_COLS])

    data = complete_dataframe(data_tmp, data_settings[SYNTH_COLS],data_settings[USE_DELTAS])

    partial_std = ((data.astype(float) - global_mean.squeeze())**2).sum(axis = 0)

    return {
        "std" : partial_std.to_frame().T,
        #"std" : tmp3.to_frame().T,
        "size" : data.shape[0]
    }

