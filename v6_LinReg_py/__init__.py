import psycopg2
from vantage6.tools.util import info
#from constants import *
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier, SGDRegressor, LinearRegression
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../v6_LinReg_py'))
import math
from .local_constants import *
from .local import build_dataframe, complete_dataframe, create_test_train_split, make_boxplot, normalise



def RPC_train_round(db_client, data_settings, classif_settings):
    info("starting train_round")
    data = complete_dataframe(data_settings)

    # info(str(data.columns))
    data = normalise(data.astype(float), data_settings)

    if data_settings[STRATIFY] == True:
        for strat_col, strat_val in zip(data_settings[STRATIFY_GROUPS], data_settings[STRATIFY_VALUES]):
            data = data.loc(data[strat_col] == strat_val)

    y_full = data[data_settings[TARGET]]
    X_full = data.drop(columns = data_settings[TARGET])

    info("creating test/train split")
    X_train, X_test, y_train, y_test = create_test_train_split(X_full, y_full, classif_settings[SEED])

    model = SGDRegressor(loss="squared_error", penalty=None, max_iter = 1, eta0=classif_settings[LR], fit_intercept=True)
    # model = LinearRegression(fit_intercept= False, warm_start = True)
    model.coef_ = classif_settings[COEF].values[0,:]
    model.feature_names_in_ = classif_settings[COEF].columns
    model.intercept_ = [0]

    # info(str(model.feature_names_in_))
    # info(str(X_train.columns))

    info("calculating global predictions/errors")
    # test_pred = model.predict(X_test)
    # train_pred = model.predict(X_train)
    full_pred = model.predict(X_full)


    # train_mae = np.mean(abs(train_pred - y_train))
    # test_mae = np.mean(abs(test_pred - y_test))
    full_mae = np.mean(abs(full_pred - y_full))

    # test_loss = np.mean((model.predict(X_test) - y_test) **2)

    info("fitting model")
    # info(str(X_train))
    # info(str(y_train))
    model.partial_fit(X_full, y_full.values)
    # model.fit(X_full, y_full.values)

    info("model fitted")
    return_params = pd.DataFrame(data = [model.coef_], columns = model.feature_names_in_)

    info("creating boxplots")
    residual = full_pred - y_full.values
    fig, ax, bp, bin_start = make_boxplot(X_full[data_settings[BP_1]], residual, data_settings[BIN_WIDTH_BOXPLOT])
    return {
        LOCAL_COEF: return_params,
        "full_mae" : full_mae,
        # TRAIN_MAE : train_mae,
        # TEST_MAE : test_mae,
        # TEST_LOSS : test_loss,
        LOCAL_TRAIN_SIZE : y_full.shape[0],
        BP: {
            "fig" : fig,
            "ax" : ax,
            "bp" : bp,
            "bin_start" : bin_start
        }
    }


## TODO: implement
def RPC_predict_disease(db_client, data_settings, classif_settings):
    data = complete_dataframe(data_settings)

    # info(str(data.columns))
    data = normalise(data.astype(float), data_settings)

    # shouldn't ever do this for classification I think, but just in case    
    if data_settings[STRATIFY] == True:
        for strat_col, strat_val in zip(data_settings[STRATIFY_GROUPS], data_settings[STRATIFY_VALUES]):
            data = data.loc(data[strat_col] == strat_val)


    label_columns = data[data_settings[CLASSIF_TARGETS]].values
    y_full = np.array([label_column * 2**i  for i, label_column in enumerate(label_columns)]) # this way we can combine multiple columns. only works for binary outcomes
    X_full = data.drop(columns = data_settings[CLASSIF_TARGETS])
    info("creating test/train split")
    X_train, X_test, y_train, y_test = create_test_train_split(X_full, y_full, classif_settings[SEED])

    model = SGDClassifier(loss="log_loss", penalty=None, learning_rate="constant", max_iter = 1, eta0=classif_settings[LR], fit_intercept=True, warm_start=True)
    model.coef_ = classif_settings[COEF].values[0,:]
    model.feature_names_in_ = classif_settings[COEF].columns
    model.intercept_ = [0]
    info("calculating global predictions/errors")
    test_acc = model.score(X_test, y_test)
    train_acc = model.score(X_train, y_train)
    full_acc = model.score(X_full, y_full)
    # test_loss = np.mean((model.predict(X_test) - y_test) **2)

    info("fitting model")
    # info(str(X_train))
    # info(str(y_train))
    model.partial_fit(X_full, y_full.values)
    # model.fit(X_full, y_full.values)

    info("model fitted")
    return_params = pd.DataFrame(data = [model.coef_], columns = model.feature_names_in_)


    return {
        LOCAL_COEF: return_params,
        "full_acc" : full_acc,
        "train_acc" : train_acc,
        "test_acc" : test_acc,
        # TEST_LOSS : test_loss,
        LOCAL_TRAIN_SIZE : y_full.shape[0],
    }



def RPC_get_avg(db_client, data_settings):
    data = complete_dataframe(data_settings)

    # info(str(data.tail())
    return {"averages" : data.mean().to_frame().T.astype(float),
            "size" : data.shape[0]
            }


def RPC_get_std(db_client, data_settings: dict, global_mean: pd.DataFrame):
    data = complete_dataframe(data_settings)

    #info(str(data.columns))
    partial_std = ((data.astype(float) - global_mean.astype(float).squeeze())**2).sum(axis = 0)

    return {
        "std" : partial_std.to_frame().T,
        #"std" : tmp3.to_frame().T,
        "size" : data.shape[0]
    }

def RPC_calc_se(db_client, data_settings: dict, classif_settings: dict):
    data = complete_dataframe(data_settings)

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
    data = complete_dataframe(data_settings)

    data = normalise(data.astype(float), data_settings)
    info(f'data columns: {data.columns}')

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

def RPC_hase_mae(db_client, data_settings: dict, classif_settings: dict):
    data = complete_dataframe(data_settings)

    data = normalise(data.astype(float), data_settings)
    X = data.drop(columns = data_settings[TARGET]).values
    y = data[data_settings[TARGET]].values

    model = SGDRegressor(loss="squared_error", penalty=None, max_iter = 1, eta0=0, fit_intercept=False)
    
    model.coef_ = np.copy(classif_settings[COEF])
    model.intercept_ = [0]
    global_pred = model.predict(X)
    mae = np.mean(abs(y - global_pred))

    return {
        "mae" : mae
    }

