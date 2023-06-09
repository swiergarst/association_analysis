import numpy as np


# we might want to try different initializations later on
def init_global_params(data_cols, extra_cols):
    if None in extra_cols:
        extra_len = 0
    elif "Sens_1" in extra_cols or "Sens_2" in extra_cols:
        extra_len = len(extra_cols) - 1
    else:
        extra_len = len(extra_cols)
    model_len = len(data_cols) + extra_len
    return np.random.normal(0, 1, model_len), [np.random.normal(0, 1)]


# simple fedAvg implementation
def average(params, sizes):
    
    #print(params.shape)
    #create size-based weights
    num_clients = sizes.size

    total_size = np.sum(sizes) 
    weights = sizes / total_size


    #do averaging
    parameters = np.zeros(params.shape[0], dtype=float)
    print(weights.shape, params.shape, parameters.shape)
    for j in range(num_clients):
        parameters += weights[j] * params[:,j]

    return parameters

def define_model(model):
    if (model == "M1"):
        data_cols = ['metabo_age']
        extra_cols = [None]
    elif (model == "M2"):
        data_cols = ["metabo_age", "sex", "dm"]
        extra_cols = ["Age", "Lag_time"]
    elif (model == "M3"):
        data_cols = ["metabo_age", "sex", "dm", "bmi", "education_category"]
        extra_cols = ["Age", "Lag_time"]
    elif (model == "M4"):
        data_cols = ["metabo_age", "Age", "sex", "Lag_time", "dm", "bmi", "education_category"]
        extra_cols = ["Age", "Lag_time", "Sens_1"]
    elif (model == "M5"):
        data_cols = ["metabo_age", "Age", "sex", "Lag_time", "dm", "bmi", "education_category"]
        extra_cols = ["Age", "Lag_time", "Sens_2"]
    elif (model == "M6"):
        data_cols = ["metabo_health", "Age", "sex", "Lag_time", "dm"]
        extra_cols = ["Age", "Lag_time"]
    elif (model == "M7"):
        data_cols = ["metabo_health", "Age", "sex", "Lag_time", "dm", "bmi", "education_category"]
        extra_cols = ["Age", "Lag_time"]
    else:
        return ValueError("invalid model option")
    return data_cols, extra_cols

