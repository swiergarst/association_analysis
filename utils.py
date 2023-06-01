import numpy as np


# we might want to try different initializations later on
def init_global_params(data_cols):
    return np.random.normal(0, 1, len(data_cols)), [np.random.normal(0, 1)]


# simple fedAvg implementation
def average(params, sizes):
    
    #print(params.shape)
    #create size-based weights
    num_clients = sizes.size



    total_size = np.sum(sizes) 
    weights = sizes / total_size

    #do averaging
    parameters = np.zeros_like(params[0].shape[0])

    for j in range(num_clients):
        parameters += weights[j] * params[:,j]

    return parameters

#TODO: models 4-7 (sensitivity analysis/metaboHealth)
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
        data_cols = ["metabo_age", "Age", "sex", "Lag_time", "dm", "bmi", "education_category", "Sens_2"]
        extra_cols = ["Age", "Lag_time", "Sens_2"]
    elif (model == "M6"):
        data_cols = ["metabo_health", "Age", "sex", "Lag_time", "dm"]
        extra_cols = ["Age", "Lag_time"]
    elif (model == "M7"):
        data_cols = ["metabo_health", "Age", "sex", "Lag_time", "dm", "bmi", "education_category"]
        extra_cols = ["Age", "Lag_time"]
    else:
        return ValueError("invalid model option")
    return data_cols

