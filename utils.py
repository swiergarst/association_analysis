import numpy as np


# we might want to try different initializations later on
def init_global_params(cols):
    return np.random.normal(0, 1, len(cols)), [np.random.normal(0, 1)]


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
        cols = ['metabo_age']
    elif (model == "M2"):
        cols = ["metabo_age", "Age", "sex", "Lag_time", "Diabetes_Mellitus"]
    elif (model == "M3"):
        cols = ["metabo_age", "Age", "sex", "Lag_time", "Diabetes_Mellitus", "BMI", "education_category"]
    elif (model == "M4"):
        cols = ["metabo_age", "Age", "sex", "Lag_time", "Diabetes_Mellitus", "BMI", "education_category", "Sens_1"]
    elif (model == "M5"):
        cols = ["metabo_age", "Age", "sex", "Lag_time", "Diabetes_Mellitus", "BMI", "education_category", "Sens_2"]
    elif (model == "M6"):
        cols = ["metabo_health", "Age", "sex", "Lag_time", "Diabetes_Mellitus"]
    elif (model == "M7"):
        cols = ["metabo_age", "Age", "sex", "Lag_time", "Diabetes_Mellitus", "BMI", "education_category"]
    else:
        return ValueError("invalid model option")
    return cols

