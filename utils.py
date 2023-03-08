import numpy as np


# we might want to try different initializations later on
def init_global_model_params(cols):
    return np.random.norm(0, 1, len(cols)), np.random.norm(0, 1)

def average(params, sizes):
    pass

#TODO: check column names
#TODO: models 4-7 (sensitivity analysis/metaboHealth)
def define_model(model):
    if (model == "M1"):
        cols = ['MetaboAge']
    elif (model == "M2"):
        cols = ["MetaboAge", "age", "sex", "lag-time", "diabetes"]
    elif (model == "M3"):
        cols = ["MetaboAge", "age", "sex", "lag-time", "diabetes", "BMI", "education_years"]
        pass
    else:
        return ValueError("invalid model option")

