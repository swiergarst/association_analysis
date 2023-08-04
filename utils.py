import numpy as np
import time
import pickle
from io import BytesIO

# we might want to try different initializations later on
def init_global_params(data_cols, extra_cols, param_seed = 42):
    if None in extra_cols:
        extra_len = 0
    elif "Sens_1" in extra_cols or "Sens_2" in extra_cols:
        extra_len = len(extra_cols) - 1
    else:
        extra_len = len(extra_cols)
    model_len = len(data_cols) + extra_len
    s = np.random.default_rng(seed=param_seed)
    return s.normal(0, 1, model_len), [s.normal(0,1)]
    #return np.random.normal(0, 1, model_len), [np.random.normal(0, 1)]


# simple fedAvg implementation
def average(params, sizes):
    
    #print(params.shape)
    #create size-based weights
    num_clients = sizes.size

    total_size = np.sum(sizes) 
    weights = sizes / total_size


    #do averaging
    parameters = np.zeros(params.shape[0], dtype=float)
    #print(weights.shape, params.shape, parameters.shape)
    for j in range(num_clients):
        parameters += weights[j] * params[:,j]

    return parameters

def define_model(model, use_dm = True):
    if (model == "M1"):
        data_cols = ['brain_age']
        extra_cols = [None]
    elif (model == "M2"):
        data_cols = ["sex", "brain_age", "dm"]
        extra_cols = ["Lag_time", "Age"]
    elif (model == "M3"):
        data_cols = ["brain_age", "sex", "dm", "bmi", "education_category"]
        extra_cols = ["Age", "Lag_time"]
    elif (model == "M4"):
        data_cols = ["brain_age", "Age", "sex", "dm", "bmi", "education_category"]
        extra_cols = ["Age", "Lag_time", "Sens_1"]
    elif (model == "M5"):
        data_cols = ["brain_age", "Age", "sex", "dm", "bmi", "education_category"]
        extra_cols = ["Age", "Lag_time", "Sens_2"]
    elif (model == "M6"):
        data_cols = ["brain_age", "Age", "sex", "Lag_time", "dm"]
        extra_cols = ["Age", "Lag_time", "mh"]
    elif (model == "M7"):
        data_cols = ["brain_age", "Age", "sex", "Lag_time", "dm", "bmi", "education_category"]
        extra_cols = ["Age", "Lag_time", "mh"]
    else:
        return ValueError("invalid model option")

    if (use_dm == False) and (model != "M1") :
        data_cols.remove("dm")
    #print(data_cols)
    return data_cols, extra_cols


def get_results(client, task, max_attempts = 20, print_log = False):
    finished = False
    attempts = 0
    while (finished == False):
        attempts += 1
        result = client.get_results(task_id=task.get("id"))
        time.sleep(10)
        if not None in [res['result'] for res in result]:
            finished = True
        if attempts > max_attempts:
            print("max attempts exceeded")
            print(result)
            exit()

    if print_log:
        for res in result:
            print(res['log'])
            
    results = [pickle.loads(BytesIO(res['result']).getvalue()) for res in result]
    return results
