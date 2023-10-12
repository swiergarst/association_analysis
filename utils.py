import numpy as np
import time
import pickle
from io import BytesIO

# we might want to try different initializations later on
def init_global_params(data_cols, extra_cols, param_seed = 42):
    data_len = len(data_cols)
    if None in extra_cols:
        extra_len = 0
    elif "Sens_1" in extra_cols or "Sens_2" in extra_cols:
        extra_len = len(extra_cols) - 1
    else:
        extra_len = len(extra_cols)
    
    # we one-hot encode this into three columns
    if "education_category_3" in data_cols:
        data_len += 2
    model_len = data_len + extra_len
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

def define_model(model, use_dm = True, use_age = True):
    if (model == "M1"):
        data_cols = ['brain_age']
        extra_cols = [None]
        to_norm_cols = ['brain_age', 'metabo_age']
    elif (model == "M2"):
        data_cols = ["brain_age", "sex", "dm"]
        extra_cols = ["Lag_time", "Age"]
        to_norm_cols = ['brain_age', 'metabo_age']
    elif (model == "M3"):
        data_cols = ["brain_age", "sex", "dm", "bmi", "education_category_3"]
        extra_cols = ["Age", "Lag_time"]
        to_norm_cols = ['brain_age', 'metabo_age', "bmi"]
    elif (model == "M4"):
        data_cols = ["brain_age", "sex", "dm", "bmi", "education_category_3"]
        extra_cols = ["Age", "Lag_time", "Sens_1"]
        to_norm_cols = ['brain_age', 'metabo_age', "bmi"]
    elif (model == "M5"):
        data_cols = ["brain_age", "Age", "sex", "dm", "bmi", "education_category_3"]
        extra_cols = ["Age", "Lag_time", "Sens_2"]
        to_norm_cols = ['brain_age', 'metabo_age', "bmi"]
    elif (model == "M6"):
        data_cols = ["brain_age", "Age", "sex", "Lag_time", "dm"]
        extra_cols = ["Age", "Lag_time", "mh"]
    elif (model == "M7"):
        data_cols = ["brain_age", "Age", "sex", "Lag_time", "dm", "bmi", "education_category_3"]
        extra_cols = ["Age", "Lag_time", "mh"]
    else:
        return ValueError("invalid model option")

    if (use_dm == False) and (model != "M1") :
        data_cols.remove("dm")
    if (use_age == False) and (model != "M1"):
        extra_cols.remove("Age")
        #to_norm_cols.remove("Age")
    #print(data_cols)
    return data_cols, extra_cols, to_norm_cols


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


def normalize_workflow(client, image_name,  to_norm_cols, extra_cols, use_deltas, normalize, normalize_cat = False):

    ids = [org['id'] for org in client.collaboration.get(1)['organizations']]

    avg_task = client.post_task(
        input_= {
            "method" : "get_avg",
            "kwargs" : {
                #"data_cols" : data_cols,
                #"data_cols" : data_cols_norm,
                #"data_cols" : ['brain_age', "metabo_age"],#, "bmi"],
                "data_cols" : to_norm_cols,
                "extra_cols" : extra_cols,
                "use_deltas" : use_deltas,
                "normalize_cat" : normalize_cat
               # "col_name" :  ['metabo_age', 'brain_age']
            }
        },
        name= "get average",
        image = image_name,
        organization_ids= ids,
        collaboration_id=1
    )
    avg_results = get_results(client, avg_task, print_log=True)
    avg_cols = np.array([avg_result['cols'] for avg_result in avg_results])
    means = np.array([result['mean'] for result in avg_results])
    sizes = np.array([result['size'] for result in avg_results])
    print(means[0].shape)
    global_mean = np.sum([(mean * size) for mean, size in zip(means, sizes)], axis = 0) / np.sum(sizes)

    if normalize == "global":
        std_task = client.post_task(
            input_ = {
                "method" : "get_std",
                "kwargs" : {
                    "global_mean" : global_mean,
                    #"data_cols" : data_cols,
                    "data_cols" : to_norm_cols,
                    #"data_cols" : ['brain_age', "metabo_age"],#, "bmi"],
                    "extra_cols" : extra_cols,
                    "use_deltas" : use_deltas,
                    "normalize_cat" : normalize_cat
                }
            },
            name = "get std",
            image = image_name,
            organization_ids=ids,
            collaboration_id=1
        )
        std_results = get_results(client, std_task, print_log=False)
        stds = np.array([result['std_part'] for result in std_results])
        #print(std_results[0]['cols'])
        global_std = np.sqrt(np.sum(stds, axis = 0)/ np.sum(sizes))

        # we need to put metabo_age at the end of the means/stds, since that is where the other task expects it
        # print(f'global mean before swap: {global_mean}')
        global_mean[[1, -1]] = global_mean[[-1, 1]]
        global_std[[1, -1]] = global_std[[-1, 1]]
        # print(f'global mean after swap: {global_mean}')
    else:
        global_std = None

    return global_mean, global_std, np.sum(sizes), avg_cols