import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from V6_implementation.v6_LinReg_py.constants import *

import time
import pickle
from io import BytesIO
import pandas as pd
import numpy as np

# maybe put some asserts in this?
def generate_v6_info(client, image_name, ids, collab_id):
    v6_info = {}
    v6_info[CLIENT] = client
    v6_info[IMAGE_NAME] = image_name
    v6_info[ORG_IDS] = ids
    v6_info[COLLAB_ID] = collab_id
    return v6_info

def generate_data_settings(model, normalize, use_age, use_dm, use_deltas, normalize_cat):
    data_settings = {}
    data_settings[NORMALIZE] = normalize
    data_settings[USE_AGE] = use_age
    data_settings[USE_DM] = use_dm
    data_settings[USE_DELTAS] = use_deltas
    data_settings[NORMALIZE_CAT] = normalize_cat
    
    if (model == "M6") or  (model == "M7"):
        data_settings[TARGET] = METABO_HEALTH
    else:
        data_settings[TARGET] = METABO_AGE

    if model == "M1":
        data_settings[DATA_COLS] = [BRAIN_AGE]
    elif model == "M2":
        data_settings[DATA_COLS] = [BRAIN_AGE, SEX, DM]
    elif model == "M3":
        data_settings[DATA_COLS] = [BRAIN_AGE, SEX, DM, BMI, EDUCATION_CATEGORY, LAG_TIME, AGE]
    elif model == "M4":
        data_settings[DATA_COLS] = [BRAIN_AGE, SEX, DM, BMI, EDUCATION_CATEGORY, LAG_TIME, AGE, SENSITIVITY_1]
    elif model == "M5":
        data_settings[DATA_COLS] = [BRAIN_AGE, SEX, DM, BMI, EDUCATION_CATEGORY, LAG_TIME, AGE, SENSITIVITY_2]
    elif model == "M6":
        data_settings[DATA_COLS] = [BRAIN_AGE, SEX, DM, BMI, EDUCATION_CATEGORY, LAG_TIME, AGE]
    elif model == "M7":
        data_settings[DATA_COLS] = [BRAIN_AGE, SEX, DM, BMI, EDUCATION_CATEGORY, LAG_TIME, AGE]
    
    return split_direct_synth(data_settings)

def normalize_workflow(v6_info, data_settings):
    # this makes the client-side code a little more flexible
    data_settings[DIRECT_COLS].append(data_settings[TARGET])

    avg_task = v6_info[CLIENT].post_task(
        input_ = {
            "method" : "get_avg",
            "kwargs" : {
                "data_settings": data_settings
                }
        },
        name = "get average",
        image = v6_info[IMAGE_NAME],
        organization_ids = v6_info[ORG_IDS],
        collaboration_id = v6_info[COLLAB_ID]
    )
    avg_results = get_results(v6_info[CLIENT], avg_task)

    # means = avg_results[0]['averages']
    # for i in range(1, len(avg_results)):
    #     means.append(avg_results[i]['averages'])

    means = pd.concat([avg_result['averages'] for avg_result in avg_results], ignore_index=True)
    sizes = np.array([avg_result['size'] for avg_result in avg_results])
    
    global_mean_arr = np.array([np.sum([(mean * size) for mean, size in zip(means.values, sizes)], axis = 0) / np.sum(sizes)])
    global_mean_df = pd.DataFrame(data=global_mean_arr, columns = means.columns)

    #print(means.head())
    #print(global_mean_df.head())
    return(global_mean_df)

 #determine which columns to normalize based on the data columns and normalization settings 
def det_norm_cols(data_settings):
    norm_cat = data_settings[NORM_CAT]
    data_cols = data_settings[DATA_COLS]
    if ~norm_cat:
        norm_cols = [col for col in data_cols if col not in CAT_COLS_VALUES]
    else:
        norm_cols = data_cols
    norm_cols.append(data_settings[TARGET])
    return norm_cols

# split based on whether they are already in the db or not
def split_direct_synth(data_settings):
    data_settings[DIRECT_COLS] = [col for col in data_settings[DATA_COLS] if col not in EXTRA_COLS_VALUES]
    data_settings[SYNTH_COLS] = [col for col in data_settings[DATA_COLS] if col in EXTRA_COLS_VALUES]
    return data_settings




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

