import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../V6_implementation'))

from V6_implementation.v6_LinReg_py.local_constants import *
from V6_implementation.run_constants import *
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

def generate_classif_settings(lr: float, seed: int, data_settings: dict):
    coef_names = data_settings[MODEL_COLS]
    if data_settings[CLASSIF_TARGETS] == None:
        coef_names.remove(data_settings[TARGET])
    else:
        # print(f'classif targets: {data_settings[CLASSIF_TARGETS]}, coef names: {coef_names}')
        for col in data_settings[CLASSIF_TARGETS]:
            coef_names.remove(col)
    classif_settings = {}
    classif_settings[LR] = lr
    classif_settings[SEED] = seed

    # generate initial model params
    s = np.random.default_rng(seed=seed) # we use the same seed as for the test/train split
    model_len = len(coef_names)
    coefs = s.normal(0, 1, (1,model_len))

    coefs_df = pd.DataFrame(data=coefs, columns = coef_names)
    classif_settings[COEF] = coefs_df
    return classif_settings

def generate_data_settings(model, normalize, use_deltas, normalize_cat, bin_width = 0.2):
    data_settings = {}
    data_settings[NORMALIZE] = normalize
    data_settings[USE_DELTAS] = use_deltas
    data_settings[NORM_CAT] = normalize_cat
    data_settings[BIN_WIDTH_BOXPLOT] = bin_width
    data_settings[SENS] = 0
    data_settings[STRATIFY] = False
    data_settings[DEFINES] = { # this way we can define the column names locally
        ALL_COLS: ALL_EXISTING_COLS_VALUES,
        OPTION_COLS: OPTION_COLS_VALUES,
        CAT_COLS: CAT_COLS_VALUES,
        LAG_TIME_COL: LAG_TIME,
        AGE_COL : AGE,
        DATE_METABOLOMICS_COL: DATE_METABOLOMICS,
        DATE_MRI_COL : DATE_MRI,
        BIRTH_YEAR_COL : BIRTH_YEAR,
        METABO_AGE_COL : METABO_AGE,
        BRAIN_AGE_COL : BRAIN_AGE,
        EDUCATION_CATEGORY_COL : EDUCATION_CATEGORY,
        EDUCATION_CATEGORIES_LIST : [EC1, EC3],
        ID_COL : ID
    }
    # data_settings[STRATIFY_GROUPS] = STRATIFY_GROUPS_VALUES
    data_settings[ALL_COLS] = ALL_EXISTING_COLS_VALUES
    data_settings[OPTION_COLS] = OPTION_COLS_VALUES
    data_settings[CAT_COLS] = CAT_COLS_VALUES
    data_settings[BP_1] = BRAIN_AGE
    data_settings[CLASSIF_TARGETS] = None
    data_settings[TARGET] = METABO_AGE

    if model == "M1":
        data_settings[MODEL_COLS] = [BRAIN_AGE, METABO_AGE]
    elif model == "M1.5":
        data_settings[MODEL_COLS] = [BRAIN_AGE, METABO_AGE, AGE]
    elif model == "M2":
        data_settings[MODEL_COLS] = [BRAIN_AGE, METABO_AGE, AGE, LAG_TIME, SEX, DM]
    elif model == "M2.5":
        data_settings[MODEL_COLS] = [BRAIN_AGE, METABO_AGE, LAG_TIME, SEX, DM]
    elif model == "M3":
        data_settings[MODEL_COLS] = [BRAIN_AGE, METABO_AGE ,SEX, DM, BMI, LAG_TIME, AGE, EC1, EC3]
    elif model == "M3.5":
        data_settings[MODEL_COLS] = [BRAIN_AGE, METABO_AGE, DM, BMI, LAG_TIME, AGE, EC1, EC3]
    elif model == "M4":
        data_settings[MODEL_COLS] = [BRAIN_AGE, METABO_AGE, SEX, DM, BMI, LAG_TIME, AGE, EC1, EC3]
        data_settings[SENS] = 1
    elif model == "M5":
        data_settings[MODEL_COLS] = [BRAIN_AGE, METABO_AGE, SEX, DM, BMI, LAG_TIME, AGE, EC1, EC3]
        data_settings[SENS] = 2
    elif model == "M6":
        data_settings[TARGET] = METABO_HEALTH
        data_settings[MODEL_COLS] = [BRAIN_AGE, METABO_HEALTH, SEX, DM, BMI, LAG_TIME, AGE, EC1, EC3]
    elif model == "M7":
        data_settings[TARGET] = METABO_HEALTH
        data_settings[MODEL_COLS] = [BRAIN_AGE, METABO_HEALTH, SEX, DM, BMI, LAG_TIME, AGE, EC1, EC3]

    # # this shouldn't be necessary once we change the system to be more grid-search-friendly
    # if use_age == False:
    #     if AGE in data_settings[MODEL_COLS]:
    #         data_settings[MODEL_COLS].remove(AGE)
    # if use_dm == False:
    #     if DM in data_settings[MODEL_COLS]:
    #         data_settings[MODEL_COLS].remove(DM)



    data_settings[DATA_COLS] = infer_data_cols(data_settings)
    return data_settings

def append_data_settings(data_settings, label_cols):
    if DEMENTIA in label_cols:
        data_settings[MODEL_COLS].append(DEMENTIA)
        data_settings[DATA_COLS] = infer_data_cols(data_settings)
    return data_settings

# see which data columns we need to get based on which model columns we have
def infer_data_cols(data_settings: dict):
    #easiest way to get a start
    data_cols = data_settings[MODEL_COLS].copy()
    if (AGE in data_settings[MODEL_COLS]) or (LAG_TIME in data_settings[MODEL_COLS]) or (data_settings[USE_DELTAS] == True):
        data_cols.append(DATE_MRI)
        data_cols.append(DATE_METABOLOMICS)
        data_cols.append(BIRTH_YEAR)
        if AGE in data_cols:
            data_cols.remove(AGE)
        if LAG_TIME in data_cols:
            data_cols.remove(LAG_TIME)
    
    if EC1 in data_settings[MODEL_COLS]:
        data_cols.append(EDUCATION_CATEGORY)
        data_cols.remove(EC1)
        # data_cols.remove(EC2)
        data_cols.remove(EC3)
    return data_cols



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
    

    if '' in [res['result'] for res in result]:
        print("Error from client: ")
        for res in result:
            print(res['log'])     
        raise(RuntimeError(f"error from client: see log above."))        

    if print_log:
    
        for res in result:
            print(res['log'])
            
    results = [pickle.loads(BytesIO(res['result']).getvalue()) for res in result]
    return results

def post_vantage_task(v6_info: dict, method_name: str, kwargs: dict):
    task = v6_info[CLIENT].post_task(
        input_ = {
            "method" : method_name,
            "kwargs" : kwargs
        },
        name = "get average or std",
        image = v6_info[IMAGE_NAME],
        organization_ids = v6_info[ORG_IDS],
        collaboration_id = v6_info[COLLAB_ID]
    )
    return task

# simple fedAvg implementation
def average(params, sizes):
    
    #print(params.shape)
    #create size-based weights
    num_clients = sizes.size

    total_size = np.sum(sizes) 
    weights = sizes / total_size


    #do averaging
    for i in range(num_clients):
        params.iloc[i] = params.iloc[i] * weights[i]


    global_parameters = params.sum().to_frame().T
    #print(weights.shape, params.shape, parameters.shape)
    return global_parameters


def hase_workflow():
    pass