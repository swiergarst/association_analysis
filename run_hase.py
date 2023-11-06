from vantage6.client import Client
import os
import numpy as np
import json
import datetime
import matplotlib.pyplot as plt
dir_path = os.path.dirname(os.path.realpath(__file__))
import pickle
from tqdm import tqdm
import pandas as pd
import sys
import os
import copy
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../V6_implementation'))

from V6_implementation.utils2 import generate_v6_info, generate_data_settings, generate_classif_settings, post_vantage_task, average, get_results
from V6_implementation.workflows import normalize_workflow, se_workflow

from V6_implementation.v6_LinReg_py.constants import *

## vantage6 settings ##
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "password")
client.setup_encryption(None)
ids = [org['id'] for org in client.collaboration.get(1)['organizations']]
# ids = [3]
image_name = "sgarst/association-analysis:1.8"
v6_info = generate_v6_info(client, image_name, ids, 1)

## data settings ##
model = "M3" # options : M1, M2, M3, M4, M5, M6, M7
normalize = "global" # options: global, local, none
use_age = False # whether to use age as a covariate
use_dm = True # whether to use dm as a covariate
use_deltas = False # whether to look at delta metabo/brainage
normalize_cat = True # whether to normalize categorical variables
data_settings = generate_data_settings(model, normalize, use_age, use_dm, use_deltas, normalize_cat)

# other settings

write_file = True

def run_hase(v6_info, data_settings):

    # print(f'data cols: {data_settings[DATA_COLS]}')
    task_kwargs = {"data_settings" : data_settings}
    avg_fed, std_fed = normalize_workflow(v6_info, copy.deepcopy(data_settings))
    # print("normalizing complete")
    data_settings[GLOBAL_MEAN] = avg_fed
    data_settings[GLOBAL_STD] = std_fed


    # print(task_kwargs["classif_settings"][COEF])
    abc_task = post_vantage_task(v6_info, "calc_ABC", task_kwargs)
    ABC_results = get_results(client, abc_task)

    As = np.array([ABC_result['A'] for ABC_result in ABC_results])
    Bs = np.array([ABC_result['B'] for ABC_result in ABC_results])
    Cs = np.array([ABC_result['C'] for ABC_result in ABC_results])
    sizes = np.array([ABC_result[SIZE] for ABC_result in ABC_results])

    global_size = np.sum(sizes)


    full_A = np.sum(As, axis = 0)
    full_B = np.sum(Bs, axis = 0)
    full_C = np.sum(Cs, axis = 0)

    A_inv = np.linalg.pinv(full_A)
    A_inv_diag = np.diag(A_inv)
    beta_hat = np.matmul(A_inv, full_B)

    se_part = np.matmul(full_B.T,  A_inv)
    se_part2 = np.matmul(se_part, full_B)
    se_part3 = full_C - se_part2

    bot = full_C - full_B.T @ A_inv @ full_B
    df = global_size - full_B.shape[0]

    # print(f'shapes: {A_inv.shape, As.shape, bot.shape}')
    se = 1/np.sqrt(df / (A_inv_diag * bot))

    classif_settings = generate_classif_settings(0, 0, copy.deepcopy(data_settings))
    classif_settings[COEF] = np.copy(beta_hat)
    mae_kwargs = { 
        "data_settings" : data_settings,
        "classif_settings" : classif_settings}
    
    mae_task = post_vantage_task(v6_info, "hase_mae", mae_kwargs)
    mae_results = get_results(client, mae_task)
    mae = np.array([mae_result['mae'] for mae_result in mae_results])

    final_results = {
        "global_betas" : beta_hat.tolist(), # ideally this is a pd dataframe
        "coef_names" : data_settings[MODEL_COLS],
        "mae" : mae,
        "standard_error" : se.tolist(),
        #"se_columns" : se.columns.tolist(),
        "sizes" : sizes.tolist()       
        }
    
    return final_results

if __name__ == "__main__":

    final_results  = run_hase(v6_info, copy.deepcopy(data_settings))

    if write_file:
        date = datetime.datetime.now()
        file_str = f'hase_model_{model}_centers_{ids}_age_{use_age}_dm_{use_dm}_{date.strftime("%d")}_{date.strftime("%m")}_{date.strftime("%Y")}'
        jo = json.dumps(final_results)

        with open(file_str + ".json", "w", encoding="utf8") as f:
            #json.dump(jo, f)
            f.write(jo)
