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
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
# sys.path.insert(1, os.path.join(sys.path[0], '../V6_implementation'))

from utils import generate_v6_info, generate_data_settings, generate_classif_settings, post_vantage_task, average, get_results
from workflows import normalize_workflow, se_workflow

from v6_LinReg_py.local_constants import *

## vantage6 settings ##
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "password")
client.setup_encryption(None)
ids = [org['id'] for org in client.collaboration.get(1)['organizations']]
# ids = [3]
image_name = "sgarst/association-analysis:1.10"
v6_info = generate_v6_info(client, image_name, ids, 1)

v6_test_info = copy.deepcopy(v6_info)

v6_test_info['image_name'] = "sgarst/association-analysis:test"

## data settings ##
model = "M3" # options : M1, M2, M3, M4, M5, M6, M7
normalize = "global" # options: global, local, none
use_age = False # whether to use age as a covariate
use_dm = True # whether to use dm as a covariate
use_deltas = False # whether to look at delta metabo/brainage
normalize_cat = False # whether to normalize categorical variables
bin_width = 0.2
data_settings = generate_data_settings(model, normalize, use_deltas, normalize_cat, bin_width)


## regression settings ##
n_runs = 1
n_rounds = 5
lr = 0.05
seed_offset = 0

# other settings

write_file = True




def run(v6_info, data_settings, n_runs, n_rounds):

    print(f'pht normalization: {data_settings[NORMALIZE]}')
    local_train_maes = np.zeros((n_runs, n_rounds, len(v6_info[ORG_IDS])))
    local_test_maes = np.zeros_like(local_train_maes)
    for run in range(n_runs):
        seed = run + seed_offset
        classifier_settings = generate_classif_settings(lr, seed, copy.deepcopy(data_settings))
        task_kwargs = {
                "data_settings" : data_settings,
                "classif_settings" : classifier_settings}
        
        # print(f'image name{v6_info[IMAGE_NAME]}')
        avg_fed, std_fed = normalize_workflow(v6_info, copy.deepcopy(data_settings))
        data_settings[GLOBAL_MEAN] = avg_fed
        data_settings[GLOBAL_STD] = std_fed

        for round in range(n_rounds):
            # print(task_kwargs["classif_settings"][COEF])
            round_task = post_vantage_task(v6_info, "train_round", task_kwargs)
            results = get_results(client, round_task, print_log=False)

            local_coefs = pd.concat([result[LOCAL_COEF] for result in results])
            sizes = np.array([result[LOCAL_TRAIN_SIZE] for result in results])

            global_coefs = average(local_coefs, sizes)
            classifier_settings[COEF] = global_coefs
            # local_test_maes[run, round, :]  = [result[TEST_MAE] for result in results]
            local_test_maes[run, round, :]  = [result["full_mae"] for result in results]
            # local_train_maes[run, round, :] = [result[TRAIN_MAE] for result in results]

        se = se_workflow(v6_info, copy.deepcopy(data_settings), copy.deepcopy(classifier_settings))


    branges = [result[BP]['bin_start'].tolist() for result in results]

    final_results = {
        "lr" : lr,
        "nruns" : n_runs,
        "nrounds" : n_rounds,
        "model" : model,
        "global_betas" : global_coefs.values.tolist(), # ideally this is a pd dataframe
        "coef_names" : global_coefs.columns.tolist(),
        "local_betas" : local_coefs.values.tolist(),
        "test_mae" : local_test_maes.tolist(),
        # "train_mae" : local_train_maes.tolist(),
        "bin_ranges" : branges,
        "standard_error" : se.tolist(),
        #"se_columns" : se.columns.tolist(),
        "sizes" : sizes.tolist()       
        }
    
    return final_results, results

if __name__ == "__main__":

    final_results, results = run(v6_info, copy.deepcopy(data_settings), n_runs, n_rounds)
    print(f'coef names: {final_results["coef_names"]}')
    print(f'maes: {final_results["test_mae"]}')

    if write_file:
        date = datetime.datetime.now()
        file_str = f'analysis_model_{model}_centers_{ids}_age_{use_age}_dm_{use_dm}_{date.strftime("%d")}_{date.strftime("%m")}_{date.strftime("%Y")}'
        jo = json.dumps(final_results)

        with open(file_str + ".json", "w", encoding="utf8") as f:
            #json.dump(jo, f)
            f.write(jo)

        for c in range(len(ids)):

            fig1 = results[c][BP]['fig']
            ax1 = results[c][BP]['ax']
            bp1 = results[c][BP]['bp']
    #print(len(bp1['boxes']))
            n_boxplots = len(bp1['boxes'])
            ax1.set_xticks(np.arange(1, len(final_results['bin_ranges']) + 1), [str(brange) for brange in final_results['bin_ranges']])


            with open("resplot_" + file_str + "center_" + str(c) +  ".pickle", "wb") as f:
                pickle.dump(ax1, f)
