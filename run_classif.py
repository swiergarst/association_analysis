from vantage6.client import Client
import numpy as np
import pandas as pd
import sys
import os
import datetime
import json
import copy
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../V6_implementation'))

from V6_implementation.utils2 import generate_v6_info, generate_data_settings, generate_classif_settings, post_vantage_task, average, get_results, append_data_settings
from V6_implementation.run_constants import *
from V6_implementation.v6_LinReg_py.local_constants import *
from V6_implementation.run import run
from V6_implementation.workflows import normalize_workflow

## vantage6 settings ##
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "password")
client.setup_encryption(None)
ids = [org['id'] for org in client.collaboration.get(1)['organizations']]
# ids = [3]
image_name = "sgarst/association-analysis:1.9"
v6_info = generate_v6_info(client, image_name, ids, 1)

## data settings ##
model = "M3" # options : M1, M2, M3, M4, M5, M6, M7
normalize = "global" # options: global, local, none
use_age = False # whether to use age as a covariate
use_dm = True # whether to use dm as a covariate
use_deltas = False # whether to look at delta metabo/brainage
normalize_cat = False # whether to normalize categorical variables
bin_width = 0.2
data_settings = generate_data_settings(model, normalize, use_deltas, normalize_cat, bin_width)




## classifier settings ##
n_runs = 1
n_rounds = 2
lr = 0.005
seed_offset = 0
label_cols = [DM, DEMENTIA]
data_settings[CLASSIF_TARGETS] = label_cols 

def run_classif(v6_info, data_settings, n_runs, n_rounds, classif_targets):
    local_test_accuracies = np.zeros((n_runs, n_rounds, len(v6_info[ORG_IDS])))
    data_settings[CLASSIF_TARGETS] = classif_targets
    data_settings = append_data_settings(data_settings, classif_targets)
    for run in range(n_runs):
        seed = run + seed_offset
        classifier_settings = generate_classif_settings(lr, seed, copy.deepcopy(data_settings))
        task_kwargs = {
                "data_settings" : data_settings,
                "classif_settings" : classifier_settings}
        avg_fed, std_fed = normalize_workflow(v6_info, copy.deepcopy(data_settings))
        data_settings[GLOBAL_MEAN] = avg_fed
        data_settings[GLOBAL_STD] = std_fed

        for round in range(n_rounds):
            # print(task_kwargs["classif_settings"][COEF])
            round_task = post_vantage_task(v6_info, "predict_disease", task_kwargs)
            results = get_results(client, round_task)

            local_coefs = pd.concat([result[LOCAL_COEF] for result in results])
            sizes = np.array([result[LOCAL_TRAIN_SIZE] for result in results])

            global_coefs = average(local_coefs, sizes)
            classifier_settings[COEF] = global_coefs
            # local_test_maes[run, round, :]  = [result[TEST_MAE] for result in results]
            local_test_accuracies[run, round, :]  = [result["test_acc"] for result in results]
            # local_train_maes[run, round, :] = [result[TRAIN_MAE] for result in results]

            final_results = {
                    "lr" : classifier_settings[LR],
                    "nruns" : n_runs,
                    "nrounds" : n_rounds,
                    "model" : model,
                    "global_betas" : global_coefs.values.tolist(), # ideally this is a pd dataframe
                    "coef_names" : global_coefs.columns.tolist(),
                    "local_betas" : local_coefs.values.tolist(),
                    "test_acc" : local_test_accuracies.tolist(),
                    #"se_columns" : se.columns.tolist(),
                    "sizes" : sizes.tolist()       
                }
    return final_results
