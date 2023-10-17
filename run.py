from vantage6.client import Client
import os
import numpy as np
import json
from io import BytesIO
import datetime
import psycopg2
from utils import init_global_params, average, define_model, get_results
import time
import matplotlib.pyplot as plt
dir_path = os.path.dirname(os.path.realpath(__file__))
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import SGDRegressor
from utils2 import generate_v6_info, generate_data_settings, generate_classif_settings, post_vantage_task, average
from workflows import normalize_workflow

from v6_LinReg_py.constants import *
import sys
import os
import copy

sys.path.insert(1, os.path.join(sys.path[0], '..'))

## vantage6 settings ##
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "password")
client.setup_encryption(None)
ids = [org['id'] for org in client.collaboration.get(1)['organizations']]

image_name = "sgarst/association-analysis:1.7.0"
v6_info = generate_v6_info(client, image_name, ids, 1)

v6_test_info = copy.deepcopy(v6_info)

v6_test_info['image_name'] = "sgarst/association-analysis:test"

## data settings ##
model = "M3" # options : M1, M2, M3, M4, M5, M6, M7
normalize = "global" # options: global, local, none
use_age = True # whether to use age as a covariate
use_dm = True # whether to use dm as a covariate
use_deltas = False # whether to look at delta metabo/brainage
normalize_cat = False # whether to normalize categorical variables
bin_width = 0.2
data_settings = generate_data_settings(model, normalize, use_age, use_dm, use_deltas, normalize_cat, bin_width)


## regression settings ##
n_runs = 1
n_rounds = 2
lr = 0.005
seed_offset = 0

# other settings
write_file = True


local_train_maes = np.zeros((n_runs, n_rounds, len(v6_info[ORG_IDS])))
local_test_maes = np.zeros_like(local_train_maes)
for run in range(n_runs):
    seed = run + seed_offset
    classifier_settings = generate_classif_settings(lr, seed_offset)
    task_kwargs = {
            "data_settings" : data_settings,
            "classif_settings" : classifier_settings}
    
    avg_fed, std_fed = normalize_workflow(v6_info, copy.deepcopy(data_settings))
    data_settings[GLOBAL_MEAN] = avg_fed
    data_settings[GLOBAL_STD] = std_fed
    for round in range(n_rounds):
        print(task_kwargs["classif_settings"][COEF])
        round_task = post_vantage_task(v6_info, "RPC_train_round", task_kwargs)
        results = get_results(client, round_task)

        local_coefs = pd.concat([result[LOCAL_COEF] for result in results])
        sizes = [result[LOCAL_TRAIN_SIZE] for result in results]

        global_coefs = average(local_coefs, sizes)
        local_test_maes[run, round, :]  = [result[TEST_MAE] for result in results]
        local_train_maes[run, round, :] = [result[TRAIN_MAE] for result in results]


branges = [result[BP]['bin_start'].tolist() for result in results]

final_results = {
    "lr" : lr,
    "nruns" : n_runs,
    "nrounds" : n_rounds,
    "model" : model,
    "global_betas" : global_coefs, # ideally this is a pd dataframe
    "local_betas" : local_coefs,
    "test_mae" : local_test_maes.tolist(),
    "train_mae" : local_train_maes.tolist(),
    "bin_ranges" : branges,
    "standard_error" : se,
    "sizes" : sizes        
    }


if write_file:
    date = datetime.datetime.now()
    file_str = f'analysis_model_{model}_centers_{ids}_age_{use_age}_dm_{use_dm}_{date.strftime("%d")}_{date.strftime("%m")}_{date.strftime("%Y")}'
    jo = json.dumps(final_results)

    with open(file_str + ".json", "w", encoding="utf8") as f:
        #json.dump(jo, f)
        f.write(jo)

    for c in range(len(ids)):

        fig1 = results[c]['resplot']['fig']
        ax1 = results[c]['resplot']['ax']
        bp1 = results[c]['resplot']['bp']
#print(len(bp1['boxes']))
        n_boxplots = len(bp1['boxes'])
        ax1.set_xticks(np.arange(1, len(branges) + 1), [str(brange) for brange in branges])


        with open("resplot_" + file_str + "center_" + str(c) +  ".pickle", "wb") as f:
            pickle.dump(ax1, f)
