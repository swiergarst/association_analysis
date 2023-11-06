from vantage6.client import Client
import os
import numpy as np
import pandas as pd
import datetime
import json
from v6_LinReg_py.constants import *
from utils2 import generate_v6_info, generate_data_settings, infer_data_cols, post_vantage_task, get_results
from workflows import mean_workflow, std_workflow
from run_hase import run_hase


## vantage6 settings ##
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "password")
client.setup_encryption(None)
ids = [org['id'] for org in client.collaboration.get(1)['organizations']]
# ids = [3]
image_name = "sgarst/association-analysis:1.8"
v6_info = generate_v6_info(client, image_name, ids, 1)

all_considered_columns = [DM, METABO_AGE, BRAIN_AGE, AGE, LAG_TIME, SEX, EDUCATION_CATEGORY, BMI]
write_file = False


# generate initial data settings (most of these will be overwritten/not used)
data_settings = generate_data_settings("M3", normalize = "none", use_age = True, use_dm = True, use_deltas = True, normalize_cat=True)
data_settings[MODEL_COLS] = all_considered_columns
data_settings[DATA_COLS] = infer_data_cols(data_settings)


global_mean, local_means = mean_workflow(v6_info, data_settings)
global_std = std_workflow(v6_info, data_settings, global_mean)

# we need one manual task for getting the local stds
res_arr = []
# sizes = []
for i, id in enumerate(ids):
    v6_info[ID] = id
    local_mean = local_means.iloc[i]
    std_task = post_vantage_task(v6_info, "get_std", {"data_settings" : data_settings, "global_mean" : local_mean})
    std_res = get_results(client, std_task)
    # sizes.append(std_res['size'])
    res_arr.append(std_res[0]['std']/std_res[0]['size'])

res_arr.append(global_std)
all_means = pd.concat((local_means, global_mean))
all_stds = pd.concat(res_arr)

all_means.to_csv("averages.csv", index=False)
all_stds.to_csv("stds.csv", index = False)