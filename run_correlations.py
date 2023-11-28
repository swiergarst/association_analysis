from vantage6.client import Client
import os
import numpy as np
import pandas as pd
import datetime
import json
from NCDC.V6_implementation.v6_LinReg_py.local_constants import *
from utils2 import generate_v6_info, generate_data_settings
from run_hase import run_hase
import copy
from workflows import normalize_workflow
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
data_settings = generate_data_settings("M3", normalize = "global", use_age = True, use_dm = True, use_deltas = True, normalize_cat=True)
avg_fed, std_fed = normalize_workflow(v6_info, copy.deepcopy(data_settings))
data_settings[GLOBAL_MEAN] = avg_fed
data_settings[GLOBAL_STD] = std_fed



total_results = {}
# full_table = pd.DataFrame( columns = all_considered_columns)

while len(all_considered_columns) > 0:
    target_column = all_considered_columns[0]
    total_results[target_column] = {}
    data_settings[TARGET] = target_column
    all_considered_columns.remove(target_column)
    # print(all_considered_columns)
    for other_column in all_considered_columns:
        print(target_column, other_column)
        data_settings[MODEL_COLS] = [target_column, other_column] 
        hase_out = run_hase(v6_info, data_settings)
        total_results[target_column][other_column] = hase_out["global_betas"]
    



    if write_file:
        date = datetime.datetime.now()
        file_str = f'correlations_{date.strftime("%d")}_{date.strftime("%m")}_{date.strftime("%Y")}'
        jo = json.dumps(total_results)

        with open(file_str + ".json", "w", encoding="utf8") as f:
            #json.dump(jo, f)
            f.write(jo)