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
from utils2 import normalize_workflow, generate_v6_info, generate_data_settings
from v6_LinReg_py.constants import *
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from V6_test_module.run_test import average_ground_truth

## vantage6 settings ##
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "password")
client.setup_encryption(None)
ids = [org['id'] for org in client.collaboration.get(1)['organizations']]

image_name = "sgarst/association-analysis:1.7.0"
v6_info = generate_v6_info(client, image_name, ids, 1)

v6_test_info = v6_info.copy()

v6_test_info['image_name'] = "sgarst/association-analysis:test"
## data settings ##
model = "M1" # options : M1, M2, M3, M4, M5, M6, M7
normalize = "global" # options: global, local, none
use_age = True # whether to use age as a covariate
use_dm = True # whether to use dm as a covariate
use_deltas = False # whether to look at delta metabo/brainage
normalize_cat = False # whether to normalize categorical variables
data_settings = generate_data_settings(model, normalize, use_age, use_dm, use_deltas, normalize_cat)


## regression settings ##
n_runs = 1
n_rounds = 2
lr = 0.005

#print(normalize_workflow(v6_info, data_settings))
print(average_ground_truth(v6_test_info, data_settings))