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



client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "password")
client.setup_encryption(None)
ids = [org['id'] for org in client.collaboration.get(1)['organizations']]


model = "M1"
use_dm = True
use_age = True
use_deltas = False

all_cols =  ["id", "metabo_age", "brain_age", "date_metabolomics", "date_mri","birth_year", "sex",  "dm", "education_category_3", "bmi"]
#all_cols = [None]
image_name = "sgarst/association-analysis:1.6.1"

data_cols, extra_cols, to_norm_cols = define_model(model, use_dm = use_dm, use_age = use_age)

task = client.post_task(
    input_= {
        "method" : "calc_ABC",
        "kwargs" : {
            #"data_cols" : data_cols,
            #"data_cols" : data_cols_norm,
            #"data_cols" : ['brain_age', "metabo_age"],#, "bmi"],
            "data_cols" : to_norm_cols,
            "extra_cols" : extra_cols,
            "use_deltas" : use_deltas
            # "col_name" :  ['metabo_age', 'brain_age']
        }
    }, 
    name = "calculate A, B and C"
    image=image_name
    organization_ids=ids,
    collaboration_id=1
)

ABC_results = get_results(client, task, print_log=False)

As = np.array([ABC_result['A'] for ABC_result in ABC_results])
Bs = np.array([ABC_result['B'] for ABC_result in ABC_results])

full_A = np.sum(As, axis = 0)
full_B = np.concatenate((Bs), axis = 1)

A_inv = np.linalg.pinv(full_A)

beta_hat_part = np.matmul(A_inv, full_B)
beta_hat = np.sum(beta_hat_part, axis = 1)

data_task = client.post_task(
    input_={
        "method" : "return_data",
        "kwargs" : {
            "data_cols" : to_norm_cols,
            "extra_cols" : extra_cols,
            "use_deltas" : use_deltas
        }
    },
    name = "get data",
    image = image_name,
    organization_ids=ids,
    collaboration_id=1
)

data_results = get_results(client, task, print_log = False)

full_data_conc = np.array([data_result['data'] for data_result in data_results])
full_data = np.concatenate((full_data_conc), axis = 1)

print(full_data.shape)
