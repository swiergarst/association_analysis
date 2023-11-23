from V6_implementation.v6_LinReg_py.constants import *
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from V6_implementation.utils2 import get_results, post_vantage_task
import copy
def mean_workflow(v6_info, data_settings):
    # this makes the client-side code a little more flexible

    task = post_vantage_task(v6_info, "get_avg", {"data_settings": data_settings})
    results = get_results(v6_info[CLIENT], task, print_log=False)

    for i, id in enumerate(v6_info[ORG_IDS]):
        results[i]['averages']['ID'] = id
    # means = avg_results[0]['averages']
    # for i in range(1, len(avg_results)):
    #     means.append(avg_results[i]['averages'])

    full_df = pd.concat([avg_result['averages'] for avg_result in results], ignore_index=True)
    sizes = np.array([avg_result['size'] for avg_result in results])
    
    global_mean_arr = np.array([np.sum([(mean * size) for mean, size in zip(full_df.drop(columns = 'ID').values, sizes)], axis = 0) / np.sum(sizes)])
    global_df = pd.DataFrame(data=global_mean_arr, columns = full_df.drop(columns = "ID").columns)

    return(global_df, full_df)


def std_workflow(v6_info, data_settings, global_mean):

    task = post_vantage_task(v6_info, "get_std", {"data_settings": data_settings, "global_mean" : global_mean})
    results = get_results(v6_info[CLIENT], task, print_log=False)
    full_df = pd.concat([avg_result['std'] for avg_result in results], ignore_index=True)
    
    sizes = np.array([result['size'] for result in results])

    glob_var = full_df.sum(axis=0)/np.sum(sizes)
    global_std = glob_var.pow(0.5).to_frame().T
    return global_std

def normalize_workflow(v6_info, data_settings):
    global_mean, _ = mean_workflow(v6_info, copy.deepcopy(data_settings)) # still required for standard error calculation (probably)

    if data_settings[NORMALIZE] == "global":
        #global_mean = mean_workflow(v6_info, data_settings)
        global_std = std_workflow(v6_info, copy.deepcopy(data_settings), global_mean)
    elif data_settings[NORMALIZE] == "local" or data_settings[NORMALIZE] == "none":
        global_std = None
    
    return global_mean, global_std


def se_workflow(v6_info, data_settings, classif_settings):
    task = post_vantage_task(v6_info, "calc_se", {"data_settings" : data_settings, "classif_settings" : classif_settings})
    results = get_results(v6_info[CLIENT], task)

    tops = np.concatenate([result[TOP] for result in results])
    bots = np.array([result[BOT] for result in results])
    sizes = np.array([result[SIZE] for result in results])
    full_size = np.sum(sizes)

    bot_full = np.sum(bots, axis = 0)
    top_sum = np.sum(tops)
    bot_inv = np.linalg.pinv(bot_full)
    bot_sum = bots.sum(axis = 0)

    se = np.sqrt((np.diag(bot_inv) * top_sum) * (1/ (full_size - bot_sum.shape[0])))

    return se


def hase_workflow(v6_info, data_settings):
    task = post_vantage_task(v6_info, "calc_ABC", {"data_settings" : data_settings})

    results = get_results(v6_info[CLIENT], task)

    A = np.array([ABC_result['A'] for ABC_result in results])
    B = np.array([ABC_result['B'] for ABC_result in results])
    C = np.array([ABC_result['C'] for ABC_result in results])
    
    sizes = np.array([result[SIZE] for result in results])
    full_size = np.sum(sizes)
   


    full_A = np.sum(A, axis = 0)
    full_B = np.sum(B, axis = 0)
    full_C = np.sum(C, axis = 0)


    A_inv = np.linalg.pinv(full_A)
    beta_hat = A_inv @ full_B


    A_inv_diag = np.diag(A_inv)
    df = full_size -  full_B.shape[0]
    bot = full_C - full_B.T @ A_inv @ full_B

    se = np.sqrt((A_inv_diag * bot)/ df)

    return beta_hat, se
