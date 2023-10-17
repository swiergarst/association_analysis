from V6_implementation.v6_LinReg_py.constants import *
import numpy as np
import pandas as pd
from .utils2 import get_results, post_vantage_task

def mean_workflow(v6_info, data_settings):
    # this makes the client-side code a little more flexible
    data_settings[DIRECT_COLS].append(data_settings[TARGET])

    task = post_vantage_task(v6_info, "get_avg", {"data_settings": data_settings})
    results = get_results(v6_info[CLIENT], task)

    # means = avg_results[0]['averages']
    # for i in range(1, len(avg_results)):
    #     means.append(avg_results[i]['averages'])

    full_df = pd.concat([avg_result['averages'] for avg_result in results], ignore_index=True)
    sizes = np.array([avg_result['size'] for avg_result in results])
    

    global_mean_arr = np.array([np.sum([(mean * size) for mean, size in zip(full_df.values, sizes)], axis = 0) / np.sum(sizes)])
    global_df = pd.DataFrame(data=global_mean_arr, columns = full_df.columns)
    #print(global_df.head())
    #print(global_mean_df.head())
    return(global_df)


def std_workflow(v6_info, data_settings, global_mean):
    data_settings[DIRECT_COLS].append(data_settings[TARGET])

    #print(global_mean.head())
    task = post_vantage_task(v6_info, "get_std", {"data_settings": data_settings, "global_mean" : global_mean})
    results = get_results(v6_info[CLIENT], task, print_log=False)
    full_df = pd.concat([avg_result['std'] for avg_result in results], ignore_index=True)
    # print(f'full_df: { full_df.head()}')
    sizes = np.array([result['size'] for result in results])
    # print(f'total size: {np.sum(sizes)}')
    glob_var = full_df.sum(axis=0)/np.sum(sizes)
    # print(glob_var)
    global_std = glob_var.pow(0.5).to_frame().T
    #global_var_sum = glob_var.pow(2).to_frame().T
    #global_std = (full_df.sum(axis=0)/np.sum(sizes)).pow(0.5).to_frame().T
    # print(global_std)
    #print(f'glob var: {glob_var.head()}')
    #print(f'glob var sq: {global_var_sum.head()}')
    return global_std

def normalize_workflow(v6_info, data_settings):
    global_mean = mean_workflow(v6_info, data_settings) # still required for standard error calculation (probably)

    if data_settings[NORMALIZE] == "global":
        #global_mean = mean_workflow(v6_info, data_settings)
        global_std = std_workflow(v6_info, data_settings, global_mean)
    elif data_settings[NORMALIZE] == "local" or data_settings[NORMALIZE] == "local":
        global_std = None
    
    return global_mean, global_std

