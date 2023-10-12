from vantage6.client import Client
import os
import numpy as np
import json
from io import BytesIO
import datetime
import psycopg2
from utils import init_global_params, average, define_model, get_results, normalize_workflow
import time
import matplotlib.pyplot as plt
dir_path = os.path.dirname(os.path.realpath(__file__))
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import SGDRegressor

print("Attempt login to Vantage6 API")
# client = Client("https://vantage6-server.researchlumc.nl", 443, "/api")
# client.authenticate("researcher", "password")
# client.setup_encryption(None)

client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "password")
client.setup_encryption(None)
ids = [org['id'] for org in client.collaboration.get(1)['organizations']]

#ID mapping:
# 2 - Leiden
# 3 - Rotterdam
# 4 - Maastricht


#ids = [org['id'] for org in client.collaboration.get(1)['organizations']]
#ids = [2,3]
#ids = [2]
## Parameter settings ##

n_runs = 1 # amount of runs 
n_rounds = 2 # communication rounds between centers
lr = 0.005 # learning rate
model = "M3" # model selection (see analysis plan)
bin_width = 0.2
write_file = False
use_dm = True
use_age = True
use_deltas = True #whether to look at delta metabo/brainage
normalize_cat = True
normalize = "global" # options: local, global, none
n_clients = len(ids)
seed_offset = 0

all_cols =  ["id", "metabo_age", "brain_age", "date_metabolomics", "date_mri","birth_year", "sex",  "dm", "education_category_3", "bmi"]
#all_cols = [None]
image_name = "sgarst/association-analysis:haseTest"
#image_name = "sgarst/federated-learning:1.6"
## init data structures ## 

betas = np.zeros((n_runs, n_rounds, n_clients))
mses = np.zeros_like(betas)
maes = np.zeros_like(betas)
data_cols, extra_cols, to_norm_cols = define_model(model, use_dm = use_dm, use_age = use_age)
#data_cols_norm = data_cols.copy()
#data_cols_norm.append("metabo_age")
#print(data_cols_norm)
def verify_se(X_full, y_full, coefs, n_tot):
    lm = SGDRegressor(fit_intercept=False)
    lm.coef_ = coefs
    lm.intercept_ = 0
    y_pred = lm.predict(X_full)
    res = np.sum((y_full - y_pred)**2)
    #sos = res.T @ res
    p = X_full.shape[1]
    bot = np.sum((X_full - np.mean(X_full, axis = 0))**2, axis = 0)
    print(f'bot shape: {bot.shape}')
    #var_beta_hat = np.linalg.inv(X_full.T @ X_full) * sigma_squared_hat
    se = np.sqrt(1/ (n_tot - p) * (res / bot))
    return se

def verify_hase_se(X_full, y_full, coefs, n_tot):
    lm = SGDRegressor(fit_intercept=False)
    lm.coef_ = coefs
    lm.intercept_ = 0
    y_pred = lm.predict(X_full)
    res = np.sum((y_full - y_pred)**2)
    #sos = res.T @ res
    p = X_full.shape[1]
    bot = np.diag(np.linalg.pinv(X.T @ X))
    #var_beta_hat = np.linalg.inv(X_full.T @ X_full) * sigma_squared_hat
    se = np.sqrt(1/ (n_tot - p) * (res * bot))
    return se

for run in range(n_runs):

    param_seed = run + seed_offset
    tt_split_seed = run + seed_offset
    # federated iterative process
    global_coefs, global_intercepts = init_global_params(data_cols, extra_cols, param_seed = param_seed)
    # global_mean, global_std, _ = normalize_workflow(client, image_name,  to_norm_cols, extra_cols, use_deltas, normalize, normalize_cat = normalize_cat)
    global_mean, global_std, _, avg_cols = normalize_workflow(client, image_name,  data_cols, extra_cols, use_deltas, normalize, normalize_cat = normalize_cat)

    print(global_std.shape, global_mean.shape)




    for round in tqdm(range(n_rounds)):
        #print(global_coefs, global_intercepts)
        #print("posting fit_round task to ids " + str(ids))
        task = client.post_task(
            input_ = {
                'method' : 'fit_round',
                'kwargs' : {
                    'coefs' : global_coefs,
                    "intercepts" :  global_intercepts,
                    "data_cols" : data_cols,
                    "extra_cols" : extra_cols,
                    "all_cols": all_cols,
                    "lr" : lr,
                    "bin_width" : bin_width,
                    "seed": tt_split_seed,
                    "normalize" : normalize,
                    "global_mean" : global_mean,
                    "global_std" : global_std,
                    "use_deltas" : use_deltas,
                    "normalize_cat" : normalize_cat
                    }
                },
            name = "Analysis fit regressor, round" + str(round),
            image = image_name,
            organization_ids=ids,
            collaboration_id=1
        )
        finished = False
        #local_coefs = np.empty((n_rounds, 3), dtype=object)
        #local_intercepts = np.empty((n_rounds, 3), dtype=object)
        dataset_sizes = np.empty(n_clients, dtype = object)
        local_coefs = np.empty((len(global_coefs), n_clients))
        local_intercepts = np.empty((len(global_intercepts),n_clients))
        local_cols = np.empty((n_clients), dtype = object)

        results = get_results(client, task, print_log = True)

        if psycopg2.Error in results:
            print("query error: ", results)
            break
        else:
            for i in range(n_clients):
                local_coefs[:,i], local_intercepts[:, i] = results[i]['param']
                dataset_sizes[i] = results[i]['size']
                mses[run, round, i] = results[i]['mse']
                maes[run, round, i] = results[i]["mae"]
                local_cols[i] = results[i]["data_cols"]
                #print(local_cols[i])
        #print(local_coefs, local_intercepts)
        global_coefs = average(local_coefs, dataset_sizes)
        global_intercepts = average(local_intercepts, dataset_sizes)
        #print(global_coefs, global_intercepts)
        #betas[run, round, 0] = np.copy(global_coefs)
#print(losses)

    print('finished model fitting, calculating standard error..')

    se_task = client.post_task(
        input_ = {
            "method" : "calc_se",
            "kwargs" : {
                "global_mean" : global_mean,
                "global_std" : global_std,
                "global_coefs" : global_coefs,
                "global_inter" : global_intercepts,
                "data_cols" : data_cols,
                "all_cols" : all_cols,
                "extra_cols" : extra_cols,
                "use_deltas" : use_deltas,
                "normalize" : normalize,
                "norm_cat" : normalize_cat

            }
        },
            name ="se calculation",
            image = image_name,
            organization_ids=ids,
            collaboration_id=1
    )   

    se_results = get_results(client, se_task, print_log = True)
    tops = np.array([result['top'] for result in se_results])
    bots = np.array([result['bot'] for result in se_results])
    sizes = [result['size'] for result in se_results]

    top_sum = np.sum(tops)
    bot_sum = np.sum(np.concatenate([bots[i,:,:] for i in range(bots.shape[0])]),axis = 0)
    full_size = np.sum(sizes)
    print(top_sum.shape, bot_sum.shape)

    se = np.sqrt((1/(full_size - bot_sum.shape[0])) * (float(top_sum) / bot_sum.astype(float)))
    print(f'standard error federated: {se}')


data_task = client.post_task(
    input_={
        "method" : "return_data",
        "kwargs" : {
            "all_cols" : all_cols,
            "data_cols" : data_cols,
            "extra_cols" : extra_cols,
            "use_deltas" : use_deltas,
            "normalize" : normalize,
            "global_mean" : global_mean,
            "global_std" : global_std,
            "normalize_cat" :normalize_cat
        }
    },
    name = "get data",
    image = image_name,
    organization_ids=ids,
    collaboration_id=1
)

data_results = get_results(client, data_task, print_log = False)

full_data = pd.concat([data_result['data'] for data_result in data_results])

central_data_cols = data_results[0]['data_cols']

X = full_data[central_data_cols].values.astype(float)
y = full_data["metabo_age"].values.astype(float)

se_central = verify_se(X, y, global_coefs,full_size)
se_hase_central = verify_hase_se(X, y, global_coefs, full_size)
print(f'se central: {se_central}')
print(f'se hase central: {se_hase_central}')


branges = [result['resplot']['ranges'].tolist() for result in results]

print("finished! writing to file")
print(maes)
# write output to json file
final_results = {
    "lr" : lr,
    "nruns" : n_runs,
    "nrounds" : n_rounds,
    "model" : model,
    "betas" : global_coefs.tolist(),
    "cols" : local_cols[0],
    "intercept" : global_intercepts.tolist(),
    "mse" : mses.tolist(),
    "mae" : maes.tolist(),
    "bin_ranges" : branges,
    "standard_error" : se,
    "sizes" : sizes
}

print(branges)
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



