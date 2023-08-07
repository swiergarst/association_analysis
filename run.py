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

print("Attempt login to Vantage6 API")
# client = Client("https://vantage6-server.researchlumc.nl", 443, "/api")
# client.authenticate("sgarst", "password")
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
ids = [2,3]

## Parameter settings ##

n_runs = 1 # amount of runs 
n_rounds = 2 # communication rounds between centers
lr = 0.000005 # learning rate
model = "M3" # model selection (see analysis plan)
n_bins = 10
write_file = True
use_dm = False
n_clients = len(ids)
seed_offset = 0

#all_cols =  ["id", "metabo_age", "brain_age","date_metabolomics", "date_mri","birth_year", "sex", "bmi", "dm", "education_category_3" ]
#all_cols = [None]
image_name = "sgarst/association-analysis:1.4.2"
## init data structures ## 

betas = np.zeros((n_runs, n_rounds, n_clients))
mses = np.zeros_like(betas)
maes = np.zeros_like(betas)
data_cols, extra_cols = define_model(model, use_dm = use_dm)

for run in range(n_runs):

    param_seed = run + seed_offset
    tt_split_seed = run + seed_offset
    # federated iterative process
    global_coefs, global_intercepts = init_global_params(data_cols, extra_cols, param_seed = param_seed)
    
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
                    #"all_cols": all_cols,
                    "lr" : lr,
                    "seed": tt_split_seed,
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
    avg_task = client.post_task(
        input_= {
            "method" : "get_avg",
            "kwargs" : {     }
        },
        name= "get average",
        image = image_name,
        organization_ids= ids,
        collaboration_id=1
    )
    avg_results = get_results(client, avg_task, print_log=False)

    means = np.array([result['mean'] for result in avg_results])
    sizes = np.array([result['size'] for result in avg_results])

    global_mean = np.sum([(mean * size) for mean, size in zip(means, sizes)]) / np.sum(sizes)

    #print(means, global_mean)

    se_task = client.post_task(
        input_ = {
            "method" : "calc_se",
            "kwargs" : {
                "global_mean" : global_mean,
                "global_coefs" : global_coefs,
                "global_inter" : global_intercepts,
                "data_cols" : data_cols,
                #"all_cols" : all_cols,
                "extra_cols" : extra_cols,
            }
        },
            name ="se calculation",
            image = image_name,
            organization_ids=ids,
            collaboration_id=1
    )   

    se_results = get_results(client, se_task, print_log = False)
    tops = [np.sum(result['top']) for result in se_results]
    bots = [np.sum(result['bot']) for result in se_results]
    sizes = [result['size'] for result in se_results]

    top_sum = np.sum(tops)
    bot_sum = np.sum(bots)
    full_size = np.sum(sizes)

    se = np.sqrt((1/(full_size - 2)) * (float(top_sum) / float(bot_sum)))

    #print(f'standard error: {se}')

branges = [result['resplot']['ranges'].tolist() for result in results]

print("finished! writing to file")
print(mses)
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
    "standard_error" : se
}

if write_file:
    date = datetime.datetime.now()
    file_str = f'analysis_model_{model}_centers_{ids}_{date.strftime("%d")}_{date.strftime("%m")}_{date.strftime("%Y")}'
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



