from vantage6.client import Client
import os
import numpy as np
import json
from io import BytesIO
import datetime
import psycopg2
from utils import init_global_params, average, define_model
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


ids = [org['id'] for org in client.collaboration.get(1)['organizations']]
#ids = [2, 3]

## Parameter settings ##

n_runs = 1 # amount of runs 
n_rounds = 3 # communication rounds between centers
lr = 0.000005 # learning rate
model = "M1" # model selection (see analysis plan)
n_bins = 3
write_file = False
n_clients = len(ids)
seed_offset = 0
## init data structures ## 

betas = np.zeros((n_runs, n_rounds, n_clients))
losses = np.zeros_like(betas)
data_cols, extra_cols = define_model(model)

for run in range(n_runs):
    param_seed = run + seed_offset
    tt_split_seed = run + seed_offset
    # federated iterative process
    global_coefs, global_intercepts = init_global_params(data_cols, extra_cols, param_seed = param_seed)
    
    for round in range(n_rounds):
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
                    "all_cols" : ["id", "metabo_age", "brain_age","date_metabolomics", "date_mri","birth_year", "dm"],
                    "lr" : lr,
                    "seed": tt_split_seed,
                    "n_bins":n_bins
                    }
                },
            name = "Analysis fit regressor, round" + str(round),
            image = "sgarst/association-analysis:1.3.1",
            organization_ids=ids,
            collaboration_id=1
        )
        finished = False
        #local_coefs = np.empty((n_rounds, 3), dtype=object)
        #local_intercepts = np.empty((n_rounds, 3), dtype=object)
        dataset_sizes = np.empty(n_clients, dtype = object)
        local_coefs = np.empty((len(global_coefs), n_clients))
        local_intercepts = np.empty((len(global_intercepts),n_clients))
        max_attempts = 20
        attempts = 0
        while (finished == False):
            attempts += 1
            result = client.get_results(task_id=task.get("id"))
            time.sleep(10)
            if not None in [res['result'] for res in result]:
                finished = True
            if attempts > max_attempts:
                print("max attempts exceeded")
                print(result)
                exit()

        
        #print("fit round task finished")
        #results = [res['result'] for res in result]
        
        for res in result:
            print(res['log'])

        #results = [np.load(BytesIO(res['result']), allow_pickle=True) for res in result]

        results = [pickle.loads(BytesIO(res['result']).getvalue()) for res in result]
        print(results[0]['size'], results[1]['size'])
        #print(results)

        '''
        fig1 = results[0]['resplot']['fig']
        fig2 = results[1]['resplot']['fig']

        ax1 = results[0]['resplot']['ax']
        bp1 = results[0]['resplot']['bp']
        ax1.set_title("figure uno")
        minlim1 = min([box.get_ydata()[1] for box in bp1['whiskers']])
        maxlim1 = max([box.get_ydata()[1] for box in bp1['whiskers']])

        plt.close(fig1)
        plt.close(fig2)
        # ax1.set_ylim([minlim1 , maxlim1*1.05])
        '''
        # ax1.set_ylim([0, 20])
        # fig1.show()


        
        # ax1.show()

        if psycopg2.Error in results:
            print("query error: ", results)
            break
        else:
            for i in range(n_clients):
                local_coefs[:,i], local_intercepts[:, i] = results[i]['param']
                dataset_sizes[i] = results[i]['size']
                losses[run, round, i] = results[i]['loss']

        #print(local_coefs, local_intercepts)
        global_coefs = average(local_coefs, dataset_sizes)
        global_intercepts = average(local_intercepts, dataset_sizes)
        #print(global_coefs, global_intercepts)
        betas[run, round, 0] = np.copy(global_coefs)
#print(losses)
fig1 = results[0]['resplot']['fig']
ax1 = results[0]['resplot']['ax']
branges = results[0]['resplot']['ranges']
#fig1.show()
#plt.show()

print("finished! writing to file")
# write output to json file
final_results = {
    "lr" : lr,
    "nruns" : n_runs,
    "nrounds" : n_rounds,
    "model" : model,
    "betas" : global_coefs.tolist(),
    "intercept" : global_intercepts.tolist(),
    "loss" : losses.tolist(),
    "bin_ranges" : branges.tolist()
}

if write_file:
    date = datetime.datetime.now()
    file_str = f'analysis_model_{model}_{date.strftime("%d")}_{date.strftime("%m")}_{date.strftime("%Y")}'
    jo = json.dumps(final_results)

    with open(file_str + ".json", "w", encoding="utf8") as f:
        json.dump(jo, f)
        #f.write(jo)

    with open("resplot_" + file_str + ".pickle", "wb") as f:
        pickle.dump(ax1, f)







