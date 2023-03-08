from vantage6.client import Client
import os
import numpy as np
import json
from utils import init_global_params, average, define_model
dir_path = os.path.dirname(os.path.realpath(__file__))


#TODO: change these for Leiden server
print("Attempt login to Vantage6 API")
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "1234")


privkey = dir_path + "/home/swier/Documents/FedvsCent/privkeys/privkey_testOrg0.pem"
#client.setup_encryption(privkey)

#TODO: know which ID is from which center
ids = [org['id'] for org in client.collaboration.get(1)['organizations']]


## Parameter settings ##

n_runs = 1 # amount of runs to ensure low variance
n_rounds = 10 # communication rounds between centers
lr = 0.0001 # learning rate
model = "M1" # model selection (see analysis plan)


## init data structures ## 

betas = np.zeros((n_runs, n_rounds, 3))
losses = np.zeros_like(betas)
cols = define_model(model)


for run in range(n_runs):
    # federated iterative process
    global_coefs, global_intercepts = init_global_params(cols)
    for round in range(n_rounds):
        task = client.post_task(
            input_ = {
                'method' : 'fit_round',
                'kwargs' : {
                    'parameters' : [global_coefs, global_intercepts],
                    "cols" : cols,
                    "lr" : lr
                    }
                },
            name = "Analysis fit regressor, round" + str(round),
            image = "sgarst/federated-learning:fedReg",
            organization_ids=ids,
            collaboration_id=1
        )
        finished = False
        #local_coefs = np.empty((n_rounds, 3), dtype=object)
        #local_intercepts = np.empty((n_rounds, 3), dtype=object)
        dataset_sizes = np.empty(3, dtype = object)

        while (finished == False):
            result = client.get_results(task_id=task.get("id"))
            if not None in result:
                finished = True
            
        #TODO: fix this
        local_coefs, local_intercepts = result['param']
        dataset_sizes = result['size']
        losses = result['loss']

        global_coefs = average(local_coefs, dataset_sizes)
        global_intercepts = average(local_intercepts, dataset_sizes)


# write output to json file
final_results = {
    "lr" : lr,
    "nruns" : n_runs,
    "nrounds" : n_rounds,
    "model" : model,
    "betas" : betas,
    "cols" : cols,
    "loss" : losses
}

file_str = "analysis_model_" + model + ".json"
jo = json.dumps(final_results)

with open(file_str, "wb") as f:
    f.write(jo)









