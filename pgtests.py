from vantage6.client import Client
from utils import init_global_params, average, define_model
import numpy as np
from io import BytesIO


client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "password")
client.setup_encryption(None)
ids = [org['id'] for org in client.collaboration.get(1)['organizations']]


model = "M2"
lr = 0.05

fit = True
create = False
pr = False
image_name = "sgarst/association-analysis:1.1.2"

data_cols, extra_cols = define_model(model)
global_coefs, global_intercepts = init_global_params(data_cols, extra_cols)

if fit:
    print("posting fit task")
    fit_task = client.post_task(
        input_ = {
            'method' : 'fit_round',
            'kwargs' : {
                'coefs' : global_coefs,
                "intercepts" :  global_intercepts,
                "data_cols" : data_cols,
                "extra_cols" : extra_cols,
                "lr" : lr,
                "seed": 42
                }
            },
        name = "Analysis fit regressor, round" + str(round),
        image = image_name,
        organization_ids=ids,
        collaboration_id=1
    )

    finished = False
    while (finished == False):
        result = client.get_results(task_id=fit_task.get("id"))
        if not None in [result[i]['result'] for i in range(len(result)) ]:
            finished = True

    results = [np.load(BytesIO(res['result']), allow_pickle=True) for res in result]

    print(result[0]['log'])
    print(results[0])



if create:
    print("posting create task")

    create_task = client.post_task(
        input_ = {
            'method' : 'create_db',
            'kwargs' : {            }
            },
        name = "Analysis fit regressor, round" + str(round),
        image = image_name,
        organization_ids=ids,
        collaboration_id=1
    )

    finished = False

    while (finished == False):
        result = client.get_results(task_id=create_task.get("id"))
        #print(result)
        #time.sleep(10)
        if not None in [result[i]['result'] for i in range(len(result)) ]:
            finished = True

    print("create task finished: ")
    for res in result:
        print(res['log'])
#print([result[i]['log'] for i in range(len(result))])

if pr:
    print("posting print task")

    print_task = client.post_task(
        input_ = {
            'method' : 'pgdb_print',
            'kwargs' : { 
                'col' : "metabo_age"
                       }
            },
        name = "Analysis fit regressor, round" + str(round),
        image = image_name,
        organization_ids=ids,
        collaboration_id=1
    )

    finished = False

    while (finished == False):
        result = client.get_results(task_id=print_task.get("id"))
        #print(result)
        #time.sleep(10)
        if not None in [result[i]['result'] for i in range(len(result)) ]:
            finished = True


    print("print task finished: ")
    for res in result:
        print(res['log'])


#print([res['log'] for res in  result])
#print(result[0]['result'])
#results = [np.load(BytesIO(result[i]['result']), allow_pickle=True) for i in range(3)]






exit()
