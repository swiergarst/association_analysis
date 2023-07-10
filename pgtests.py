from vantage6.client import Client


print("posting create task")

client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "password")
client.setup_encryption(None)
ids = [org['id'] for org in client.collaboration.get(1)['organizations']]


'''
task = client.post_task(
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
    image = "sgarst/association-analysis:1.1",
    organization_ids=ids,
    collaboration_id=1
)
'''

create_task = client.post_task(
    input_ = {
        'method' : 'create_db',
        'kwargs' : {            }
        },
    name = "Analysis fit regressor, round" + str(round),
    image = "sgarst/association-analysis:pgTest2",
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


print("posting print task")

print_task = client.post_task(
    input_ = {
        'method' : 'pgdb_print',
        'kwargs' : {            }
        },
    name = "Analysis fit regressor, round" + str(round),
    image = "sgarst/association-analysis:pgTest2",
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
print(result[0]['result'])
#results = [np.load(BytesIO(result[i]['result']), allow_pickle=True) for i in range(3)]






exit()
