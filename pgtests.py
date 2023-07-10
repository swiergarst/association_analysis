# this file holds all functions used for testing on postgres databases locally


from vantage6.client import Client
from utils import init_global_params, average, define_model
import numpy as np
from io import BytesIO


client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "password")
client.setup_encryption(None)
ids = [org['id'] for org in client.collaboration.get(1)['organizations']]


model = "M5"
lr = 0.05

fit = False
create = True
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

    print(result[0]['log'])

    results = [np.load(BytesIO(res['result']), allow_pickle=True) for res in result]
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


### I probably won't need these functions anymore, but just in case I decided to keep them in this file as an archive

import os
from vantage6.tools.util import info
import psycopg2


def RPC_pgdb_print(db_client, PG_URI = None, col = None):

        
    # could also use this to set URI
    if PG_URI == None:
        PG_URI = 'postgresql://{}:{}@{}:{}'.format(
            os.getenv("PGUSER"),
            os.getenv("PGPASSWORD"),
            os.getenv("PGHOST"),
            os.getenv("PGPORT"))


   
    info("connecting to PG DB:" + str(PG_URI))

    # get data from postgres DB
    try:
        connection = psycopg2.connect(PG_URI)
        cursor = connection.cursor()

        info("connected to PG database")
        if col == None:
            data_pg = cursor.execute("SELECT * FROM ncdc")
        else:
            data_pg = cursor.execute("SELECT {} FROM ncdc".format(col))
        info(str(cursor.fetchall()))
        data = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return data
    except psycopg2.Error as e:
        info("error: " + str(e))

        cursor.close()
        connection.close()
        return e






def RPC_create_db(db_client, PG_URI = None):


    rng = np.random.default_rng()
    if PG_URI == None:
        PG_URI = 'postgresql://{}:{}@{}:{}'.format(
            os.getenv("PGUSER"),
            os.getenv("PGPASSWORD"),
            os.getenv("PGHOST"),
            os.getenv("PGPORT"))

    info("connecting to PG DB:" + str(PG_URI))

    try:
        connection = psycopg2.connect(PG_URI)
        #print(connection)

        cursor = connection.cursor()

        info("connected")
        all_cols =  ["metabo_age", "brain_age", "date_metabolomics", "date_mri", "birth_year", "sex", "dm", "bmi", "education_category"]
        vals = abs(rng.standard_normal(size = (len(all_cols), 30))).astype(object)
        #print(len(all_cols))

        
        info("adding ncdc table")
        cursor.execute("DROP TABLE IF EXISTS ncdc")
        cursor.execute("CREATE TABLE ncdc ()")
        for i, col in enumerate(all_cols):
            if col == "date_metabolomics" or col == "date_mri":
                #modify the value in 'vals' for these columns
                base_time = np.datetime64('2010-01-01')
                vals[i,:] = np.array(base_time + rng.integers(0, 2000, size = len(vals[i,:]))).astype(str)
                query = "ALTER TABLE ncdc ADD " + col + " date;"

            else:
                query = "ALTER TABLE ncdc ADD " + col + " int;"
            cursor.execute(query)
            #cursor.execute("""ALTER TABLE ncdc ADD %s float;""", (col,))
        info("adding dummy data to ncdc table")
        for value in vals.T:
            #query = "INSERT INTO ncdc '" + col + "' VALUES " + str(value)
            #cursor.execute(query)

            cursor.execute("INSERT INTO ncdc VALUES {}".format(tuple(value)))
        


        cursor.execute("SELECT * FROM ncdc")
        data = cursor.fetchall()

        info("done!")

        connection.commit()
        cursor.close()
        connection.close()

        return data
    except psycopg2.Error as e:
        info("error: " + str(e))

        return e