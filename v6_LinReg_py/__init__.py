from sklearn.linear_model import SGDRegressor
import numpy as np
from vantage6.tools.util import info
import math
import os
import psycopg2
import pandas as pd
from datetime import datetime

def master():
    pass

def RPC_fit_round(db_client, coefs, intercepts, data_cols, extra_cols, lr, seed, PG_URI = None, all_cols = [None]):


    info("starting fit_round")
    #TODO: calc metaboage/health through PHT
    
    data_df = pd.DataFrame()

    # in case we want to use different columns later on, no need to change image this way
    if None in all_cols:
        all_cols =  ["metabo_age", "brain_age", "date_metabolomics", "date_mri", "birth_year", "sex", "dm", "bmi", "education_category"]
    
    # could also use this to set URI
    if PG_URI == None:
        PG_URI = 'postgresql://{}:{}@{}:{}'.format(
            os.getenv("PGUSER"),
            os.getenv("PGPASSWORD"),
            os.getenv("PGHOST"),
            os.getenv("PGPORT"))

       # PG_URI = 'postgresql://{}:{}@{}:{}'.format(
       #         os.getenv("DB_USER"),
       ##         os.getenv("DB_PASSWORD"),
       #         os.getenv("DB_HOST"),
       #         os.getenv("DB_PORT"))
   
    info("connecting to PG DB:" + str(PG_URI))
    # get data from postgres DB
    try:
        connection = psycopg2.connect(PG_URI)
        cursor = connection.cursor()
        info("connected to PG database. building dataframe..")

        #incrementally build dataframe
        for col in all_cols:
            data_pg = cursor.execute("SELECT {} FROM ncdc".format(col))
            column_data = cursor.fetchall()

            #fetchall returns a tuple for some reason, so we have to do this
            data_df[col] = [column_val[0] for column_val in column_data]
    except psycopg2.Error as e:
        return e

    info("dataframe built")
    data = data_df[all_cols].dropna()

    blood_dates = np.array([date.strftime("%Y") for date in data["date_metabolomics"].values]).astype(int)
    mri_dates = np.array([date.strftime("%Y") for date in data["date_mri"].values]).astype(int)

    Age_met = blood_dates - data["birth_year"].values
    Age_brain = mri_dates- data["birth_year"].values

    if "Lag_time" in extra_cols:
        data["Lag_time"] = Age_met - Age_brain
        data_cols.append("Lag_time")
    if "Age" in extra_cols:
        data["Age"] = np.mean(np.vstack((Age_met, Age_brain)), axis = 0)
        data_cols.append("Age")


    if "Sens_1" in extra_cols:
        data = data.loc[abs(data['Lag_time']) <= 1]
        #cols.remove("Sens_1")
    if "Sens_2" in extra_cols:
        data = data.loc[abs(data["Lag_time"]) <= 2]
        #cols.remove("Sens_2")

    
    info("calculated all covariates")


    rng = np.random.default_rng(seed = seed)
    
    X = data[data_cols].values
    y = data["brain_age"].values.reshape(-1, 1)

    info(str(X.shape) + str(len(X)))

    train_inds = rng.choice(len(X), math.floor(len(X)* 0.8), replace=False)
    train_inds = np.sort(train_inds)
    train_mask = np.zeros((len(X)), dtype=bool)
    train_mask[train_inds] = True
    X_train = X[train_mask]
    X_test = X[~train_mask]
    
    y_train = y[train_mask]
    y_test = y[~train_mask]

    info("X_train shape: " + str(X_train.shape))
    info("Y_train shape: " + str(y_train.shape))

    model = SGDRegressor(loss="squared_error", penalty=None, max_iter = 1, eta0=lr)
    
    model.coef_ = np.copy(coefs)
    model.intercept_ = np.copy(intercepts)
    
    model.partial_fit(X_train, y_train[:,0])

    info("model fitted")


    loss = np.mean((model.predict(X_test) - y_test) **2)

    return {
        "param": (model.coef_, model.intercept_),
        "data_cols" : data_cols,
        "loss": loss,
        "size": y_train.shape[0]
    }
    


<<<<<<< Updated upstream


### I probably won't need these functions anymore, but just in case I decided to keep them in this file as an archive

import os
from vantage6.tools.util import info
import psycopg2


=======
>>>>>>> Stashed changes
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


    frac_split = 0.5
    size_tosplit = math.floor(vals.shape[1] * frac_split)

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
        

        col_mri = all_cols.index("brain_age")
        col_mridate = all_cols.index("date_mri")
        col_birth = all_cols.index("birth_year")
        col_id = all_cols.index("id")

        cols_split = np.array([col_mri, col_mridate])
        cols_shared = np.array([col_id, col_birth])

        
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

        info("splitting dummy data")
        split_mask = np.zeros(len(all_cols), dtype= bool)
        split_mask[cols_split] = True
        share_mask = np.zeros(len(all_cols), dtype=bool)
        share_mask[cols_shared] = True


        vals_split1 = np.copy(vals[:,:size_tosplit])
        vals_split2 = np.copy(vals[:,:size_tosplit])

        vals_split1[split_mask & ~share_mask,:] = np.NaN
        vals_split2[~split_mask & ~share_mask,:] = np.NaN

        vals_nosplit = vals[:, size_tosplit:]

        vals = np.concatenate((vals_split1, vals_split2, vals_nosplit), axis = 1)


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