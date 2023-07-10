from sklearn.linear_model import SGDRegressor
import numpy as np
from vantage6.tools.util import info
import math
import os
import psycopg2
import pandas as pd

def master():
    pass

def RPC_fit_round(db_client, coefs, intercepts, data_cols, extra_cols, lr, seed, PG_URI = None, all_cols = [None]):


    info("starting fit_round")
    #TODO: calc metaboage/health through PHT
    
    data_df = pd.DataFrame()

    # in case we want to use different columns later on, no need to change image this way
    if None in all_cols:
        all_cols =  ["metabo_age", "brain_age", "date_blood", "date_mri", "birth_year", "sex", "dm", "bmi", "education_category"]
    
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
            data_df[col] = cursor.fetchall()
    except psycopg2.Error as e:
        return e

    info("dataframe built")
    data = data_df[all_cols].dropna()

    blood_dates = data["date_blood"].values
    mri_dates = data["date_mri"].values

    info("blood dates: " + str(blood_dates[0]))
    blood_dates_years = np.array([int(blood_date[:4]) for blood_date in blood_dates])
    mri_dates_years = np.array([int(mri_date[:4]) for mri_date in mri_dates])

    Age_met = blood_dates_years - data["birth_year"].values
    Age_brain = mri_dates_years - data["birth_year"].values


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
    
    X = data[data_cols].values.reshape(-1, 1)
    y = data["brain_age"].values.reshape(-1, 1)
    train_inds = rng.choice(len(X), math.floor(len(X)* 0.8), replace=False)
    train_inds = np.sort(train_inds)
    train_mask = np.zeros((len(X)), dtype=bool)
    train_mask[train_inds] = True
    X_train = X[train_mask]
    X_test = X[~train_mask]
    
    y_train = y[train_mask]
    y_test = y[~train_mask]
    #info("train inds: " + str(train_inds))
    #info("train shape: " + str(X_train.shape))
    #info("test shape: " + str(X_test.shape))
    #info("X full shape: " + str(X.shape))
    

    #info("data loaded")

    model = SGDRegressor(loss="squared_error", penalty=None, max_iter = 1, eta0=lr)
    
    model.coef_ = np.copy(coefs)
    model.intercept_ = np.copy(intercepts)
    
    #info("training data shape: " + str(y_train.shape))

    model.partial_fit(X_train, y_train)

    info("model fitted")


    loss = np.mean((model.predict(X_test) - y_test) **2)

    #TODO: how do we make sure which coefs correspond to which covariates?
    return {
        "param": (model.coef_, model.intercept_),
        "loss": loss,
        "size": y_test.shape[0]
    }
    




### I probably won't need these functions anymore, but just in case I decided to keep them in this file as an archive

import os
from vantage6.tools.util import info
import psycopg2


def RPC_pgdb_print(db_client, PG_URI = None):

        
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
        data_pg = cursor.execute("SELECT * FROM ncdc")
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
        all_cols =  ["metabo_age", "brain_age", "date_blood", "date_mri", "birth_year", "sex", "dm", "bmi", "education_category"]
        vals = abs(rng.standard_normal(size = (len(all_cols), 30))).astype(object)
        #print(len(all_cols))

        
        info("adding ncdc table")
        cursor.execute("DROP TABLE IF EXISTS ncdc")
        cursor.execute("CREATE TABLE ncdc ()")
        for i, col in enumerate(all_cols):
            if col == "date_blood" or col == "date_mri":
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