import psycopg2
from vantage6.tools.util import info
#from constants import *
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from constants import *




def RPC_get_avg(db_client, cols_to_avg, data_settings):
    data_tmp = build_dataframe(cols_to_avg[DIRECT_COLS])

    data = complete_dataframe(data_tmp, cols_to_avg[SYNTH_COLS],data_settings[USE_DELTAS])

    return {"averages: " : data.mean(),
            "size" : data.shape[1]
            }


def build_dataframe(cols, PG_URI = None):

    data_df = pd.DataFrame()

    if PG_URI == None:
            PG_URI = 'postgresql://{}:{}@{}:{}'.format(
                os.getenv("PGUSER"),
                os.getenv("PGPASSWORD"),
                os.getenv("PGHOST"),
                os.getenv("PGPORT"))
    info("connecting to PG DB:" + str(PG_URI))
    try:
        connection = psycopg2.connect(PG_URI)
        cursor = connection.cursor()
        info("connected to PG database. building dataframe..")

        #incrementally build dataframe
        for col in cols:
            data_pg = cursor.execute("SELECT {} FROM ncdc".format(col))
            column_data = cursor.fetchall()

            #fetchall returns a tuple for some reason, so we have to do this
            data_df[col] = [column_val[0] for column_val in column_data]
    except psycopg2.Error as e:
        return e
    
        # merge rows from same patients (but different visits)

    merge_cols = cols.copy()
    merge_cols.remove(ID)
    data_df = data_df.groupby([ID]).agg({col : 'first' for col in merge_cols}).reset_index()

    data = data_df[cols].dropna()
    info("base dataframe built.")

    return data

def complete_dataframe(df, cols_to_add, use_deltas = False):
    info("adding columns based on covariates")

    if (LAG_TIME in cols_to_add) or (AGE in cols_to_add) or (use_deltas == True):
        #calculate age at blood draw and mri scan
        blood_dates = np.array([date.strftime("%Y") for date in data[DATE_METABOLOMICS].values]).astype(int)
        mri_dates = np.array([date.strftime("%Y") for date in data[DATE_MRI].values]).astype(int)

        Age_met = blood_dates - data[BIRTH_YEAR].values
        Age_brain = mri_dates- data[BIRTH_YEAR].values
        if LAG_TIME in cols_to_add:
            data[LAG_TIME] = Age_met - Age_brain
        if (AGE in cols_to_add):
            data[AGE] = np.mean(np.vstack((Age_met, Age_brain)), axis = 0)
        if (use_deltas == True):
            age = np.mean(np.vstack((Age_met, Age_brain)), axis = 0)
            data[METABO_AGE] = data[METABO_AGE].values.astype(float) - age
            data[BRAIN_AGE] = data[BRAIN_AGE].values.astype(float) - age

    if SENSITIVITY_1 in cols_to_add:
        data = data.loc[abs(data[LAG_TIME]) <= 1].reset_index(drop=True)
    elif SENSITIVITY_2 in cols_to_add:
        data = data.loc[abs(data[LAG_TIME]) <= 2].reset_index(drop=True)

    if EDUCATION_CATEGORY in cols_to_add:
        enc = OneHotEncoder(categories = [[0, 1, 2]], sparse=False)
        mapped_names = [EC1, EC2, EC3]
        mapped_arr = enc.fit_transform(data[EDUCATION_CATEGORY].values.reshape(-1, 1))
        data[mapped_names] = mapped_arr

    return data