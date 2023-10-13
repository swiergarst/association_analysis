import psycopg2
from vantage6.tools.util import info
#from constants import *
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from V6_implementation.v6_LinReg_py.constants import *



def RPC_get_avg(db_client, data_settings):
    data_tmp = build_dataframe(data_settings[DIRECT_COLS])

    data = complete_dataframe(data_tmp, data_settings[SYNTH_COLS],data_settings[USE_DELTAS])

    return {"averages" : data.mean().to_frame().T,
            "size" : data.shape[0]
            }


def build_dataframe(cols, PG_URI = None):

    info(f'building dataframe with columns: {cols}')
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
        
        # also add id to the initial dataframe, so we can sort using that
        data_pg = cursor.execute("SELECT {} FROM ncdc".format(ID))
        column_data = cursor.fetchall()
        data_df[ID] = [column_val[0] for column_val in column_data]

    except psycopg2.Error as e:
        info(f"psycopg2 error: {e}")
        return e
    
    # merge rows from same patients (but different visits)
    data_df = data_df.groupby([ID]).agg({col : 'first' for col in cols}).reset_index()

    data = data_df[cols].dropna()
    info("base dataframe built.")

    return data

def complete_dataframe(df, cols_to_add, use_deltas = False):
    info("adding columns based on covariates")

    if (LAG_TIME in cols_to_add) or (AGE in cols_to_add) or (use_deltas == True):
        #calculate age at blood draw and mri scan
        blood_dates = np.array([date.strftime("%Y") for date in df[DATE_METABOLOMICS].values]).astype(int)
        mri_dates = np.array([date.strftime("%Y") for date in df[DATE_MRI].values]).astype(int)

        Age_met = blood_dates - df[BIRTH_YEAR].values
        Age_brain = mri_dates- df[BIRTH_YEAR].values
        if LAG_TIME in cols_to_add:
            df[LAG_TIME] = Age_met - Age_brain
        if (AGE in cols_to_add):
            df[AGE] = np.mean(np.vstack((Age_met, Age_brain)), axis = 0)
        if (use_deltas == True):
            age = np.mean(np.vstack((Age_met, Age_brain)), axis = 0)
            df[METABO_AGE] = df[METABO_AGE].values.astype(float) - age
            df[BRAIN_AGE] = df[BRAIN_AGE].values.astype(float) - age

    if SENSITIVITY_1 in cols_to_add:
        df = df.loc[abs(df[LAG_TIME]) <= 1].reset_index(drop=True)
    elif SENSITIVITY_2 in cols_to_add:
        df = df.loc[abs(df[LAG_TIME]) <= 2].reset_index(drop=True)

    if EDUCATION_CATEGORY in cols_to_add:
        enc = OneHotEncoder(categories = [[0, 1, 2]], sparse=False)
        mapped_names = [EC1, EC2, EC3]
        mapped_arr = enc.fit_transform(df[EDUCATION_CATEGORY].values.reshape(-1, 1))
        df[mapped_names] = mapped_arr
    info("dataframe finished")
    return df