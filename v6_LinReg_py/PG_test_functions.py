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
        vals = abs(rng.standard_normal(size = (len(all_cols), 30)))
        #print(len(all_cols))

        
        info("adding ncdc table")
        cursor.execute("CREATE TABLE ncdc ()")
        for i, col in enumerate(all_cols):
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