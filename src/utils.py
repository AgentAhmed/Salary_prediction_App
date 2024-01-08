import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pymysql

import pickle
import numpy as np

load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv('db')



def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established",mydb)
        df=pd.read_sql_query('Select * from survey_results_public',mydb)
        print(df.head())

        return df



    # except Exception as ex:
    #     raise CustomException(ex)
    except Exception as ex:
        raise CustomException(str(ex), sys.exc_info())

