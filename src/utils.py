import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from src.logger import logging
from src.exception import CustomException
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        for name, model in models.items():          
            model.fit(x_train, y_train)

            train_score = r2_score(y_train, model.predict(x_train))
            test_score  = r2_score(y_test,  model.predict(x_test))

            logging.info(f"{name} | Train R²: {train_score:.4f} | Test R²: {test_score:.4f}")
            report[name] = test_score               

        return report
    except Exception as e:
        raise CustomException(e, sys)

    except Exception as e:
        raise CustomException(e,sys)
