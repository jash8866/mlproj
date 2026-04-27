import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

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
    
def evaluate_models(x_train, y_train, x_test, y_test, models,params):
    try:
        report = {}
        for name, model in models.items():          
            para=params[name]

            if len(para) == 0:
                model.fit(x_train, y_train)
                best_model = model
            else:
                gs=RandomizedSearchCV(model,para,cv=3,n_iter=10,n_jobs=-1)
                gs.fit(x_train,y_train)
                best_model = gs.best_estimator_

            train_score = r2_score(y_train, best_model.predict(x_train))
            test_score  = r2_score(y_test,  best_model.predict(x_test))

            logging.info(f"{name} | Train R²: {train_score:.4f} | Test R²: {test_score:.4f}")
            report[name] = test_score               

        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise  CustomException(e,sys)