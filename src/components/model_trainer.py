# import os
# import sys
# from dataclasses import dataclass

# from catboost import  CatBoostRegressor
# from xgboost import XGBRegressor     

# from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
# from sklearn.linear_model import LinearRegression    
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor    

# from sklearn.metrics import r2_score

# from src.exception import CustomException
# from src.logger import logging    
# from src.utils import save_object,evaluate_models

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path=os.path.join("artifacts","model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config=ModelTrainerConfig()

#     def initiate_model_trainer(self,train_array,test_array):
#         try:
#             logging.info("Split trin and test innput data")
#             x_train,y_train,x_test,y_test=  (
#                 train_array[:,:-1],  #whole TrainArr except last column
#                 train_array[:,-1] ,  #only the last column
#                 test_array[:,:-1],   #whole TestArr except last column
#                 test_array[:,-1]     #only the last column
#             )  

#             models={
#                 "Random Forest": RandomForestRegressor(),
#                 "Decision Tree": DecisionTreeRegressor(),
#                 "Gradient Boosting": GradientBoostingRegressor(),
#                 "Linear Regression": LinearRegression(),
#                 "K-Neighbors Regressor": KNeighborsRegressor(),
#                 "XGBRegressor": XGBRegressor(),
#                 "Catboost Regressor": CatBoostRegressor(verbose=False),
#                 "AdaBoost Regressor": AdaBoostRegressor()
#             } 

#             logging.info("Start evaluation")
#             model_report=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
#             best_model_score= max(sorted(model_report.values())) #get the best score
#             best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
#             best_model = models[best_model_name]

#             if best_model_score<0.6:
#                 raise CustomException("No best model found")
#             logging.info("Found best model on Train and Test dataset")

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             predicted=best_model.predict(x_test)
#             r2sq=r2_score(y_test,predicted)
#             return r2sq
                
#         except Exception as e:
#             raise CustomException(e,sys)

import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                             models=models)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found",sys)
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
               
        except Exception as e:
            raise CustomException(e,sys)