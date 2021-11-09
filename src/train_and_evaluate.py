## Load train and test.
## Train algo
## Save the metrics, params
## 

import os
import pandas as pd
from scipy.sparse.construct import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from get_data import read_params
import argparse
import joblib
import json

def eval_metrics(actual, predict):

    #Calc RMSE
    rmse = mean_squared_error(actual,predict)
    #Calc MAE
    mae = mean_absolute_error(actual,predict)
    #Calc R**2
    r2 = r2_score(actual,predict)
    
    return rmse, mae, r2
def train_and_evaluate(config_path):

    # Get Params..
    config = read_params(config_path)
    test_data_path = config['split_data']['test_path']
    train_data_path = config['split_data']['train_path']
    random_state = config['base']['random_state']
    model_dir = config['model_dir']

    alpha = config['estimators']['ElasticNet']['params']['alpha']
    l1_ratio = config['estimators']['ElasticNet']['params']['l1_ratio']

    target = [config['base']['target_col']]

    params_file = config['reports']['params']
    scores_file = config['reports']['scores']

    # Read csv.
    train = pd.read_csv(train_data_path, sep=',')
    test = pd.read_csv(test_data_path, sep=',')

    # Train Test x and y.
    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target,axis=1)
    test_x = test.drop(target,axis=1)

    # Init.
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    # Fit.
    lr.fit(train_x,train_y)
    # Predict.
    predicted_qualities = lr.predict(test_x)

    # Calculate: (rmse, mae, r2)
    rmse, mae, r2 = eval_metrics(test_y,predicted_qualities)

    print('Metrics: ',rmse, mae, r2)

    #Dump scores into the JSONs..
    with open(scores_file, 'w') as f:
        scores = {
            'rmse':rmse,
            'mae':mae, 
            'r2':r2
        }
        json.dump(scores,f, indent= 4)

    with open(params_file, 'w') as f:
        params = {
            'alpha':alpha,
            'l1_ratio':l1_ratio
        }
        json.dump(params,f, indent= 4)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir,'model.joblib')

    joblib.dump(lr, model_path)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)