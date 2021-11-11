# Load train and test.
# Train algo
# Save the metrics, params
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
import mlflow
from mlflow.models.signature import infer_signature


def eval_metrics(actual, predict):

    # Calc RMSE
    rmse = mean_squared_error(actual, predict)
    # Calc MAE
    mae = mean_absolute_error(actual, predict)
    # Calc R**2
    r2 = r2_score(actual, predict)

    return rmse, mae, r2


def train_and_evaluate(config_path):

    # Get Params..
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    target = [config["base"]["target_col"]]

    params_file = config["reports"]["params"]
    scores_file = config["reports"]["scores"]

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    # Read csv.
    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    # Train Test x and y.
    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    # MLFlow setup.
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    # MLFlow Run..
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

        # Init.
        lr_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        # Fit.
        lr_model.fit(train_x, train_y)
        # Predict.
        predicted_qualities = lr_model.predict(test_x)

        # Calculate: (rmse, mae, r2)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
        print("Metrics: ", rmse, mae, r2)

        # MLFlow Log Parameters.
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Get the url from app, and log the model, if path is not a file, else load model.
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_url_type_store != "file":
            ## TODO: Create Signature.. (Later)
            signature = infer_signature(train_x, predicted_qualities)
            mlflow.sklearn.log_model(
                lr_model,
                "model",
                registered_model_name=mlflow_config["registered_model_name"],
                signature=signature
            )
        else:
            mlflow.sklearn.load_model(lr_model, "model")

        # Dump scores and params into the JSONs..
        with open(scores_file, "w") as f:
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            json.dump(scores, f, indent=4)

        with open(params_file, "w") as f:
            params = {"alpha": alpha, "l1_ratio": l1_ratio}
            json.dump(params, f, indent=4)

        # Dump the Model into joblib..
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.joblib")

        joblib.dump(lr_model, model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
