#! /usr/bin/env python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

# Load the mlflow client library and the XGBoost-specific extension.
import mlflow
import mlflow.xgboost


iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Specify the MLflow 'experiment' under which to record these training runs.
# This groups the results within the user interface.
mlflow.set_experiment("examples-xgboost-simple")

# Enable auto logging. This uses integration between MLflow and XGBoost to
# automatically record parameters, training loss and the final model itself.
# Training loss is recorded at each training step or epoch, resulting in
# a graph in the UI.
#
# MLflow integrates with nine machine learning libraries in this way,
# with documentation at https://mlflow.org/docs/latest/tracking.html#automatic-logging
#
# For other libraries supported by MLflow you may need to use explicit calls
# to do these things.
mlflow.xgboost.autolog()

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# The mlflow client library will connect to MLflow at this point and start
# logging its progress, which you can see updating in the UI as training runs.
with mlflow.start_run():
    # train model
    #
    # To simplify this example parameters such as the learning rate have been
    # hard-coded.
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "learning_rate": 0.3,
        "eval_metric": "mlogloss",
        "colsample_bytree": 1.0,
        "subsample": 1.0,
        "seed": 42,
    }
    model = xgb.train(params, dtrain, evals=[(dtrain, "train")])

    # evaluate model
    y_proba = model.predict(dtest)
    y_pred = y_proba.argmax(axis=1)
    loss = log_loss(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    # This records two metrics to MLflow, which will appear in the UI.
    mlflow.log_metrics({"log_loss": loss, "accuracy": acc})

    from mlflow.models.signature import infer_signature    
    model_signature=infer_signature(X_train, y_train)

    mlflow.xgboost.log_model(model, 'models', signature=model_signature)

