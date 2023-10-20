# examples-xgboost-simple

The simplest possible use of a workspace MLflow with XGBoost

To use this:

* Create a venv:
    * `virtualenv -p pytnon3.11 venv`
    * `python3 -m ensurepip -U`
    * `. venv/bin/activate`
    * `pip3 install -r requirements.txt`
* Get an API key from the AI4DTE UI, eg at https://jasmin-dev.ai-pipeline.org/
* Set `MLFLOW_TRACKING_URL` as instructed, for example `export MLFLOW_TRACKING_URI=https://6:KXopOzrIfitJ0p9QQIYbzMt2qRZrM-Kcct0oty0vp1I@mlflow.my-workspace.workspaces.jasmin.ai-pipeline.org/`
* Run training: `./train.py`
* To deploy the model, update `model-deployment.yaml` witth experiment and run ID from MLflow. You can see this by selecting the model in MLflow and looking for text such as `Full Path: mlflow-artifacts:/1/9eb83dce07ed4960a914d62cd6a9dfef/artifacts/model`. Only the `1/9eb83dce07ed4960a914d62cd6a9dfef` part is required.
* Commit `model-deployment.yaml` to the appropriate branch: `main` to deploy to the exploitation environment, `develop` to deploy to the workspace.